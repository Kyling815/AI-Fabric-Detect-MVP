import numpy as np
import pytest

from backend.app.ai.segmentation_onnx import OnnxParityError, compare_predictions, parity_inputs, validate_onnx_input
from backend.app.core.segmentation_onnx import OnnxExportConfig, OnnxManifest


@pytest.mark.parametrize("shape", [(1, 3, 32, 32), (1, 3, 35, 49), (2, 3, 32, 48), (4, 3, 512, 512)])
def test_fixed_shape_contract(shape):
    assert OnnxExportConfig(input_shape=shape).input_shape == shape


@pytest.mark.parametrize("shape", [(0, 3, 32, 32), (5, 3, 32, 32), (True, 3, 32, 32), (1, 1, 32, 32),
    (1, 3, 31, 32), (1, 3, 32, 513), (1, 3, 32), (1, 3, 32, 32, 1), (1, 3, 32.0, 32), "1,3,32,32"])
def test_invalid_shapes_rejected(shape):
    with pytest.raises(ValueError):
        OnnxExportConfig(input_shape=shape)


@pytest.mark.parametrize("values", [{"opset_version": 18}, {"opset_version": 17.0}, {"opset_version": True},
    {"atol": 0}, {"atol": 1}, {"atol": float("nan")}, {"rtol": float("inf")}, {"rtol": -1},
    {"atol": True}, {"rtol": "0.0001"}, {"dynamic_axes": True}])
def test_invalid_export_policy_rejected(values):
    with pytest.raises(ValueError):
        OnnxExportConfig(**values)


def manifest_values():
    return dict(model_sha256="1" * 64, source_checkpoint_sha256="2" * 64, source_model_version="fixture_v1",
        synthetic=True, taxonomy={"classes": ["background", "scratch", "stain"]}, config={}, torch_version="2.5.1+cpu",
        onnx_version="1.17.0", onnxruntime_version="1.20.1", parity_passed=True, parity_case_count=7)


def test_manifest_keeps_taxonomy_and_offline_status():
    result = OnnxManifest(**manifest_values())
    assert result.taxonomy.classes == ("background", "scratch", "stain")
    assert not result.production_approved
    assert OnnxManifest.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("changes", [{"model_filename": "../outside.onnx"}, {"production_approved": True},
    {"production_approved": 0}, {"synthetic": "true"}, {"parity_passed": False}, {"parity_case_count": 6},
    {"output_names": ["probabilities", "logits", "mask", "confidence"]}, {"model_sha256": "ABC"},
    {"source_model_version": " "}, {"taxonomy": {"classes": ["scratch", "background"]}}, {"provider": "CUDAExecutionProvider"},
    {"schema_version": True}])
def test_invalid_manifest_rejected(changes):
    with pytest.raises(ValueError):
        OnnxManifest(**{**manifest_values(), **changes})


@pytest.mark.parametrize("case", ["dtype", "rank", "batch", "channels", "size", "nan", "inf", "negative", "above_one", "list"])
def test_invalid_inputs_not_silently_coerced(case):
    config = OnnxExportConfig(input_shape=(1, 3, 32, 32))
    values = np.zeros(config.input_shape, dtype=np.float32)
    bad = {"dtype": values.astype(np.float64), "rank": values[0], "batch": np.repeat(values, 2, axis=0),
        "channels": values[:, :1], "size": values[:, :, :-1], "nan": values + np.nan, "inf": values + np.inf,
        "negative": values - 0.01, "above_one": values + 1.01, "list": values.tolist()}[case]
    with pytest.raises(ValueError, match="fixed-shape"):
        validate_onnx_input(bad, config)


def test_probe_values_are_deterministic_rgb_and_batch_sensitive_noise():
    config = OnnxExportConfig(input_shape=(2, 3, 35, 49))
    first, second = parity_inputs(config.input_shape), parity_inputs(config.input_shape)
    assert len(first) == 7
    for name, value in first.items():
        validate_onnx_input(value, config)
        np.testing.assert_array_equal(value, second[name])
    assert not np.array_equal(first["seeded_noise"][0], first["seeded_noise"][1])
    assert not np.array_equal(first["color_channels"][:, 0], first["color_channels"][:, 1])


def predictions():
    config = OnnxExportConfig(input_shape=(1, 3, 32, 32))
    logits = np.zeros((1, 2, 32, 32), dtype=np.float32)
    probabilities = np.full_like(logits, 0.5)
    mask = np.zeros((1, 32, 32), dtype=np.int64)
    confidence = np.full((1, 32, 32), 0.5, dtype=np.float32)
    return config, [logits, probabilities, mask, confidence]


def test_numerical_parity_and_exact_masks():
    config, expected = predictions()
    actual = [value.copy() for value in expected]
    actual[0] += 1e-5
    result = compare_predictions(expected, actual, config)
    assert result["logits"]["allclose"] and result["mask"]["exact"]


def test_numerical_tolerance_violation_is_failure():
    config, expected = predictions()
    actual = [value.copy() for value in expected]
    actual[0] += 0.01
    with pytest.raises(OnnxParityError, match="logits"):
        compare_predictions(expected, actual, config)


def test_even_numerically_close_argmax_flip_is_not_silently_accepted():
    config, expected = predictions()
    actual = [value.copy() for value in expected]
    actual[1][:, 0] -= 1e-6
    actual[1][:, 1] += 1e-6
    actual[2][:] = 1
    actual[3][:] += 1e-6
    with pytest.raises(OnnxParityError, match="Mask parity"):
        compare_predictions(expected, actual, config)


@pytest.mark.parametrize("failure", ["shape", "dtype", "nan", "probability_sum", "mask", "confidence"])
def test_malformed_prediction_contract_rejected(failure):
    config, expected = predictions()
    actual = [value.copy() for value in expected]
    if failure == "shape":
        actual[0] = actual[0][:, :, :-1]
    elif failure == "dtype":
        actual[2] = actual[2].astype(np.int32)
    elif failure == "nan":
        actual[0][:] = np.nan
    elif failure == "probability_sum":
        actual[1][:] = 0.1
    elif failure == "mask":
        actual[2][:] = 5
    else:
        actual[3][:] = 0.9
    with pytest.raises(ValueError):
        compare_predictions(expected, actual, config)
