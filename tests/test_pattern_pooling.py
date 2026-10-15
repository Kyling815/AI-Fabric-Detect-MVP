from dataclasses import replace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
import torch

from backend.app.ai.dense_feature_extractor import SharedDenseResNet18
from backend.app.ai.pattern_period import PatternPeriodEstimator, PatternPeriodResult
from backend.app.ai.pattern_pooling import PatternAwareDenseExtractor, PoolingPlan, plan_pattern_pooling, pool_native_features
from backend.app.core.pattern_pooling import PatternPoolingConfig
from tests.pattern_period_fixtures import pattern
from tests.pattern_pooling_fixtures import periodic


def plan(period=None, config=None, **kwargs):
    return plan_pattern_pooling(config or PatternPoolingConfig(enabled=True), period,
                                **{"resize_x": 1, "resize_y": 1, "layer2_shape": (40, 40), "layer3_shape": (20, 20), **kwargs})


def test_period_24_pixels_selects_3_cells_at_layer2_stride8():
    value = plan(periodic())
    assert value.representation == "period_aware" and value.fallback_reason is None
    assert value.layer2_kernels == ((3, 3), (7, 7), (13, 13))
    assert value.layer3_kernels == ((3, 3), (3, 3), (7, 7))


def test_source_period_is_scaled_to_processed_pixels_on_each_axis():
    value = plan(periodic(48, 24), resize_x=0.5, resize_y=0.25)
    assert value.layer2_kernels[0] == (1, 3)
    assert value.layer3_kernels[0] == (1, 3)


@pytest.mark.parametrize("shape2,shape3", [((1, 1), (1, 1)), ((2, 4), (1, 2)), ((7, 9), (4, 5)), ((40, 60), (20, 30))])
def test_kernels_are_positive_odd_and_bounded_by_actual_native_grids(shape2, shape3):
    value = plan(periodic(96, 96), layer2_shape=shape2, layer3_shape=shape3)
    for kernels, shape in ((value.layer2_kernels, shape2), (value.layer3_kernels, shape3)):
        assert len(kernels) == 3
        for kernel in kernels:
            assert all(0 < size <= min(bound, 31) and size % 2 == 1 for size, bound in zip(kernel, shape))


@pytest.mark.parametrize("kind", ["uniform", "noise", "rotate", "stripes"])
def test_nonperiodic_low_confidence_or_one_axis_uses_fixed_multiscale(kind):
    period = PatternPeriodEstimator().estimate(pattern(kind))
    value = plan(period)
    assert value.representation == "fixed_multiscale"
    assert value.layer2_kernels == value.layer3_kernels == ((3, 3), (5, 5), (7, 7))


@pytest.mark.parametrize("period", [None, PatternPeriodResult(), replace(periodic(), confidence_y=0.1),
                                   replace(periodic(), period_x_px=float("nan")), replace(periodic(), period_x_px=1e308),
                                   replace(periodic(), period_y_px=0), replace(periodic(), confidence=1.1),
                                   replace(periodic(), period_x_px=True)])
def test_invalid_period_never_forces_adaptive_kernel(period):
    assert plan(period).representation == "fixed_multiscale"


def test_fixed_multiscale_mode_ignores_valid_period_and_off_is_baseline():
    assert plan(periodic(), PatternPoolingConfig(enabled=True, mode="fixed_multiscale")).representation == "fixed_multiscale"
    assert plan(periodic(), PatternPoolingConfig()).representation == "baseline"


@pytest.mark.parametrize("override", [{"resize_x": 0}, {"resize_y": float("nan")}, {"resize_x": 1.1}, {"layer2_shape": (0, 3)}])
def test_invalid_transform_or_native_shapes_rejected(override):
    with pytest.raises(ValueError):
        plan(periodic(), **override)


@pytest.mark.parametrize("shapes", [((1, 2, 7, 9), (1, 3, 4, 5)), ((2, 4, 1, 1), (2, 8, 1, 1)),
                                    ((1, 128, 8, 12), (1, 256, 4, 6))])
def test_pool_aligns_layers_without_channel_or_spatial_shape_mismatch(shapes):
    layer2, layer3 = torch.ones(shapes[0]), torch.ones(shapes[1]) * 2
    value = plan(periodic(), layer2_shape=shapes[0][-2:], layer3_shape=shapes[1][-2:])
    first, second = pool_native_features(layer2, layer3, value), pool_native_features(layer2, layer3, value)
    assert first.shape == (shapes[0][0], shapes[0][1] + shapes[1][1], *shapes[0][-2:])
    assert torch.equal(first, second) and torch.isfinite(first).all()
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=1), torch.ones(first.shape[0], *first.shape[-2:]))
    assert torch.all(layer2 == 1) and torch.all(layer3 == 2)


@pytest.mark.parametrize("bad", ["batch", "nan", "rank", "dtype", "kernel"])
def test_incompatible_maps_or_even_kernels_fail_safely(bad):
    two, three = torch.ones(1, 2, 7, 9), torch.ones(1, 3, 4, 5)
    value = plan(periodic(), layer2_shape=(7, 9), layer3_shape=(4, 5))
    if bad == "batch":
        three = torch.ones(2, 3, 4, 5)
    elif bad == "nan":
        two[0, 0, 0, 0] = float("nan")
    elif bad == "rank":
        two = two.squeeze(0)
    elif bad == "dtype":
        three = three.double()
    else:
        value = replace(value, layer2_kernels=((2, 3),) * 3)
    with pytest.raises(ValueError):
        pool_native_features(two, three, value)


@pytest.mark.parametrize("values", [{"enabled": "true"}, {"mode": "unknown"}, {"motif_scales": []},
                                    {"motif_scales": [2, 1]}, {"motif_scales": [1, 1]}, {"motif_scales": [True]},
                                    {"fallback_kernels": [2]}, {"fallback_kernels": [True]}, {"max_kernel": 2},
                                    {"max_kernel": 3}, {"confidence_threshold": float("nan")}, {"fusion": "concat"}])
def test_invalid_or_unimplemented_config_rejected(values):
    with pytest.raises(ValidationError):
        PatternPoolingConfig.model_validate(values)


@pytest.fixture(scope="module")
def baseline_extractor():
    with torch.random.fork_rng():
        torch.manual_seed(31)
        return SharedDenseResNet18(use_pretrained_weights=False, device="cpu")


def test_off_reproduces_baseline_exactly_and_never_calls_estimator(baseline_extractor, tmp_path):
    image = Image.fromarray(pattern("checker"))
    before = baseline_extractor.extract_image(image, image_max_size=80)
    estimator = Mock(spec=PatternPeriodEstimator)
    estimator.estimate.side_effect = AssertionError("OFF must skip estimator")
    extractor = PatternAwareDenseExtractor(baseline_extractor, estimator=estimator)
    after = extractor.extract_image(image, image_max_size=80)
    np.testing.assert_array_equal(before.embeddings, after.features.embeddings)
    assert after.features.boxes == before.boxes and after.observed_period is None
    assert after.geometry.algorithm == "dense_baseline_v1" and after.pooling_plan.representation == "baseline"
    estimator.estimate.assert_not_called()
    path = tmp_path / "image.png"
    image.save(path)
    np.testing.assert_array_equal(extractor.extract_path(path, image_max_size=80).features.embeddings, before.embeddings)


@pytest.mark.parametrize("kind", ["checker", "noise", "uniform", "stripes"])
def test_on_is_deterministic_dimension_preserving_and_does_not_mutate_model(baseline_extractor, kind):
    extractor = PatternAwareDenseExtractor(baseline_extractor, config=PatternPoolingConfig(enabled=True))
    image = Image.fromarray(pattern(kind))
    before = {name: value.clone() for name, value in baseline_extractor.model.state_dict().items()}
    first, second = extractor.extract_image(image, image_max_size=80), extractor.extract_image(image, image_max_size=80)
    assert first.features.embeddings.shape == (first.features.grid_rows * first.features.grid_cols, 384)
    np.testing.assert_array_equal(first.features.embeddings, second.features.embeddings)
    assert first.geometry.signature == second.geometry.signature
    assert first.pooling_plan.representation == ("period_aware" if kind == "checker" else "fixed_multiscale")
    assert all(torch.equal(value, before[name]) for name, value in baseline_extractor.model.state_dict().items())


def test_estimator_error_falls_back_and_fixed_mode_skips_estimation(baseline_extractor):
    estimator = Mock(spec=PatternPeriodEstimator)
    estimator.estimate.side_effect = RuntimeError("estimator failed")
    config = PatternPoolingConfig(enabled=True)
    result = PatternAwareDenseExtractor(baseline_extractor, config=config, estimator=estimator).extract_image(Image.new("RGB", (32, 32)), image_max_size=32)
    assert result.pooling_plan.representation == "fixed_multiscale" and result.observed_period.reason == "estimator_error"
    estimator.reset_mock()
    PatternAwareDenseExtractor(baseline_extractor, config=config.model_copy(update={"mode": "fixed_multiscale"}), estimator=estimator).extract_image(Image.new("RGB", (32, 32)), image_max_size=32)
    estimator.estimate.assert_not_called()


def test_training_mode_is_rejected_without_changing_weights(baseline_extractor):
    baseline_extractor.model.train()
    try:
        with pytest.raises(ValueError, match="eval-mode"):
            PatternAwareDenseExtractor(baseline_extractor, config=PatternPoolingConfig(enabled=True)).extract_image(Image.new("RGB", (32, 32)), image_max_size=32)
    finally:
        baseline_extractor.model.eval()


def test_existing_384_to_128_adapter_remains_dimension_compatible_and_off_is_exact(baseline_extractor):
    from backend.app.ai.projection_head import ProjectionHead, ProjectionHeadInfo
    with torch.random.fork_rng():
        torch.manual_seed(91)
        adapter = ProjectionHead(input_dim=384, projection_dim=128).eval()
    baseline_extractor.adapter = adapter
    baseline_extractor.adapter_info = ProjectionHeadInfo(384, 128, 256, 0.1, "unit_fixture_only.pt")
    try:
        image = Image.fromarray(pattern("checker"))
        expected = baseline_extractor.extract_image(image, image_max_size=64)
        off = PatternAwareDenseExtractor(baseline_extractor, adapter_version="unit_adapter_v1").extract_image(image, image_max_size=64)
        on = PatternAwareDenseExtractor(baseline_extractor, adapter_version="unit_adapter_v1",
                                       config=PatternPoolingConfig(enabled=True)).extract_image(image, image_max_size=64)
        np.testing.assert_array_equal(off.features.embeddings, expected.embeddings)
        assert on.features.embeddings.shape == off.features.embeddings.shape and on.geometry.feature_dim == 128
        np.testing.assert_allclose(np.linalg.norm(on.features.embeddings, axis=1), 1, atol=1e-5)
        assert on.geometry.adapter_weights_sha256 is not None
    finally:
        baseline_extractor.adapter = None
        baseline_extractor.adapter_info = None
