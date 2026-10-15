import hashlib
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from backend.app.ai import segmentation_onnx as module
from backend.app.ai.segmentation_onnx import (OnnxDependencyError, OnnxParityError, OnnxSegmentationRunner,
    export_segmentation_onnx, parity_inputs)
from backend.app.ai.segmentation_trainer import load_segmentation_checkpoint, training_reproducibility
from backend.app.core.segmentation_onnx import OnnxExportConfig
from tests.segmentation_onnx_fixtures import onnx_bundle, runner_for


def test_actual_export_reload_all_outputs_match_source_checkpoint(onnx_bundle):
    bundle = onnx_bundle
    runner = runner_for(bundle)
    with training_reproducibility(16, 1):
        model, metadata = load_segmentation_checkpoint(bundle["source"], trusted_local=True)
        for values in parity_inputs(bundle["config"].input_shape).values():
            expected = model.predict(torch.from_numpy(values))
            actual = runner.predict(values)
            np.testing.assert_allclose(actual.probabilities, expected.probabilities.numpy(), atol=1e-4, rtol=1e-4)
            np.testing.assert_array_equal(actual.mask, expected.mask.numpy())
            np.testing.assert_allclose(actual.confidence, expected.confidence.numpy(), atol=1e-4, rtol=1e-4)
    assert list(runner.manifest.taxonomy.classes) == metadata["taxonomy"]["classes"]
    assert runner.manifest.synthetic and not runner.manifest.production_approved
    assert set(path.name for path in bundle["output"].iterdir()) == {"model.onnx", "manifest.json", "parity_report.json"}


@pytest.mark.parametrize("shape", [(1, 3, 32, 32), (2, 3, 32, 48)])
def test_separate_fixed_shape_exports_including_batch_two(onnx_bundle, tmp_path, shape):
    arguments = {**onnx_bundle["arguments"], "config": OnnxExportConfig(input_shape=shape)}
    result = export_segmentation_onnx(onnx_bundle["source"], output_dir=tmp_path / "new_export", **arguments)
    runner = OnnxSegmentationRunner(tmp_path / "new_export", expected_model_sha256=result["model_sha256"],
        expected_manifest_sha256=result["manifest_sha256"], trusted_local=True)
    assert runner.predict(parity_inputs(shape)["seeded_noise"]).mask.shape == (shape[0], *shape[2:])
    assert result["parity"]["passed"] and len(result["parity"]["cases"]) == 7


@pytest.mark.parametrize("failure", ["untrusted", "wrong_hash", "invalid_hash", "wrong_version", "synthetic_not_allowed", "missing_source", "corrupt_checkpoint", "invalid_config"])
def test_export_preconditions_fail_without_final_artifacts(onnx_bundle, tmp_path, failure):
    arguments = dict(onnx_bundle["arguments"])
    source = onnx_bundle["source"]
    if failure == "untrusted":
        arguments["trusted_local"] = False
    elif failure == "wrong_hash":
        arguments["expected_sha256"] = "0" * 64
    elif failure == "invalid_hash":
        arguments["expected_sha256"] = "bad"
    elif failure == "wrong_version":
        arguments["expected_model_version"] = "other_version"
    elif failure == "synthetic_not_allowed":
        arguments["allow_synthetic"] = False
    elif failure == "missing_source":
        source = tmp_path / "missing.pt"
    elif failure == "corrupt_checkpoint":
        source = tmp_path / "corrupt.pt"
        source.write_bytes(b"not a checkpoint")
        arguments["expected_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        arguments["config"] = arguments["config"].model_copy(update={"input_shape": (1, 3, 1, 1)})
    destination = tmp_path / "not_published"
    with pytest.raises((ValueError, FileNotFoundError, pickle.UnpicklingError)):
        export_segmentation_onnx(source, output_dir=destination, **arguments)
    assert not destination.exists()


def test_existing_output_is_not_overwritten(onnx_bundle):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in onnx_bundle["output"].iterdir()}
    with pytest.raises(FileExistsError):
        export_segmentation_onnx(onnx_bundle["source"], output_dir=onnx_bundle["output"], **onnx_bundle["arguments"])
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in onnx_bundle["output"].iterdir()}


@pytest.mark.parametrize("failure", ["export", "parity"])
def test_native_export_or_parity_failure_never_publishes_candidate(onnx_bundle, tmp_path, failure):
    target = tmp_path / "failed_export"
    name = "torch.onnx.export" if failure == "export" else "backend.app.ai.segmentation_onnx.compare_predictions"
    error = RuntimeError if failure == "export" else OnnxParityError
    with patch(name, side_effect=error("Injected failure")):
        with pytest.raises(error, match="Injected"):
            export_segmentation_onnx(onnx_bundle["source"], output_dir=target, **onnx_bundle["arguments"])
    assert not target.exists()


def test_missing_optional_dependency_has_explicit_action(monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("Injected missing dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(OnnxDependencyError, match="requirements-onnx.txt"):
        module.onnx_dependencies()


@pytest.mark.parametrize("failure", ["trust", "manifest_hash", "model_hash", "model_bytes", "manifest_bytes", "missing_file"])
def test_bundle_reload_failures_are_explicit(onnx_bundle, tmp_path, failure):
    result = onnx_bundle["result"]
    options = dict(expected_model_sha256=result["model_sha256"], expected_manifest_sha256=result["manifest_sha256"], trusted_local=True)
    root = onnx_bundle["output"]
    if failure == "trust":
        options["trusted_local"] = False
    elif failure == "manifest_hash":
        options["expected_manifest_sha256"] = "0" * 64
    elif failure == "model_hash":
        options["expected_model_sha256"] = "0" * 64
    else:
        root = tmp_path / "bad_bundle"
        root.mkdir()
        manifest = (onnx_bundle["output"] / "manifest.json").read_bytes()
        if failure == "manifest_bytes":
            manifest = b"{bad json"
            options["expected_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        (root / "manifest.json").write_bytes(manifest)
        if failure == "model_bytes":
            (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        OnnxSegmentationRunner(root, **options)


def test_metadata_tampering_class_order_is_bound_by_external_manifest_pin(onnx_bundle, tmp_path):
    manifest = json.loads((onnx_bundle["output"] / "manifest.json").read_bytes())
    manifest["taxonomy"]["classes"] = ["background", "stain", "scratch"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        OnnxSegmentationRunner(tmp_path, expected_model_sha256=onnx_bundle["result"]["model_sha256"],
            expected_manifest_sha256=onnx_bundle["result"]["manifest_sha256"], trusted_local=True)


def test_raw_runtime_and_host_runner_reject_different_input_shapes(onnx_bundle):
    runner = runner_for(onnx_bundle)
    wrong = np.zeros((1, 3, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="fixed-shape"):
        runner.predict(wrong)
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
    with pytest.raises(InvalidArgument, match="dimension|dimensions"):
        runner._session.run(None, {"images": wrong})


def test_checked_graph_corruption_or_external_data_rejected(onnx_bundle):
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(onnx_bundle["output"] / "model.onnx")
    graph.graph.initializer[0].data_location = onnx.TensorProto.EXTERNAL
    onnx.external_data_helper.set_external_data(graph.graph.initializer[0], location="../outside.bin")
    with pytest.raises(ValueError, match="External"):
        module._session(graph.SerializeToString(), onnx_bundle["config"], 3)


def test_unrecognized_graph_opset_rejected(onnx_bundle):
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(onnx_bundle["output"] / "model.onnx")
    graph.opset_import[0].version = 18
    with pytest.raises(ValueError, match="opset 17"):
        module._session(graph.SerializeToString(), onnx_bundle["config"], 3)


def test_export_restores_global_rng_threads_and_determinism(onnx_bundle, tmp_path):
    state, threads, deterministic = torch.random.get_rng_state(), torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with patch("torch.onnx.export", side_effect=RuntimeError("Injected failure")):
        with pytest.raises(RuntimeError):
            export_segmentation_onnx(onnx_bundle["source"], output_dir=tmp_path / "failed", **onnx_bundle["arguments"])
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_publish_failure_cleans_only_owned_new_files(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    original = Path.open
    def fail(path, *args, **kwargs):
        if path.name == "manifest.json":
            raise OSError("Injected write failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError):
        module._publish_bundle(root, {"model.onnx": b"new file", "manifest.json": b"manifest"})
    assert not root.exists()


def test_corrupt_graph_with_recomputed_pins_still_fails_validation(onnx_bundle, tmp_path):
    from google.protobuf.message import DecodeError
    graph_bytes = b"corrupt ONNX graph"
    graph_hash = hashlib.sha256(graph_bytes).hexdigest()
    manifest = {**onnx_bundle["result"]["manifest"], "model_sha256": graph_hash}
    manifest_bytes = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    (tmp_path / "model.onnx").write_bytes(graph_bytes)
    with pytest.raises(DecodeError):
        OnnxSegmentationRunner(tmp_path, expected_model_sha256=graph_hash,
            expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), trusted_local=True)


def test_checkpoint_checksum_is_checked_before_deserialization(onnx_bundle, tmp_path):
    with patch("backend.app.ai.segmentation_onnx.load_segmentation_checkpoint") as load:
        with pytest.raises(ValueError, match="SHA-256"):
            export_segmentation_onnx(onnx_bundle["source"], output_dir=tmp_path / "never_created",
                **{**onnx_bundle["arguments"], "expected_sha256": "0" * 64})
    load.assert_not_called()


@pytest.mark.parametrize("relative", ["Data/phase16_forbidden", "test_image/phase16_forbidden", "backend/phase16_forbidden",
    "frontend/phase16_forbidden", "tests/phase16_forbidden", "docs/phase16_forbidden", "config/phase16_forbidden",
    ".git/phase16_forbidden", "storage/references/phase16_forbidden"])
def test_protected_output_scopes_rejected_before_loading_or_writing(onnx_bundle, relative):
    target = Path(__file__).resolve().parents[1] / relative
    assert not target.exists()
    with patch("backend.app.ai.segmentation_onnx.load_segmentation_checkpoint") as load:
        with pytest.raises(ValueError, match="protected"):
            export_segmentation_onnx(onnx_bundle["source"], output_dir=target, **onnx_bundle["arguments"])
    load.assert_not_called()
    assert not target.exists()


def test_pinned_read_honors_size_bound_before_model_loading(tmp_path):
    path = tmp_path / "oversize"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="size bound"):
        module._read_pinned(path, hashlib.sha256(b"12345").hexdigest(), limit=4)


def test_host_input_guard_precedes_runtime_execution(onnx_bundle):
    runner = runner_for(onnx_bundle)
    invalid = np.full(onnx_bundle["config"].input_shape, np.nan, dtype=np.float32)
    with patch.object(runner, "_session") as session:
        with pytest.raises(ValueError, match="fixed-shape"):
            runner.predict(invalid)
        session.run.assert_not_called()
