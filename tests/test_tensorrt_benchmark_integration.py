"""Actual CPU export/reload/pin integration, not mocked TensorRT/GPU evidence."""

from dataclasses import replace
import json

import pytest

from backend.app.ai.tensorrt_benchmark import load_candidate
from backend.app.core.tensorrt_benchmark import BenchmarkConfig, OUTPUTS, PROBES, sha256
from scripts import prepare_phase17_candidate as preparation


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase17_cpu_integration")
    # Redirect ONLY publication guard to pytest's temporary scope. Preparation
    # still uses the real CPU interpreter/model/cache/exporter, never repo storage.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preparation, "new_candidate_directory", lambda target: root / "research_candidate")
        result = preparation.prepare(root / "research_candidate",
            replace(BenchmarkConfig(), shape=(1, 3, 32, 32), warmup=1, iterations=10, repeats=1))
    return root / "research_candidate", result


def test_full_standard_architecture_cpu_export_and_version_bridge_payload(candidate):
    root, prepared = candidate
    manifest, config, files, probes = load_candidate(root, prepared["manifest_sha256"],
        trusted_local=True, allow_untrained_probe=True)
    assert manifest["parameter_count"] == 12460898
    assert manifest["decoder_channels"] == [128, 64, 32, 16, 16]
    assert manifest["synthetic"] and manifest["decoder_untrained"] and not manifest["training_performed"]
    assert not manifest["production_approved"] and not manifest["fresh_locked_test_used"]
    assert manifest["cpu_export_parity_passed"]
    assert manifest["trace_allow_tf32"] is False
    assert len(files) == 3 and len(probes) == 35
    assert config.shape == (1, 3, 32, 32)
    assert "32x32" in manifest["workload"]
    for name in PROBES:
        assert manifest["cpu_export_parity"][name]["onnx"]["passed"]
        assert manifest["cpu_export_parity"][name]["torchscript"]["passed"]
    assert (root / "reference.ts").stat().st_size > 40_000_000
    import torch
    model = torch.jit.load(str(root / "reference.ts"), map_location="cpu")
    flags = [list(node.inputs())[-1].toIValue() for node in model.inlined_graph.nodes()
             if node.kind() == "aten::_convolution"]
    assert flags and all(flag is False for flag in flags)


@pytest.mark.parametrize("change", ["production_approved", "synthetic", "training_performed", "decoder_untrained",
    "architecture", "cpu_export_parity_passed", "output_names", "taxonomy", "probe_names", "file_traversal", "trace_allow_tf32"])
def test_manifest_provenance_and_contract_tampering_rejected(candidate, tmp_path, change):
    root, _ = candidate
    value = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    if change in ("production_approved", "training_performed", "trace_allow_tf32"):
        value[change] = True
    elif change in ("synthetic", "decoder_untrained", "cpu_export_parity_passed"):
        value[change] = False
    elif change == "architecture":
        value[change] = "other"
    elif change == "file_traversal":
        value["files"]["../model.onnx"] = value["files"].pop("model.onnx")
    else:
        value[change] = []
    data = json.dumps(value).encode()
    (tmp_path / "candidate_manifest.json").write_bytes(data)
    with pytest.raises(ValueError):
        load_candidate(tmp_path, sha256(data), trusted_local=True, allow_untrained_probe=True)


def test_same_bytes_used_for_hash_and_load_without_path_swap(candidate, monkeypatch):
    root, prepared = candidate
    _, _, files, _ = load_candidate(root, prepared["manifest_sha256"], trusted_local=True, allow_untrained_probe=True)
    assert sha256(files["model.onnx"]) == prepared["manifest"]["files"]["model.onnx"]
    assert sha256(files["reference.ts"]) == prepared["manifest"]["files"]["reference.ts"]
