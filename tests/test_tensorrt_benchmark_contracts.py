"""CPU safety/policy tests. GPU measurements live in the actual benchmark report."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backend.app.core.tensorrt_benchmark import (BenchmarkConfig, OUTPUTS, choose_configuration,
    compare_predictions, exclusive_json, new_candidate_directory, read_pinned, sha256,
    summarize_latencies, validate_images, validate_predictions)
from backend.app.ai.tensorrt_benchmark import _inspect_precision, load_candidate, measure


def predictions(shape=(1, 3, 32, 32)):
    batch, _, height, width = shape
    logits = np.zeros((batch, 2, height, width), np.float32)
    probabilities = np.full_like(logits, 0.5)
    return (logits, probabilities, np.zeros((batch, height, width), np.int64),
            np.full((batch, height, width), 0.5, np.float32))


@pytest.mark.parametrize("updates", [dict(shape=(2, 3, 256, 256)), dict(shape=(1, 1, 256, 256)),
    dict(shape=(1, 3, 16, 32)), dict(shape=(1, 3, 513, 256)), dict(shape=(True, 3, 32, 32)),
    dict(warmup=0), dict(iterations=9), dict(repeats=True), dict(workspace_mib=1024), dict(builder_level=4)])
def test_invalid_benchmark_config(updates):
    with pytest.raises(ValueError):
        BenchmarkConfig(**updates)


def test_default_workload_is_full_resolution_configuration_batch_one():
    config = BenchmarkConfig()
    assert config.shape == (1, 3, 256, 256)
    assert config.iterations * config.repeats == 300 and config.warmup == 10


def test_output_directory_restricts_protected_and_existing_paths(tmp_path):
    allowed = tmp_path / "storage/models/segmentation_candidates"
    for target in (tmp_path, tmp_path / "Data/new", tmp_path / "storage/models/old", allowed):
        with pytest.raises(ValueError):
            new_candidate_directory(target, tmp_path)
    fresh = allowed / "new_version"
    assert new_candidate_directory(fresh, tmp_path) == fresh.resolve()
    fresh.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        new_candidate_directory(fresh, tmp_path)


@pytest.mark.parametrize("pin", ["", "A" * 64, "1" * 63, "z" * 64, None])
def test_hash_pin_must_be_explicit_lowercase(tmp_path, pin):
    with pytest.raises(ValueError):
        read_pinned(tmp_path / "unused", pin)


def test_exact_bytes_are_hashed_before_load_and_size_bounded(tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(b"trusted fixture")
    assert read_pinned(path, sha256(b"trusted fixture")) == b"trusted fixture"
    with pytest.raises(ValueError):
        read_pinned(path, "1" * 64)
    with pytest.raises(ValueError):
        read_pinned(path, sha256(b"trusted fixture"), limit=2)


def test_json_publication_never_overwrites(tmp_path):
    path = tmp_path / "report.json"
    exclusive_json(path, {"production_approved": False})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        exclusive_json(path, {"production_approved": True})
    assert path.read_bytes() == original


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")], [True], ["1"]])
def test_latency_requires_actual_finite_positive_measurements(samples):
    with pytest.raises(ValueError):
        summarize_latencies(samples)


def test_latency_quantiles_and_effective_throughput():
    result = summarize_latencies([1, 2, 3, 4])
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["throughput_images_per_second"] == 400
    assert result["latency_ms"] == [1, 2, 3, 4]


@pytest.mark.parametrize("case", ["shape", "dtype", "nan", "negative", "above_one"])
def test_host_input_guard_is_not_a_traced_python_guard(case):
    images = np.zeros((1, 3, 32, 32), np.float32)
    if case == "shape":
        images = images[:, :, :-1]
    elif case == "dtype":
        images = images.astype(np.float16)
    elif case == "nan":
        images[:] = np.nan
    elif case == "negative":
        images[:] = -1
    else:
        images[:] = 2
    with pytest.raises(ValueError):
        validate_images(images, (1, 3, 32, 32))


def test_good_image_validates_without_mutation():
    images = np.zeros((1, 3, 32, 32), np.float32)
    assert validate_images(images, images.shape) is images


@pytest.mark.parametrize("precision", ["FP32", "FP16_MIXED"])
def test_correct_prediction_parity(precision):
    result = compare_predictions(predictions(), predictions(), (1, 3, 32, 32), 2, precision)
    assert result["passed"] and result["outputs"]["mask"]["mismatched_pixels"] == 0


def test_fp16_float_tolerance_does_not_allow_argmax_flips():
    expected = predictions()
    actual = list(predictions())
    actual[1][:, 0] -= 0.0001
    actual[1][:, 1] += 0.0001
    actual[2][:] = 1
    actual[3][:] = actual[1][:, 1]
    result = compare_predictions(expected, actual, (1, 3, 32, 32), 2, "FP16_MIXED")
    assert not result["passed"]
    assert result["outputs"]["probabilities"]["passed"]
    assert result["outputs"]["mask"]["mismatched_pixels"] == 1024


def test_output_semantic_drift_is_reported_rejected_not_silently_accepted():
    expected = predictions()
    actual = list(predictions())
    actual[1][:] = np.float32(0.5001220703125)
    actual[3][:] = actual[1][:, 0]
    result = compare_predictions(expected, actual, (1, 3, 32, 32), 2, "FP16_MIXED")
    assert not result["passed"]
    assert not result["output_contract"]["checks"]["probability_sum"]
    assert result["output_contract"]["max_probability_sum_error"] > 1e-5


@pytest.mark.parametrize("case", ["count", "shape", "dtype", "nan", "probabilities", "mask", "confidence"])
def test_invalid_outputs_are_not_benchmarked_as_valid(case):
    values = list(predictions())
    if case == "count":
        values.pop()
    elif case == "shape":
        values[0] = values[0][:, :, :-1]
    elif case == "dtype":
        values[0] = values[0].astype(np.float16)
    elif case == "nan":
        values[0][:] = np.nan
    elif case == "probabilities":
        values[1][:] = 0.75
    elif case == "mask":
        values[2][:] = 1
    else:
        values[3][:] = 0.1
    with pytest.raises(ValueError):
        validate_predictions(values, (1, 3, 32, 32), 2)


def test_inspector_precision_requires_actual_tensor_formats_not_fp16_flag():
    parsed = {"Layers": [{"Inputs": [{"Format/Datatype": "Row major linear Half format"}],
                          "Outputs": [{"Format/Datatype": "Row major linear Float format"}]}]}
    assert _inspect_precision(json.dumps(parsed))["fp16_tensor_descriptors"] == 1
    assert _inspect_precision('{"FP16": true}')["fp16_tensor_descriptors"] == 0


def test_fast_rejected_engine_never_selected():
    valid = {"precision": "FP32", "parity_passed": True, "precision_verified": True,
             "memory_headroom_passed": True, "host_roundtrip": {"p95_ms": 10}}
    invalid = {**valid, "precision": "FP16_MIXED", "parity_passed": False, "host_roundtrip": {"p95_ms": 1}}
    assert choose_configuration([valid, invalid]) == "FP32"
    assert choose_configuration([invalid]) is None
    assert choose_configuration([{**valid, "precision_verified": False}]) is None
    assert choose_configuration([{**valid, "memory_headroom_passed": False}]) is None


@pytest.mark.parametrize("trusted,allow", [(False, False), (True, False), (False, True), (1, True)])
def test_explicit_local_trust_and_untrained_acknowledgement_before_file_load(tmp_path, trusted, allow):
    with pytest.raises(ValueError):
        load_candidate(tmp_path, "1" * 64, trusted_local=trusted, allow_untrained_probe=allow)


def test_benchmark_cli_refuses_cpu_venv_before_loading_model():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/benchmark_phase17_tensorrt.py"),
        "--candidate-dir", "unused", "--manifest-sha256", "1" * 64, "--output-dir", "unused",
        "--trusted-local", "--allow-untrained-probe"], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1
    result = json.loads(completed.stderr)
    assert result["status"] == "FAILED_STOP_NO_FALLBACK"
    assert ".venv-tensorrt" in result["message"] and not result["phase18_started"]


def test_cpu_measurement_has_warmup_repeats_and_raw_samples(monkeypatch):
    import itertools
    from backend.app.ai import tensorrt_benchmark
    # A no-op can finish within one Windows timer tick. Use a deterministic
    # UNIT-TEST clock rather than fake/zero performance measurements.
    ticks = itertools.count(step=1_000_000)
    monkeypatch.setattr(tensorrt_benchmark, "perf_counter_ns", lambda: next(ticks))
    calls = []
    config = replace(BenchmarkConfig(), warmup=1, iterations=10, repeats=2)
    result = measure(lambda: calls.append(1), config, cuda=False, mode="CPU test fixture, not performance evidence")
    assert len(calls) == 21 and result["samples"] == 20 and len(result["repeats"]) == 2
    assert not result["cuda_synchronized"]
    assert result["latency_ms"] == [1.0] * 20


def test_strict_fp32_trace_restores_flags_even_after_failure():
    import torch
    from scripts.prepare_phase17_candidate import strict_fp32_trace
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError):
        with strict_fp32_trace():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            raise RuntimeError("fixture failure")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
