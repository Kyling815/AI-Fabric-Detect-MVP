"""CPU tests for smoke safety/fixtures only; never substitute for actual GPU smoke."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.check_phase17_environment import (INPUT_SHAPE, OUTPUT_SHAPE, compare_smoke,
                                               toy_arrays, toy_onnx, validate_environment)


def test_existing_cpu_environment_is_refused_before_gpu_imports():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/check_phase17_environment.py"), "--smoke"],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1
    result = json.loads(completed.stderr)
    assert not result["all_passed"]
    assert result["status"] == "SETUP_SMOKE_FAILED_STOP"
    assert ".venv-tensorrt" in result["message"]
    assert not result["main_model_benchmark_performed"] and not result["phase18_started"]


def test_isolated_venv_guard(tmp_path):
    prefix = tmp_path / ".venv-tensorrt"
    prefix.mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    validate_environment(prefix, tmp_path / "base", tmp_path)


@pytest.mark.parametrize("case", ["system_python", "wrong_venv", "inherited_packages"])
def test_nonisolated_venv_guard_rejects(case, tmp_path):
    prefix = tmp_path / ".venv-tensorrt"
    prefix.mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_environment(tmp_path / "other" if case == "wrong_venv" else prefix,
                             prefix if case == "system_python" else tmp_path / "base", tmp_path)


@pytest.mark.parametrize("precision", ["FP32", "FP16"])
def test_tiny_arrays_repeatable_batch_one_and_bounded(precision):
    first, second = toy_arrays(precision), toy_arrays(precision)
    assert first[0].shape == INPUT_SHAPE and INPUT_SHAPE[0] == 1
    assert first[1].shape == (4, 3, 3, 3) and first[2].shape == (4,)
    for left, right in zip(first, second):
        assert left.dtype == (np.float32 if precision == "FP32" else np.float16)
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()


@pytest.mark.parametrize("precision", ["FP32", "FP16"])
def test_toy_graph_is_self_contained_fixed_shape_not_project_model(precision):
    onnx = pytest.importorskip("onnx")
    model = toy_onnx(precision)
    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == ["Conv", "Relu"]
    assert tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim) == INPUT_SHAPE
    assert tuple(d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim) == OUTPUT_SHAPE
    expected = onnx.TensorProto.FLOAT if precision == "FP32" else onnx.TensorProto.FLOAT16
    assert model.graph.input[0].type.tensor_type.elem_type == expected
    assert all(tensor.data_type == expected and not tensor.external_data for tensor in model.graph.initializer)


@pytest.mark.parametrize("precision", ["FP32", "FP16"])
def test_toy_numerical_comparison_passes_correct_outputs(precision):
    expected = np.full(OUTPUT_SHAPE, 0.25, dtype=np.float32)
    actual = expected.astype(np.float32 if precision == "FP32" else np.float16)
    result = compare_smoke(expected, actual, precision)
    assert result["passed"] and result["max_abs_error"] == 0


@pytest.mark.parametrize("case", ["shape", "dtype", "nan", "inf", "mismatch"])
def test_wrong_toy_outputs_fail_not_silently_fallback(case):
    expected = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
    actual = expected.copy()
    if case == "shape":
        actual = actual[:, :, :-1]
    elif case == "dtype":
        actual = actual.astype(np.float16)
    elif case == "nan":
        actual[:] = np.nan
    elif case == "inf":
        actual[:] = np.inf
    else:
        actual[:] = 1
    with pytest.raises(ValueError):
        compare_smoke(expected, actual, "FP32")


def test_no_extra_precision_optimization_modes():
    with pytest.raises(ValueError):
        toy_arrays("INT8")
    with pytest.raises(ValueError):
        compare_smoke(np.zeros(OUTPUT_SHAPE), np.zeros(OUTPUT_SHAPE), "INT8")
