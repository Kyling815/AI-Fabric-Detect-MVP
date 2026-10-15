"""Isolated GPU/TensorRT setup smoke ONLY. Never benchmark project models.

Batch=1, tiny random Conv+ReLU, FP32 then a strongly-typed FP16 graph. No model
checkpoints, registry, datasets, training, throughput/latency or persistent engine.
Must run with .venv-tensorrt/Scripts/python.exe, not the production CPU .venv.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = (1, 3, 16, 16)
OUTPUT_SHAPE = (1, 4, 14, 14)


def validate_environment(prefix, base_prefix, project_root=PROJECT_ROOT):
    expected = (Path(project_root) / ".venv-tensorrt").resolve()
    if Path(prefix).resolve() != expected or Path(prefix).resolve() == Path(base_prefix).resolve():
        raise ValueError("Use .venv-tensorrt/Scripts/python.exe; the existing CPU .venv must not be changed")
    config = (expected / "pyvenv.cfg").read_text(encoding="utf-8")
    if "include-system-site-packages = false" not in config.lower():
        raise ValueError("GPU smoke requires an isolated venv without inherited site-packages")


def toy_arrays(precision):
    import numpy as np
    if precision not in {"FP32", "FP16"}:
        raise ValueError("Only FP32/FP16 smoke is supported")
    dtype = np.float32 if precision == "FP32" else np.float16
    random = np.random.default_rng(17)
    return ((random.random(INPUT_SHAPE, dtype=np.float32) * 0.5).astype(dtype),
            (random.uniform(-0.125, 0.125, (4, 3, 3, 3))).astype(dtype),
            np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype))


def toy_onnx(precision):
    import onnx
    from onnx import helper, numpy_helper
    images, weight, bias = toy_arrays(precision)
    dtype = onnx.TensorProto.FLOAT if precision == "FP32" else onnx.TensorProto.FLOAT16
    graph = helper.make_graph([
        helper.make_node("Conv", ["images", "weight", "bias"], ["features"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["features"], ["output"])], "phase17_tiny_setup_smoke_not_fabric_model",
        [helper.make_tensor_value_info("images", dtype, INPUT_SHAPE)],
        [helper.make_tensor_value_info("output", dtype, OUTPUT_SHAPE)],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def compare_smoke(expected, actual, precision):
    import numpy as np
    if precision not in {"FP32", "FP16"}:
        raise ValueError("Unsupported precision")
    tolerance = 1e-5 if precision == "FP32" else 1e-3
    expected_dtype = np.float32 if precision == "FP32" else np.float16
    if (expected.shape != OUTPUT_SHAPE or actual.shape != OUTPUT_SHAPE or actual.dtype != expected_dtype
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or not np.allclose(actual.astype(np.float32), expected, atol=tolerance, rtol=tolerance)):
        raise ValueError("Tiny GPU smoke numerical check failed for " + precision)
    return {"passed": True, "atol": tolerance, "rtol": tolerance,
            "max_abs_error": float(np.abs(actual.astype(np.float32) - expected).max())}


def build_and_run_toy(precision, path, expected):
    import torch
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("TensorRT Builder unavailable; stop setup, no fallback")
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    graph_bytes = path.read_bytes()
    if not parser.parse(graph_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Tiny ONNX parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 * 1024 * 1024)
    config.builder_optimization_level = 0
    config.clear_flag(trt.BuilderFlag.TF32)
    # FP16 is expressed in graph tensor/weight types, not guessed from a flag
    # that could allow an all-FP32 engine. No weakly-typed precision fallback.
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Tiny TensorRT engine build failed; stop, no alternate backend")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Tiny engine reload failed")
    if engine.num_io_tensors != 2 or {engine.get_tensor_name(i) for i in range(2)} != {"images", "output"}:
        raise ValueError("Unexpected engine I/O")
    expected_type = trt.float32 if precision == "FP32" else trt.float16
    if any(engine.get_tensor_dtype(name) != expected_type for name in ("images", "output")):
        raise ValueError("Engine I/O precision mismatch")
    context = engine.create_execution_context()
    if context is None or tuple(context.get_tensor_shape("output")) != OUTPUT_SHAPE:
        raise ValueError("Tiny engine execution context/shape failed")
    images, _, _ = toy_arrays(precision)
    dtype = torch.float32 if precision == "FP32" else torch.float16
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        gpu_input = torch.from_numpy(images).to(device="cuda:0", dtype=dtype).contiguous()
        gpu_output = torch.empty(OUTPUT_SHAPE, device="cuda:0", dtype=dtype)
        if not context.set_tensor_address("images", gpu_input.data_ptr()) or not context.set_tensor_address("output", gpu_output.data_ptr()):
            raise RuntimeError("Tiny engine tensor binding failed")
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("Tiny engine GPU inference failed")
    stream.synchronize()
    actual = gpu_output.cpu().numpy()
    result = compare_smoke(expected, actual, precision)
    result.update(precision=precision, batch_size=1, input_shape=list(INPUT_SHAPE), output_shape=list(OUTPUT_SHAPE),
        graph_sha256=hashlib.sha256(graph_bytes).hexdigest(), engine_sha256=hashlib.sha256(bytes(serialized)).hexdigest(),
        engine_bytes=len(bytes(serialized)), strongly_typed=True, tf32_enabled=False, workspace_limit_mib=64,
        engine_reloaded=True, gpu_execution=True, benchmark_performed=False)
    # Destroy TensorRT children before their owners, then release GPU buffers
    # before the next precision probe on this 4 GB device.
    del context, engine, runtime, parser, network, config, builder, serialized, gpu_output, gpu_input, stream
    return result


def smoke():
    validate_environment(sys.prefix, sys.base_prefix)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1 or torch.version.cuda is None:
        raise RuntimeError("CUDA device not available in isolated environment; stop, no CPU fallback")
    properties = torch.cuda.get_device_properties(0)
    if properties.major < 7 or (properties.major == 7 and properties.minor < 5):
        raise RuntimeError("Unsupported GPU compute capability")
    previous_matmul, previous_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        cpu_expected, torch_checks = {}, []
        for precision in ("FP32", "FP16"):
            images, weight, bias = toy_arrays(precision)
            expected = torch.relu(torch.nn.functional.conv2d(torch.from_numpy(images.astype(np.float32)),
                torch.from_numpy(weight.astype(np.float32)), torch.from_numpy(bias.astype(np.float32)))).numpy()
            actual = torch.relu(torch.nn.functional.conv2d(torch.from_numpy(images).cuda(),
                torch.from_numpy(weight).cuda(), torch.from_numpy(bias).cuda())).cpu().numpy()
            cpu_expected[precision] = expected
            torch_checks.append({"precision": precision, **compare_smoke(expected, actual, precision)})
        with tempfile.TemporaryDirectory(prefix="fabric_phase17_setup_smoke_") as directory:
            root = Path(directory)
            for precision in ("FP32", "FP16"):
                onnx.save(toy_onnx(precision), root / (precision + ".onnx"))
            reloaded = onnx.load(root / "FP32.onnx")
            onnx.checker.check_model(reloaded, full_check=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(reloaded.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
            cpu_output = session.run(["output"], {"images": toy_arrays("FP32")[0]})[0]
            onnx_check = compare_smoke(cpu_expected["FP32"], cpu_output, "FP32")
            # Torch has loaded its officially bundled CUDA runtime DLLs first.
            # No DLL copying/renaming, persistent PATH edits or driver changes.
            import tensorrt as trt
            if not trt.__version__.startswith("10.13."):
                raise RuntimeError("Unexpected TensorRT version for this pinned setup")
            engine_checks = []
            for precision in ("FP32", "FP16"):
                engine_checks.append(build_and_run_toy(precision, root / (precision + ".onnx"), cpu_expected[precision]))
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        return {"phase": 17, "scope": "ENVIRONMENT_SETUP_AND_TINY_SMOKE_ONLY_NOT_MAIN_MODEL_BENCHMARK",
            "all_passed": True, "environment": {"executable": sys.executable, "prefix": sys.prefix,
                "python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                "cuda_available": True, "cuda_device_count": torch.cuda.device_count(), "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor], "total_vram_bytes": properties.total_memory,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__, "tensorrt": trt.__version__},
            "smoke": {"cuda_device_detected": True, "pytorch_cuda_inference": torch_checks,
                "onnx_load_and_cpu_inference": onnx_check, "tensorrt_import": True, "tiny_engine_build_reload_inference": engine_checks},
            "model_artifacts_modified": False, "main_model_loaded": False, "main_model_benchmark_performed": False,
            "batch_2_or_4_tested": False, "latency_p50_ms": None, "latency_p95_ms": None,
            "throughput_per_second": None, "peak_vram_usage_bytes": None,
            "permanent_engine_written": False, "phase17_benchmark_started": False, "phase18_started": False,
            "status": "SETUP_SMOKE_COMPLETE_WAITING_FOR_USER_REVIEW"}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_matmul, previous_cudnn
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    try:
        result = smoke()
    except Exception as exc:
        print(json.dumps({"phase": 17, "all_passed": False, "status": "SETUP_SMOKE_FAILED_STOP",
            "error_type": type(exc).__name__, "message": str(exc), "main_model_benchmark_performed": False,
            "phase18_started": False}, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
