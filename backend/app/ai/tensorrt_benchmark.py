"""Offline TensorRT build/parity/profiling. Never imported by API/runtime.

GPU packages are lazy imports. Full source model comes from a pinned local
TorchScript artifact; torchvision/backend dependencies stay in the CPU venv.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import gc
import json
from pathlib import Path
import platform
import re
import subprocess
import threading
from time import perf_counter_ns

import numpy as np

from backend.app.core.tensorrt_benchmark import (BenchmarkConfig, OUTPUTS, PROBES, TOLERANCES,
    choose_configuration, compare_predictions, exclusive_json, new_candidate_directory,
    read_pinned, sha256, summarize_latencies, validate_images)


class DeviceMemorySampler:
    """Device-wide USED memory samples, including native TRT, NOT an exact peak.

    Native TensorRT allocations are invisible to PyTorch allocator counters.
    Kept OUT of latency measurements (independent resource pass). Global usage
    includes context/libraries/display/other processes and can miss short peaks.
    """

    def __init__(self, interval=0.01):
        self.interval, self.samples, self.errors = interval, [], []
        self.stop = threading.Event()

    def sample(self):
        import torch
        free, total = torch.cuda.mem_get_info(0)
        self.samples.append((perf_counter_ns(), total - free, total))

    def __enter__(self):
        self.sample()
        def poll():
            try:
                while not self.stop.wait(self.interval):
                    self.sample()
            except Exception as exc:
                self.errors.append(str(exc))
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("VRAM sampler did not stop")
        self.sample()
        if self.errors:
            raise RuntimeError("VRAM sampling failed: " + " | ".join(self.errors))

    def result(self):
        return {"method": "torch.cuda.mem_get_info device-wide polling, independent resource pass",
            "sample_interval_requested_ms": self.interval * 1000, "sample_count": len(self.samples),
            "max_observed_sample_gap_ms": max((b[0] - a[0]) / 1e6 for a, b in zip(self.samples, self.samples[1:])),
            "baseline_device_used_bytes": self.samples[0][1],
            "sampled_peak_device_used_bytes": max(s[1] for s in self.samples),
            "min_sampled_free_bytes": min(s[2] - s[1] for s in self.samples),
            "total_device_bytes": self.samples[0][2], "exact_peak": False,
            "includes_other_processes_context_and_libraries": True}


def measure(call, config, *, cuda, mode):
    import torch
    samples, repeats = [], []
    with torch.inference_mode():
        for _ in range(config.warmup):
            output = call()
            if cuda:
                torch.cuda.synchronize()
            del output
        for _ in range(config.repeats):
            block = []
            for _ in range(config.iterations):
                if cuda:
                    torch.cuda.synchronize()
                start = perf_counter_ns()
                output = call()
                if cuda:
                    torch.cuda.synchronize()
                block.append((perf_counter_ns() - start) / 1e6)
                del output
            repeats.append(summarize_latencies(block))
            samples.extend(block)
    return {**summarize_latencies(samples), "repeats": repeats, "mode": mode,
            "warmup_iterations": config.warmup, "cuda_synchronized": cuda,
            "throughput_mode": "serial single-request effective throughput, not saturated concurrency"}


def _inspect_precision(inspector_json):
    """Count actual inspector tensor format descriptors, NOT builder flags."""
    parsed = json.loads(inspector_json)
    descriptors = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() == "format/datatype" and isinstance(item, str):
                    descriptors.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(parsed)
    half = sum(bool(re.search(r"Half|FP16|Float16", s, re.I)) for s in descriptors)
    return {"tensor_format_descriptors": len(descriptors), "fp16_tensor_descriptors": half,
            "unique_tensor_formats": sorted(set(descriptors))}


def build_engine(graph_bytes, precision, config):
    import tensorrt as trt
    if precision not in TOLERANCES:
        raise ValueError("Unsupported precision")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("TensorRT builder unavailable; no fallback")
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if precision == "FP32" else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(graph_bytes):
        raise RuntimeError("ONNX parser failed: " + " | ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    settings = builder.create_builder_config()
    settings.clear_flag(trt.BuilderFlag.TF32)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_mib * 1024 * 1024)
    settings.builder_optimization_level = config.builder_level
    settings.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    protected = []
    if precision == "FP16_MIXED":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("GPU has no fast FP16; no alternate backend")
        settings.set_flag(trt.BuilderFlag.FP16)
        settings.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # Predetermined mixed policy: sensitive normalization/reduction/softmax
        # retain FP32. No parameter changes and no post-hoc tolerance relaxation.
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # ONNX parser may decompose Softmax into ELEMENTWISE/UNARY layers.
            # Protect their NAMED subgraph, not just a LayerType.SOFTMAX flag.
            protected_subgraph = (layer.name.startswith(("/Softmax", "/Cast", "/ReduceMax", "/ArgMax", "/model/head/"))
                                  or "InstanceNormalization" in layer.name)
            if (layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.REDUCE, trt.LayerType.SOFTMAX)
                    or protected_subgraph) and layer.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in (trt.float32, trt.float16):
                        layer.set_output_type(j, trt.float32)
                protected.append(layer.name)
    begin = perf_counter_ns()
    with DeviceMemorySampler() as memory:
        serialized = builder.build_serialized_network(network, settings)
    seconds = (perf_counter_ns() - begin) / 1e9
    if serialized is None:
        raise RuntimeError("TensorRT build failed; no CPU fallback")
    result = bytes(serialized)
    metadata = {"precision": precision, "strongly_typed": precision == "FP32", "tf32_enabled": False,
        "workspace_mib": config.workspace_mib, "builder_optimization_level": config.builder_level,
        "build_seconds": seconds, "requested_fp32_layer_constraints": protected, "build_vram": memory.result(),
        "engine_bytes": len(result), "engine_sha256": sha256(result)}
    del serialized, parser, network, settings, builder
    return result, metadata


class TensorRTRunner:
    """Fixed float32 I/O, mixed internal FP16 allowed only if inspected."""

    def __init__(self, engine_bytes, shape, classes=2):
        import torch
        import tensorrt as trt
        self.shape, self.classes = tuple(shape), classes
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("TensorRT deserialize failed")
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != {"images", *OUTPUTS}:
            raise ValueError("Unexpected TensorRT names/count")
        batch, _, height, width = shape
        expected_shapes = {"images": tuple(shape), "logits": (batch, classes, height, width),
            "probabilities": (batch, classes, height, width), "mask": (batch, height, width),
            "confidence": (batch, height, width)}
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context unavailable")
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        with torch.cuda.stream(self.stream):
            for name in expected_shapes:
                dtype = trt.int64 if name == "mask" else trt.float32
                if (tuple(self.context.get_tensor_shape(name)) != expected_shapes[name]
                        or self.engine.get_tensor_dtype(name) != dtype
                        or self.engine.get_tensor_mode(name) != (trt.TensorIOMode.INPUT if name == "images" else trt.TensorIOMode.OUTPUT)
                        or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                        or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                    raise ValueError("TensorRT fixed I/O contract mismatch: " + name)
                self.buffers[name] = torch.empty(expected_shapes[name], device="cuda:0",
                    dtype=torch.int64 if name == "mask" else torch.float32)
                if not self.context.set_tensor_address(name, self.buffers[name].data_ptr()):
                    raise RuntimeError("Failed tensor address: " + name)
        self.stream.synchronize()

    def enqueue(self):
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("TensorRT CUDA execution failed; stop")

    def predict(self, images):
        import torch
        images = validate_images(images, self.shape)
        with torch.cuda.stream(self.stream):
            self.buffers["images"].copy_(torch.from_numpy(images), non_blocking=False)
            self.enqueue()
        self.stream.synchronize()
        return tuple(self.buffers[name].cpu().numpy().copy() for name in OUTPUTS)

    def inspect(self):
        import tensorrt as trt
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.context
        data = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        summary = _inspect_precision(data)
        del inspector
        return data, summary

    def close(self):
        self.stream.synchronize()
        # TensorRT child objects must die before their owning engine/runtime.
        del self.context, self.engine, self.runtime, self.buffers, self.stream


def load_candidate(directory, manifest_pin, *, trusted_local, allow_untrained_probe):
    if trusted_local is not True or allow_untrained_probe is not True:
        raise ValueError("Explicit trusted-local AND untrained-research acknowledgement required")
    directory = Path(directory)
    raw = read_pinned(directory / "candidate_manifest.json", manifest_pin, limit=2 * 1024 * 1024)
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1
            or manifest.get("artifact_type") != "phase17_full_architecture_research_probe"
            or manifest.get("architecture") != "independent_resnet18_unet_v1"
            or manifest.get("production_approved") is not False or manifest.get("synthetic") is not True
            or manifest.get("training_performed") is not False or manifest.get("decoder_untrained") is not True
            or manifest.get("trace_allow_tf32") is not False
            or manifest.get("cpu_export_parity_passed") is not True
            or manifest.get("output_names") != list(OUTPUTS) or manifest.get("probe_names") != list(PROBES)
            or manifest.get("taxonomy") != ["background", "defect"]):
        raise ValueError("Unsupported/unverified candidate provenance or I/O contract")
    if set(manifest["files"]) != {"model.onnx", "reference.ts", "reference_probes.npz"}:
        raise ValueError("Unexpected candidate files")
    files = {name: read_pinned(directory / name, digest) for name, digest in manifest["files"].items()}
    config_values = manifest["config"].copy()
    config_values["shape"] = tuple(config_values["shape"])
    config = BenchmarkConfig(**config_values)
    import onnx
    graph = onnx.load_model_from_string(files["model.onnx"])
    if (any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer)
            or any(n.domain not in {"", "ai.onnx"} for n in graph.graph.node)
            or [(o.domain, o.version) for o in graph.opset_import] != [("", 17)]
            or len(graph.graph.input) != 1 or graph.graph.input[0].name != "images"
            or tuple(o.name for o in graph.graph.output) != OUTPUTS
            or graph.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or tuple(d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim) != config.shape):
        raise ValueError("Only pinned self-contained standard opset17 fixed float32 graphs supported")
    onnx.checker.check_model(graph, full_check=True)
    with np.load(BytesIO(files["reference_probes.npz"]), allow_pickle=False) as archive:
        expected_names = {probe + "__" + name for probe in PROBES for name in ("images", *OUTPUTS)}
        if set(archive.files) != expected_names:
            raise ValueError("Unexpected reference probe contract")
        probes = {name: archive[name].copy() for name in archive.files}
    for probe in PROBES:
        validate_images(probes[probe + "__images"], config.shape)
    return manifest, config, files, probes


def benchmark(directory, manifest_pin, output_dir, *, trusted_local=False, allow_untrained_probe=False):
    import sys
    from scripts.check_phase17_environment import validate_environment
    validate_environment(sys.prefix, sys.base_prefix)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    import tensorrt as trt
    if not trt.__version__.startswith("10.13."):
        raise RuntimeError("Use the reviewed TensorRT10.13 environment")
    target = new_candidate_directory(output_dir)
    manifest, config, files, probes = load_candidate(directory, manifest_pin,
        trusted_local=trusted_local, allow_untrained_probe=allow_untrained_probe)
    target.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()
        properties = torch.cuda.get_device_properties(0)
        source = torch.jit.load(BytesIO(files["reference.ts"]), map_location="cpu").eval()
        convolution_flags = [list(node.inputs())[-1].toIValue() for node in source.inlined_graph.nodes()
                             if node.kind() == "aten::_convolution"]
        if not convolution_flags or any(value is not False for value in convolution_flags):
            raise ValueError("TorchScript convolution TF32 was captured enabled; rebuild trace with TF32 disabled")
        # CPU reference exported with torch2.5.1, loaded on torch2.8: verify
        # version bridge before any TensorRT build. NO torchvision migration.
        bridge_parity = {}
        with torch.inference_mode():
            for name in PROBES:
                expected = tuple(probes[name + "__" + key] for key in OUTPUTS)
                actual = tuple(x.numpy() for x in source(torch.from_numpy(probes[name + "__images"])))
                bridge_parity[name] = compare_predictions(expected, actual, config.shape, 2, "FP32")
        if not all(row["passed"] for row in bridge_parity.values()):
            raise ValueError("Torch version bridge parity failed; stop before TRT build")
        images = probes["seeded_noise__images"]
        source.cuda()
        gpu_input = torch.from_numpy(images).cuda()
        gpu_parity = {}
        with torch.inference_mode():
            for name in PROBES:
                expected = tuple(probes[name + "__" + key] for key in OUTPUTS)
                actual = tuple(x.cpu().numpy() for x in source(torch.from_numpy(probes[name + "__images"]).cuda()))
                gpu_parity[name] = compare_predictions(expected, actual, config.shape, 2, "FP32")
        gpu_reference = {"parity_passed": all(r["passed"] for r in gpu_parity.values()), "parity": gpu_parity,
            "device_resident": measure(lambda: source(gpu_input), config, cuda=True, mode="GPU-resident I/O, synchronized host wall"),
            "host_roundtrip": measure(lambda: tuple(x.cpu().numpy() for x in source(torch.from_numpy(validate_images(images, config.shape)).cuda())),
                                      config, cuda=True, mode="CPU tensor H2D + full model + all four outputs D2H")}
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), DeviceMemorySampler() as memory:
            for _ in range(50):
                output = source(gpu_input)
                torch.cuda.synchronize()
                del output
        gpu_reference["vram"] = {**memory.result(), "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        if not gpu_reference["parity_passed"]:
            raise ValueError("GPU PyTorch FP32 baseline parity failed; stop before TensorRT comparison")
        del source, gpu_input
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        results = []
        for precision in ("FP32", "FP16_MIXED"):
            free, total = torch.cuda.mem_get_info()
            if free < 1024 * 1024 * 1024:
                raise RuntimeError("Less than 1GiB free before build; stop rather than risk 4GB exhaustion")
            print("Building full segmentation engine " + precision, file=sys.stderr, flush=True)
            engine_bytes, metadata = build_engine(files["model.onnx"], precision, config)
            engine_path = target / (precision.lower() + ".engine")
            with engine_path.open("xb") as stream:
                stream.write(engine_bytes)
            runner = TensorRTRunner(read_pinned(engine_path, metadata["engine_sha256"]), config.shape)
            try:
                inspected, formats = runner.inspect()
                with (target / (precision.lower() + "_inspector.json")).open("x", encoding="utf-8") as stream:
                    stream.write(inspected)
                parity = {}
                for name in PROBES:
                    expected = tuple(probes[name + "__" + key] for key in OUTPUTS)
                    parity[name] = compare_predictions(expected, runner.predict(probes[name + "__images"]),
                                                        config.shape, 2, precision)
                # Benchmark even rejected parity configurations for diagnosis,
                # but selection NEVER recommends them and registry stays untouched.
                runner.predict(images)
                resident = measure(runner.enqueue, config, cuda=True, mode="GPU-resident reusable I/O, synchronized host wall")
                roundtrip = measure(lambda: runner.predict(images), config, cuda=True,
                                    mode="CPU tensor H2D + full model + all four outputs D2H")
                torch.cuda.reset_peak_memory_stats()
                with DeviceMemorySampler() as memory:
                    for _ in range(50):
                        runner.enqueue()
                        runner.stream.synchronize()
                vram = {**memory.result(), "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "trt_context_device_memory_bytes": runner.engine.device_memory_size_v2,
                    "torch_counters_exclude_native_trt": True}
                verified = formats["tensor_format_descriptors"] > 0 and (formats["fp16_tensor_descriptors"] > 0
                    if precision == "FP16_MIXED" else formats["fp16_tensor_descriptors"] == 0)
                result = {**metadata, "precision_inspection": formats, "precision_verified": verified,
                    "engine_reloaded": True, "actual_gpu_inference": True,
                    "parity_passed": all(row["passed"] for row in parity.values()), "parity": parity,
                    "device_resident": resident, "host_roundtrip": roundtrip, "inference_vram": vram,
                    "memory_headroom_passed": vram["min_sampled_free_bytes"] >= 768 * 1024 * 1024,
                    "status": "MEASURED_NOT_RELEASE_APPROVED"}
                results.append(result)
                exclusive_json(target / (precision.lower() + "_result.json"), result)
                print(precision + " parity=" + str(result["parity_passed"]) + " p95_resident_ms="
                      + str(resident["p95_ms"]), file=sys.stderr, flush=True)
            finally:
                runner.close()
                del runner
            del engine_bytes
            gc.collect()
            torch.cuda.empty_cache()
        selected = choose_configuration(results)
        report = {"phase": 17, "scope": "FULL_ARCHITECTURE_RESEARCH_BENCHMARK_NOT_DETECTION_ACCURACY",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": {k: v for k, v in manifest.items() if k not in {"cpu_benchmarks", "cpu_export_parity"}},
            "candidate_manifest_sha256": manifest_pin, "candidate_directory": str(Path(directory).resolve()),
            "output_directory": str(target), "config": asdict(config), "tolerances_fixed_before_measurement": TOLERANCES,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                "cuda_build": torch.version.cuda, "tensorrt": trt.__version__, "gpu": properties.name,
                "compute_capability": [properties.major, properties.minor], "total_device_bytes": properties.total_memory,
                "nvidia_smi_gpu_driver_vram": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], text=True, timeout=20).strip()},
            "source_cpu_export_parity_passed": True, "cpu_version_bridge_parity": bridge_parity,
            "cpu_baselines": manifest["cpu_benchmarks"], "pytorch_cuda_fp32": gpu_reference, "tensorrt": results,
            "selected_research_configuration": selected, "at_least_one_engine_eligible": selected is not None,
            "selection_rule": "Exact masks + fixed float tolerances + verified precision + >=768MiB sampled headroom; then lowest host-roundtrip P95",
            "batch_2_or_4_tested": False, "batch_policy": "Batch1 matches single-image API; batch2/4 deferred, no concurrent anomaly GPU residency tested",
            "hybrid_pipeline_benchmarked": False, "anomaly_faiss_converted": False,
            "model_training_performed": False, "real_data_accuracy_evaluated": False, "production_approved": False,
            "production_registry_modified": False, "existing_artifacts_overwritten": False, "phase18_started": False,
            "status": "MEASURED_WAITING_FOR_USER_REVIEW" if selected else "NO_PARITY_ELIGIBLE_ENGINE_STOP"}
        exclusive_json(target / "benchmark_report.json", report)
        return report
    except Exception as exc:
        exclusive_json(target / "failure_report.json", {"phase": 17, "passed": False,
            "error_type": type(exc).__name__, "message": str(exc),
            "production_approved": False, "phase18_started": False,
            "partial_artifacts_retained": True, "automatic_fallback": False})
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn
        torch.backends.cudnn.benchmark = old_benchmark
