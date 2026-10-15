"""CPU-only preparation of a FULL, explicitly UNTRAINED research architecture.

No training, downloads, existing checkpoint mutation, registry or locked data.
Uses the actual ResNet18UNet/ExportPrediction classes, not a replacement model.
TorchScript bridges the existing CPU stack to the isolated GPU stack WITHOUT
installing torchvision/Pydantic/backend dependencies into the GPU environment.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter_ns
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.tensorrt_benchmark import (BenchmarkConfig, PROBES, OUTPUTS, compare_predictions,
    exclusive_json, new_candidate_directory, sha256, summarize_latencies)


@contextmanager
def strict_fp32_trace():
    import torch
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        # aten::_convolution stores allow_tf32 as a trace constant. Disabling
        # it ONLY in the later GPU process cannot override that captured flag.
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def prepare(output_dir, config):
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise ValueError("Prepare with the unchanged CPU .venv, not the isolated GPU venv")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from backend.app.ai.segmentation_model import ResNet18UNet, ARCHITECTURE_VERSION, PREPROCESSING
    from backend.app.ai.segmentation_onnx import ExportPrediction, parity_inputs
    from backend.app.ai.segmentation_trainer import training_reproducibility
    from backend.app.core.segmentation_data import SegmentationTaxonomy

    target = new_candidate_directory(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    with training_reproducibility(17, 2), strict_fp32_trace():
        model = ResNet18UNet(SegmentationTaxonomy(), pretrained=True).eval()
        wrapper = ExportPrediction(model).eval()
        probes = parity_inputs(config.shape)
        expected = {}
        with torch.inference_mode():
            for name, images in probes.items():
                expected[name] = tuple(x.numpy() for x in wrapper(torch.from_numpy(images)))
        onnx_path, jit_path = target / "model.onnx", target / "reference.ts"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with torch.inference_mode():
                example = torch.from_numpy(probes["seeded_noise"])
                torch.onnx.export(wrapper, (example,), str(onnx_path), input_names=["images"],
                    output_names=list(OUTPUTS), opset_version=17, dynamic_axes=None, dynamo=False,
                    external_data=False, training=torch.onnx.TrainingMode.EVAL, export_params=True)
                traced = torch.jit.trace(wrapper, example, check_trace=True)
                torch.jit.save(traced, str(jit_path))
        graph = onnx.load(onnx_path)
        onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(onnx_path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])
        reloaded = torch.jit.load(str(jit_path), map_location="cpu").eval()
        parity = {}
        reference_arrays = {}
        with torch.inference_mode():
            for name in PROBES:
                actual = session.run(list(OUTPUTS), {"images": probes[name]})
                jit_actual = tuple(x.numpy() for x in reloaded(torch.from_numpy(probes[name])))
                parity[name] = {"onnx": compare_predictions(expected[name], actual, config.shape, 2, "FP32"),
                                "torchscript": compare_predictions(expected[name], jit_actual, config.shape, 2, "FP32")}
                for output, values in zip(OUTPUTS, expected[name]):
                    reference_arrays[name + "__" + output] = values
                reference_arrays[name + "__images"] = probes[name]
        if not all(row[k]["passed"] for row in parity.values() for k in ("onnx", "torchscript")):
            raise ValueError("CPU source export parity failed; do not benchmark GPU")
        np.savez(target / "reference_probes.npz", **reference_arrays)
        cpu_benchmarks = {}
        for label, call in (("pytorch_cpu_eager_fp32", lambda: wrapper(torch.from_numpy(probes["seeded_noise"]))),
                             ("onnxruntime_cpu_fp32", lambda: session.run(list(OUTPUTS), {"images": probes["seeded_noise"]}))):
            samples = []
            with torch.inference_mode():
                for _ in range(config.warmup):
                    call()
                for _ in range(config.repeats * config.iterations):
                    start = perf_counter_ns()
                    result = call()
                    samples.append((perf_counter_ns() - start) / 1e6)
                    del result
            cpu_benchmarks[label] = summarize_latencies(samples)
        files = {name: sha256((target / name).read_bytes())
                 for name in ("model.onnx", "reference.ts", "reference_probes.npz")}
        manifest = {"schema_version": 1, "artifact_type": "phase17_full_architecture_research_probe",
            "model_version": target.name, "architecture": ARCHITECTURE_VERSION,
            "preprocessing": PREPROCESSING, "taxonomy": ["background", "defect"],
            "decoder_channels": list(model.decoder_channels), "parameter_count": sum(p.numel() for p in model.parameters()),
            "encoder_weights_sha256": model.encoder_weights_sha256, "seed": 17,
            "synthetic": True, "training_performed": False, "decoder_untrained": True,
            "production_approved": False, "registry_status": "NOT_REGISTERED",
            "accuracy_evaluated": False, "fresh_locked_test_used": False,
            "config": asdict(config), "probe_names": list(PROBES), "output_names": list(OUTPUTS),
            "files": files, "source_torch_version": str(torch.__version__), "source_onnx_version": onnx.__version__,
            "source_ort_version": ort.__version__, "cpu_threads": 2,
            "trace_allow_tf32": False,
            "cpu_export_parity_passed": True, "cpu_export_parity": parity,
            "cpu_benchmarks": cpu_benchmarks,
            "export_warnings": sorted(set(type(w.message).__name__ + ": " + str(w.message) for w in captured)),
            "workload": f"Fixed single-image RGB {config.shape[2]}x{config.shape[3]} research segmentation; ImageNet normalization and all four outputs inside graph. No decode, resize, FAISS, fusion, QC, API, persistence or hybrid end-to-end timing."}
        exclusive_json(target / "candidate_manifest.json", manifest)
    return {"candidate_directory": str(target), "manifest_sha256": sha256((target / "candidate_manifest.json").read_bytes()),
            "manifest": manifest, "phase18_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-probe", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.output_dir, BenchmarkConfig())
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": str(exc),
                          "phase18_started": False}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
