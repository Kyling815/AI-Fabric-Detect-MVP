"""Offline Phase 16 supervised export. --demo uses temporary SYNTHETIC weights.

Real export requires an explicit trusted checkpoint, SHA pin, model version and
NEW output directory. Never downloads/trains/activates a candidate or touches
anomaly artifacts. Optional dependencies: requirements-onnx.txt.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.segmentation_onnx import OnnxSegmentationRunner, export_segmentation_onnx, parity_inputs
from backend.app.core.segmentation_onnx import OnnxExportConfig


def demo():
    import torch
    from backend.app.ai.segmentation_model import ResNet18UNet
    from backend.app.ai.segmentation_trainer import save_segmentation_checkpoint, training_reproducibility
    from backend.app.core.segmentation_training import SegmentationBaselineConfig
    results = []
    with tempfile.TemporaryDirectory(prefix="fabric_phase16_demo_") as directory:
        root = Path(directory)
        source = root / "synthetic_UNTRAINED_runtime_probe_v1"
        source.mkdir()
        checkpoint = source / "fixture.pt"
        baseline = SegmentationBaselineConfig()
        with training_reproducibility(16, 1):
            # Independent pretrained encoder with randomly initialized standard
            # decoder. This is a software fixture, NOT a trained candidate.
            model = ResNet18UNet(baseline.pipeline.taxonomy, pretrained=True).eval()
            save_segmentation_checkpoint(checkpoint, model, config=baseline, epoch=1, validation_loss=1.0,
                dataset_fingerprint=hashlib.sha256(b"phase16_synthetic_software_probe_no_training").hexdigest(), synthetic=True)
            expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            del model
        for index, shape in enumerate(((1, 3, 64, 64), (1, 3, 35, 49), (2, 3, 32, 48))):
            config = OnnxExportConfig(input_shape=shape)
            target = root / f"export_{index}"
            result = export_segmentation_onnx(checkpoint, output_dir=target, expected_sha256=expected_hash,
                expected_model_version=source.name, trusted_local=True, allow_synthetic=True, config=config)
            runner = OnnxSegmentationRunner(target, expected_model_sha256=result["model_sha256"],
                expected_manifest_sha256=result["manifest_sha256"], trusted_local=True)
            prediction = runner.predict(parity_inputs(shape)["seeded_noise"])
            result["reload_inference"] = {"passed": True, "logits_shape": list(prediction.logits.shape),
                "mask_shape": list(prediction.mask.shape), "mask_dtype": str(prediction.mask.dtype)}
            result["output_dir"] = "<temporary_export_removed_after_demo>"
            results.append(result)
    return {"phase": 16, "scope": "SYNTHETIC_SOFTWARE_PARITY_ONLY_UNTRAINED_DECODER",
        "exports": results, "fixed_shape_exports": len(results), "parity_cases": sum(len(r["parity"]["cases"]) for r in results),
        "all_passed": True, "provider": "CPUExecutionProvider", "training_performed": False,
        "permanent_model_artifacts_written": False, "production_registry_modified": False,
        "anomaly_converted": False, "production_approved": False, "next_phase_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--model-version")
    parser.add_argument("--trusted-local", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--shape", type=int, nargs=4, metavar=("N", "C", "H", "W"))
    args = parser.parse_args()
    if args.demo:
        if any((args.output_dir, args.expected_sha256, args.model_version, args.trusted_local, args.allow_synthetic, args.shape)):
            parser.error("--demo is isolated; do not combine with checkpoint export options")
    elif not all((args.output_dir, args.expected_sha256, args.model_version, args.trusted_local)):
        parser.error("Checkpoint export requires --output-dir, --expected-sha256, --model-version and --trusted-local")
    try:
        result = demo() if args.demo else export_segmentation_onnx(args.checkpoint, output_dir=args.output_dir,
            expected_sha256=args.expected_sha256, expected_model_version=args.model_version,
            trusted_local=args.trusted_local, allow_synthetic=args.allow_synthetic,
            config=OnnxExportConfig(input_shape=tuple(args.shape or (1, 3, 64, 64))))
    except Exception as exc:
        # Failure is explicit/nonzero; never label a failed export approved.
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
