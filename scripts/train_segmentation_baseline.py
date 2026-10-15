"""Phase 8 offline CNN training. Default OFF; demo trains TEMP SYNTHETIC fixtures only."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.segmentation_dataset import SegmentationDataset
from backend.app.ai.segmentation_trainer import load_segmentation_checkpoint, train_segmentation_baseline, training_reproducibility
from backend.app.core.segmentation_data import SegmentationManifest
from backend.app.core.segmentation_training import SegmentationBaselineConfig
from scripts.check_segmentation_pipeline import create_demo_manifest


def demo_config() -> SegmentationBaselineConfig:
    values = SegmentationBaselineConfig().model_dump(mode="json")
    values["pipeline"]["training"].update(enabled=True, allow_synthetic=True, image_size=[32, 32], batch_size=3, epochs=2)
    values["pipeline"]["split"].update(group_by_lot=True, group_by_camera=True)
    values.update(device="cpu", decoder_channels=[16, 8, 8, 4, 4])
    return SegmentationBaselineConfig.model_validate(values)


def demo() -> dict:
    with TemporaryDirectory(prefix="fabric-phase8-synthetic-") as directory:
        root = Path(directory)
        manifest = create_demo_manifest(root)
        config = demo_config()  # Real independent pretrained ResNet18, tiny decoder, explicit synthetic gates.
        output_dir = root / "temporary_candidate_v1"
        report = train_segmentation_baseline(manifest, config, output_dir=output_dir, allow_synthetic=True)
        with training_reproducibility(config.seed, config.cpu_threads):
            model, _ = load_segmentation_checkpoint(output_dir / report["best_checkpoint"], trusted_local=True,
                                                     expected_taxonomy=manifest.taxonomy)
            validation = SegmentationDataset(manifest, partition="validation", split_config=config.pipeline.split,
                                             output_size=config.pipeline.training.image_size, allow_synthetic=True)
            image = validation[0]["image"].unsqueeze(0)
            first, second = model.predict(image), model.predict(image)
            if not torch.equal(first.probabilities, second.probabilities):
                raise AssertionError("Evaluation was not deterministic")
            report["prediction_contract"] = {"probability_shape": list(first.probabilities.shape), "mask_shape": list(first.mask.shape),
                                               "confidence_shape": list(first.confidence.shape), "deterministic_evaluation": True,
                                               "confidence_calibrated": False}
        report["artifacts_persisted"] = False
        report["note"] += " Demo checkpoints and fixtures are in a temporary directory, removed on exit."
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/segmentation_baseline_example.json")
    parser.add_argument("--output-dir", type=Path, help="Explicit NEW candidate version directory; existing paths are refused")
    parser.add_argument("--allow-synthetic", action="store_true", help="Research-only opt-in; must also be allowed in config")
    args = parser.parse_args()
    if args.demo:
        if args.output_dir is not None or args.allow_synthetic:
            parser.error("Demo manages its own temporary artifacts and synthetic opt-in")
        report = demo()
    else:
        if args.output_dir is None:
            parser.error("--manifest training requires an explicit --output-dir")
        manifest = SegmentationManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        config = SegmentationBaselineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        report = train_segmentation_baseline(manifest, config, output_dir=args.output_dir,
                                             base_dir=args.manifest.resolve().parent, allow_synthetic=args.allow_synthetic)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
