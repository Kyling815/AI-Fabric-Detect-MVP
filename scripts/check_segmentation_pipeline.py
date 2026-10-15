"""Read-only Phase 7 preflight, schemas and SYNTHETIC pipeline demo. No trainer."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.segmentation_dataset import SegmentationDataset, load_segmentation_sample, sha256_file
from backend.app.ai.segmentation_metrics import SegmentationMetrics
from backend.app.ai.segmentation_split import grouped_segmentation_split
from backend.app.core.segmentation_data import (GroupedSplitConfig, SegmentationAugmentationConfig, SegmentationManifest,
                                                SegmentationPipelineConfig, SegmentationSample, SegmentationTaxonomy)


def create_demo_manifest(root: Path) -> SegmentationManifest:
    """Nine tiny isolated fixtures, not training samples added to real datasets."""
    samples = []
    for group in range(3):
        for crop in range(3):
            sample_id = f"demo_{group}_{crop}"
            mask = np.zeros((24, 32), dtype=np.uint8)
            if crop:
                mask[5:15, 7 + crop:12 + crop] = 1
            image = np.full((24, 32, 3), 40 + group * 20 + crop, dtype=np.uint8)
            image[mask == 1] = (200, 40, 20)
            image_path, mask_path = root / (sample_id + ".png"), root / (sample_id + "_mask.png")
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            samples.append(SegmentationSample(sample_id=sample_id, fabric_id="fabric_demo", image_path=str(image_path),
                                               image_sha256=sha256_file(image_path), mask_path=str(mask_path), mask_sha256=sha256_file(mask_path),
                                               label="defect" if crop else "normal", defect_classes=("defect",) if crop else (),
                                               origin="synthetic", annotation_source="synthetic_fixture", source_id=f"capture_{group}",
                                               roll_id=f"roll_{group}", lot_id=f"lot_{group}", camera_id=f"camera_{group}"))
    return SegmentationManifest(dataset_version="phase7_tiny_synthetic_fixture_v1", samples=tuple(samples))


def demo() -> dict:
    with TemporaryDirectory(prefix="fabric-phase7-demo-") as directory:
        manifest = create_demo_manifest(Path(directory))
        config = GroupedSplitConfig(group_by_lot=True, group_by_camera=True)
        split = grouped_segmentation_split(manifest, config)
        batches = {}
        for partition in ("train", "validation", "test"):
            dataset = SegmentationDataset(manifest, partition=partition, split_config=config, output_size=(32, 32), allow_synthetic=True,
                                          augmentation=SegmentationAugmentationConfig(enabled=True))
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
            batches[partition] = {"sample_count": len(dataset), "image_shape": list(batch["image"].shape),
                                  "mask_shape": list(batch["mask"].shape), "image_dtype": str(batch["image"].dtype), "mask_dtype": str(batch["mask"].dtype)}
        metrics = SegmentationMetrics(manifest.taxonomy.classes)
        first = load_segmentation_sample(manifest.samples[1], manifest.taxonomy, allow_synthetic=True)
        metrics.update(first.mask, first.mask)  # Oracle arithmetic self-check, NOT a model prediction.
        return {"phase": 7, "status": "SYNTHETIC_SOFTWARE_CHECK_ONLY", "taxonomy": list(manifest.taxonomy.classes),
                "split": asdict(split), "batches": batches, "oracle_metric_self_check": metrics.compute(),
                "trained_model": False, "production_accuracy_evaluated": False, "dataset_files_modified": False,
                "note": "Tiny fixtures in temp only. Perfect oracle metric checks arithmetic, not trained model quality."}


def audit_existing() -> dict:
    fabrics = []
    for fabric_id in ("fabric_001", "fabric_002"):
        path = PROJECT_ROOT / "storage/metadata/company_demo_manifests" / (fabric_id + "_all.csv")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = [row for row in rows if row["mask_path"] and row["split"] == "adapter_train"][:2]
        pair_checks = []
        for row in selected:
            sample = SegmentationSample(sample_id=row["image_id"], fabric_id=fabric_id, image_path=row["image_path"],
                                         image_sha256=sha256_file(Path(row["image_path"])), mask_path=row["mask_path"],
                                         mask_sha256=sha256_file(Path(row["mask_path"])), label="defect", defect_classes=("defect",),
                                         mask_encoding="binary_255", binary_class_name="defect", origin=row["origin"],
                                         annotation_source="existing_binary", source_id=row["source_original"], split="excluded")
            pair = load_segmentation_sample(sample, SegmentationTaxonomy(), allow_synthetic=True)
            pair_checks.append({"sample_id": sample.sample_id, "origin": sample.origin, "shape": list(pair.mask.shape),
                                "mapped_class_ids": np.unique(pair.mask).tolist(), "geometry_valid": True})
        fabrics.append({"fabric_id": fabric_id, "row_count": len(rows), "origin_counts": dict(Counter(row["origin"] for row in rows)),
                        "masked_origin_counts": dict(Counter(row["origin"] for row in rows if row["mask_path"])),
                        "split_counts": dict(Counter(row["split"] for row in rows)),
                        "legacy_defect_names": dict(Counter(row["defect_type"] for row in rows if row["defect_type"])),
                        "missing_capture_fields": [field for field in ("roll_id", "lot_id", "camera_id") if field not in rows[0]],
                        "binary_pair_spot_checks": pair_checks})
    return {"phase": 7, "scope": "Company runtime manifests; spot-check adapter_train masks only; no locked image evaluation",
            "fabrics": fabrics, "multiclass_semantic_mapping_inferred": False, "trained_model": False,
            "production_training_readiness": "UNVERIFIED_REAL_PIXEL_LABELS_AND_CAPTURE_GROUP_METADATA",
            "note": "All masked Company rows are declared synthetic. Folder lot_001 is not verified acquisition provenance. Existing historic locked entries remain untouched, not fresh holdout. Binary generic defect mapping is explicit, not eight-class ground truth."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--demo", action="store_true")
    choice.add_argument("--audit-existing", action="store_true")
    choice.add_argument("--schema", action="store_true")
    choice.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/segmentation_pipeline_example.json")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicit validation-only opt-in, not real training approval")
    args = parser.parse_args()
    if args.demo:
        result = demo()
    elif args.audit_existing:
        result = audit_existing()
    elif args.schema:
        result = SegmentationManifest.model_json_schema()
    else:
        manifest = SegmentationManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        config = SegmentationPipelineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if manifest.taxonomy != config.taxonomy:
            raise ValueError("Manifest and pipeline taxonomy must match")
        split = grouped_segmentation_split(manifest, config.split)
        checked = 0
        for sample in manifest.samples:
            if split.assignments[sample.sample_id] in {"train", "validation", "test"}:
                load_segmentation_sample(sample, manifest.taxonomy, base_dir=args.manifest.resolve().parent,
                                         allow_synthetic=args.allow_synthetic)
                checked += 1
        result = {"phase": 7, "status": "VALIDATED_RESEARCH_DATA_ONLY", "dataset_version": manifest.dataset_version,
                  "split": asdict(split), "checked_samples": checked, "trained_model": False,
                  "writes_performed": False, "production_accuracy_evaluated": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
