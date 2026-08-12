"""Recalibrate and store v2 anomaly thresholds from held-out validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from scripts.evaluate_pipeline import evaluate_dme, parse_thresholds


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Recalibrate anomaly thresholds for a v2 reference.")
    parser.add_argument("--dataset-root", type=Path, default=resolve_project_path(settings.dme_dataset_dir))
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--fabric-type")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--thresholds", default="0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30")
    parser.add_argument("--max-images-per-class", type=int)
    parser.add_argument("--min-area-ratio", type=float, default=settings.good_area_threshold)
    parser.add_argument("--min-suspicious-patches", type=int, default=1)
    parser.add_argument("--min-cluster-patches", type=int, default=settings.min_defect_cluster_patches)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-reference-metadata", action="store_true")
    parser.add_argument("--registry", type=Path, default=resolve_project_path(settings.reference_registry_path))
    args = parser.parse_args()
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    if metadata.get("preprocessing_version", "v1") != "v2":
        raise SystemExit("Refusing to calibrate v2 thresholds against a non-v2 reference.")
    report = evaluate_dme(
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        thresholds=parse_thresholds(args.thresholds),
        max_images_per_class=args.max_images_per_class,
        min_area_ratio=args.min_area_ratio,
        min_suspicious_patches=args.min_suspicious_patches,
        min_cluster_patches=args.min_cluster_patches,
        split_manifest=args.split_manifest,
    )
    fabric = args.fabric_type or report.get("fabric_type") or "default"
    output = args.output or Path("storage/evaluation_results/v2") / fabric / "threshold_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.update_reference_metadata:
        _store_threshold(args.metadata_path, args.registry, report["best_threshold"]["threshold"])
    print(
        json.dumps(
            {
                "output": str(output),
                "fabric_type": fabric,
                "best_threshold": report["best_threshold"],
                "reference_metadata_updated": args.update_reference_metadata,
            },
            indent=2,
        )
    )
    return 0


def _store_threshold(metadata_path: Path, registry_path: Path, threshold: float) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    thresholds = dict(metadata.get("thresholds") or {})
    thresholds["anomaly_threshold"] = threshold
    metadata["thresholds"] = thresholds
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not registry_path.exists():
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    reference_version = metadata.get("reference_version")
    for record in registry.get("references", []):
        if record.get("reference_version") == reference_version:
            record["thresholds"] = thresholds
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
