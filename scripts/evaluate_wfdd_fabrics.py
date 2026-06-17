import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS
from scripts.evaluate_pipeline import (
    choose_best_threshold,
    compute_threshold_metrics,
    parse_thresholds,
    score_image,
)
from backend.app.ai.anomaly_detector import FaissAnomalyDetector
from backend.app.ai.feature_extractor import CNNFeatureExtractor


def evaluate_wfdd_fabrics(
    wfdd_root: Path,
    fabrics: list[str] | None,
    thresholds: list[float],
    max_reference_images: int | None,
    max_images_per_class: int | None,
    min_area_ratio: float,
    min_suspicious_patches: int,
    min_cluster_patches: int,
) -> dict:
    settings = get_settings()
    selected_fabrics = fabrics or [
        item.name for item in sorted(wfdd_root.iterdir(), key=lambda path: path.name.lower()) if item.is_dir()
    ]
    reports = []

    for fabric_type in selected_fabrics:
        fabric_dir = wfdd_root / fabric_type
        train_good = fabric_dir / "train" / "good"
        if not train_good.exists():
            reports.append({"fabric_type": fabric_type, "status": "missing_train_good"})
            continue

        reference_images = _list_images(train_good, max_reference_images)
        reference_version = _reference_version(settings.model_name, fabric_type)
        builder = ReferenceLibraryBuilder(
            indexes_dir=resolve_project_path(settings.indexes_dir),
            metadata_dir=resolve_project_path(settings.metadata_dir),
        )
        artifact = builder.build(
            ReferenceBuildConfig(
                dataset_name="wfdd",
                fabric_type=fabric_type,
                normal_path=train_good,
                reference_version=reference_version,
                model_name=settings.model_name,
                device=settings.device,
                use_pretrained_weights=settings.use_pretrained_weights,
                image_max_size=settings.image_max_size,
                model_input_size=settings.model_input_size,
                patch_size=settings.patch_size,
                patch_stride=settings.patch_stride,
                batch_size=settings.batch_size,
                max_images=max_reference_images,
                projection_head_path=resolve_project_path(settings.projection_head_path)
                if settings.use_projection_head and settings.projection_head_path
                else None,
            ),
            image_paths=reference_images,
        )

        eval_images = collect_wfdd_test_images(fabric_dir, max_images_per_class)
        detector = FaissAnomalyDetector(metadata_path=artifact.metadata_path, threshold=0.0)
        extractor = CNNFeatureExtractor(
            model_name=settings.model_name,
            device=settings.device,
            use_pretrained_weights=settings.use_pretrained_weights,
            model_input_size=settings.model_input_size,
        )
        image_scores = [
            score_image(path, expected_defective, extractor, detector)
            for path, expected_defective in eval_images
        ]
        metrics = [
            compute_threshold_metrics(
                image_scores,
                threshold=threshold,
                min_area_ratio=min_area_ratio,
                min_suspicious_patches=min_suspicious_patches,
                min_cluster_patches=min_cluster_patches,
            )
            for threshold in thresholds
        ]
        best = choose_best_threshold(metrics) if metrics else None
        reports.append(
            {
                "fabric_type": fabric_type,
                "status": "evaluated",
                "reference_version": artifact.reference_version,
                "reference_image_count": artifact.processed_image_count,
                "reference_patch_count": artifact.patch_count,
                "evaluation_image_count": len(image_scores),
                "best_threshold": asdict(best) if best else None,
                "thresholds": [asdict(item) for item in metrics],
            }
        )

    best_thresholds = {
        report["fabric_type"]: report["best_threshold"]["threshold"]
        for report in reports
        if report.get("status") == "evaluated" and report.get("best_threshold")
    }
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_name": "wfdd",
        "wfdd_root": str(wfdd_root),
        "model_name": settings.model_name,
        "use_pretrained_weights": settings.use_pretrained_weights,
        "min_area_ratio": min_area_ratio,
        "min_suspicious_patches": min_suspicious_patches,
        "min_cluster_patches": min_cluster_patches,
        "best_thresholds_by_fabric": best_thresholds,
        "fabrics": reports,
    }


def collect_wfdd_test_images(fabric_dir: Path, max_images_per_class: int | None) -> list[tuple[Path, bool]]:
    test_dir = fabric_dir / "test"
    if not test_dir.exists():
        return []

    images: list[tuple[Path, bool]] = []
    for class_dir in sorted((item for item in test_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        expected_defective = class_dir.name.lower() != "good"
        class_images = _list_images(class_dir, max_images_per_class)
        images.extend((path, expected_defective) for path in class_images)
    return images


def _list_images(root: Path, limit: int | None) -> list[Path]:
    images = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
    return images[:limit] if limit is not None else images


def _reference_version(model_name: str, fabric_type: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    raw = f"{model_name}_wfdd_{fabric_type}_{timestamp}".lower()
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build and evaluate per-fabric WFDD reference libraries.")
    parser.add_argument("--wfdd-root", default=settings.wfdd_dataset_dir)
    parser.add_argument("--fabric", action="append", default=None, help="Fabric folder to evaluate. Repeat for multiple.")
    parser.add_argument("--max-reference-images", type=int, default=25)
    parser.add_argument("--max-images-per-class", type=int, default=10)
    parser.add_argument("--thresholds", default="0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.50,0.75")
    parser.add_argument("--min-area-ratio", type=float, default=settings.good_area_threshold)
    parser.add_argument("--min-suspicious-patches", type=int, default=1)
    parser.add_argument("--min-cluster-patches", type=int, default=settings.min_defect_cluster_patches)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = evaluate_wfdd_fabrics(
        wfdd_root=resolve_project_path(args.wfdd_root),
        fabrics=args.fabric,
        thresholds=parse_thresholds(args.thresholds),
        max_reference_images=args.max_reference_images,
        max_images_per_class=args.max_images_per_class,
        min_area_ratio=args.min_area_ratio,
        min_suspicious_patches=args.min_suspicious_patches,
        min_cluster_patches=args.min_cluster_patches,
    )
    output_path = Path(args.output) if args.output else resolve_project_path(settings.metadata_dir) / "wfdd_fabric_thresholds_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "best_thresholds_by_fabric": report["best_thresholds_by_fabric"],
                "fabrics": [
                    {
                        "fabric_type": item["fabric_type"],
                        "status": item["status"],
                        "best_threshold": item.get("best_threshold"),
                    }
                    for item in report["fabrics"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
