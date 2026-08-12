import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.anomaly_detector import FaissAnomalyDetector
from backend.app.ai.anomaly_detector import PatchAnomaly
from backend.app.ai.bbox import filter_boxes_by_cluster_size, merge_suspicious_patches
from backend.app.ai.feature_extractor import CNNFeatureExtractor
from backend.app.ai.patching import PatchBox
from backend.app.ai.preprocess import ImageReadError
from backend.app.ai.preprocessing.adapter import configuration_from_reference_metadata, prepare_image
from backend.app.ai.projection_head import project_embeddings
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageScore:
    path: str
    class_name: str
    expected_defective: bool
    patch_count: int
    max_anomaly_score: float
    mean_anomaly_score: float
    anomaly_scores: list[float]
    boxes: list[dict]


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    total: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    average_defect_area_ratio: float
    average_box_count: float


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def compute_threshold_metrics(
    scores: list[ImageScore],
    threshold: float,
    min_area_ratio: float,
    min_suspicious_patches: int,
    min_cluster_patches: int,
) -> ThresholdMetrics:
    tp = fp = tn = fn = 0
    area_ratios: list[float] = []
    box_counts: list[int] = []

    for image_score in scores:
        merged_boxes = _merged_boxes_for_threshold(
            image_score=image_score,
            threshold=threshold,
            min_cluster_patches=min_cluster_patches,
        )
        total_area = _image_area_from_boxes(image_score.boxes)
        suspicious_area = sum(box.area for box in merged_boxes)
        area_ratio = min(1.0, suspicious_area / total_area) if total_area > 0 else 0.0
        area_ratios.append(area_ratio)
        box_counts.append(len(merged_boxes))
        clustered_patch_count = sum(box.patch_count for box in merged_boxes)
        predicted_defective = clustered_patch_count >= min_suspicious_patches and area_ratio >= min_area_ratio

        if predicted_defective and image_score.expected_defective:
            tp += 1
        elif predicted_defective and not image_score.expected_defective:
            fp += 1
        elif not predicted_defective and image_score.expected_defective:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return ThresholdMetrics(
        threshold=threshold,
        total=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=_safe_div(tp + tn, total),
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=_safe_div(fp, fp + tn),
        false_negative_rate=_safe_div(fn, fn + tp),
        average_defect_area_ratio=sum(area_ratios) / len(area_ratios) if area_ratios else 0.0,
        average_box_count=sum(box_counts) / len(box_counts) if box_counts else 0.0,
    )


def choose_best_threshold(metrics: list[ThresholdMetrics]) -> ThresholdMetrics:
    if not metrics:
        raise ValueError("No metrics available.")
    return sorted(metrics, key=lambda item: (item.f1, item.recall, -item.false_positive_rate), reverse=True)[0]


def evaluate_dme(
    dataset_root: Path,
    metadata_path: Path,
    thresholds: list[float],
    max_images_per_class: int | None,
    min_area_ratio: float,
    min_suspicious_patches: int,
    min_cluster_patches: int,
    split_manifest: Path | None = None,
) -> dict:
    settings = get_settings()
    image_paths = (
        collect_manifest_evaluation_images(split_manifest, max_images_per_class=max_images_per_class)
        if split_manifest
        else collect_dme_images(dataset_root, max_images_per_class=max_images_per_class)
    )
    detector = FaissAnomalyDetector(metadata_path=metadata_path, threshold=0.0)
    extractor = CNNFeatureExtractor(
        model_name=settings.model_name,
        device=settings.device,
        use_pretrained_weights=settings.use_pretrained_weights,
        model_input_size=settings.model_input_size,
    )

    image_scores = [
        score_image(path, expected_defective, extractor, detector)
        for path, expected_defective in image_paths
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
    best = choose_best_threshold(metrics)

    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_root": str(dataset_root),
        "metadata_path": str(metadata_path),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "reference_version": detector.reference_version,
        "preprocessing_version": configuration_from_reference_metadata(detector.metadata)[0],
        "configuration_hash": detector.metadata.get("configuration_hash"),
        "fabric_type": detector.metadata.get("fabric_type"),
        "model_name": settings.model_name,
        "use_pretrained_weights": settings.use_pretrained_weights,
        "image_count": len(image_scores),
        "max_images_per_class": max_images_per_class,
        "min_area_ratio": min_area_ratio,
        "min_suspicious_patches": min_suspicious_patches,
        "min_cluster_patches": min_cluster_patches,
        "best_threshold": asdict(best),
        "thresholds": [asdict(item) for item in metrics],
        "images": [asdict(item) for item in image_scores],
    }


def collect_dme_images(dataset_root: Path, max_images_per_class: int | None) -> list[tuple[Path, bool]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"DME dataset root not found: {dataset_root}")

    image_paths: list[tuple[Path, bool]] = []
    for class_dir in sorted((item for item in dataset_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        expected_defective = class_dir.name.lower() not in {"non defective", "non_defective", "good", "normal"}
        class_images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda item: str(item).lower(),
        )
        if max_images_per_class is not None:
            class_images = class_images[:max_images_per_class]
        image_paths.extend((path, expected_defective) for path in class_images)
    return image_paths


def collect_manifest_evaluation_images(manifest_path: Path, max_images_per_class: int | None) -> list[tuple[Path, bool]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for item in manifest["evaluation"]:
        grouped.setdefault(item["class_name"], []).append(item)

    image_paths: list[tuple[Path, bool]] = []
    for class_name in sorted(grouped):
        records = sorted(grouped[class_name], key=lambda item: item["path"].lower())
        if max_images_per_class is not None:
            records = records[:max_images_per_class]
        image_paths.extend((Path(item["path"]), bool(item["expected_defective"])) for item in records)
    return image_paths


def score_image(
    path: Path,
    expected_defective: bool,
    extractor: CNNFeatureExtractor,
    detector: FaissAnomalyDetector,
) -> ImageScore:
    settings = get_settings()
    version, preprocessing_config = configuration_from_reference_metadata(detector.metadata)
    try:
        prepared = prepare_image(
            path,
            preprocessing_version=version,
            image_max_size=int(detector.metadata.get("preprocess", {}).get("image_max_size", settings.image_max_size)),
            patch_size=int(detector.metadata.get("preprocess", {}).get("patch_size", settings.patch_size)),
            patch_stride=int(detector.metadata.get("preprocess", {}).get("patch_stride", settings.patch_stride)),
            preprocessing_config=preprocessing_config,
            fabric_type=detector.metadata.get("fabric_type"),
            split="validation",
        )
    except (ImageReadError, OSError, ValueError) as exc:
        raise RuntimeError(f"Image failed {version} preprocessing during evaluation: {path}") from exc
    patches = prepared.patches
    embeddings = extractor.extract((patch.image for patch in patches), batch_size=settings.batch_size)
    projection_metadata = detector.metadata.get("projection_head")
    if projection_metadata:
        embeddings, _ = project_embeddings(
            embeddings,
            checkpoint_path=Path(projection_metadata["checkpoint_path"]),
            device=extractor.info.device,
            batch_size=max(settings.batch_size, 1),
        )
    elif settings.use_projection_head and settings.projection_head_path:
        embeddings, _ = project_embeddings(
            embeddings,
            checkpoint_path=resolve_project_path(settings.projection_head_path),
            device=extractor.info.device,
            batch_size=max(settings.batch_size, 1),
        )
    result = detector.score(embeddings, [patch.box for patch in patches])
    anomaly_scores = [patch.anomaly_score for patch in result.patch_scores]
    boxes = [
        {
            "patch_index": patch.patch_index,
            "x": patch.box.x,
            "y": patch.box.y,
            "w": patch.box.w,
            "h": patch.box.h,
        }
        for patch in result.patch_scores
    ]

    return ImageScore(
        path=str(path),
        class_name=path.parent.name,
        expected_defective=expected_defective,
        patch_count=len(patches),
        max_anomaly_score=max(anomaly_scores) if anomaly_scores else 0.0,
        mean_anomaly_score=sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0,
        anomaly_scores=anomaly_scores,
        boxes=boxes,
    )


def _image_area_from_boxes(boxes: list[dict]) -> int:
    if not boxes:
        return 0
    width = max(box["x"] + box["w"] for box in boxes)
    height = max(box["y"] + box["h"] for box in boxes)
    return width * height


def _merged_boxes_for_threshold(image_score: ImageScore, threshold: float, min_cluster_patches: int):
    patches = [
        PatchAnomaly(
            patch_index=box["patch_index"],
            reference_offset=-1,
            score=1.0 - image_score.anomaly_scores[box["patch_index"]],
            anomaly_score=image_score.anomaly_scores[box["patch_index"]],
            suspicious=image_score.anomaly_scores[box["patch_index"]] >= threshold,
            box=PatchBox(x=box["x"], y=box["y"], w=box["w"], h=box["h"]),
        )
        for box in image_score.boxes
    ]
    merged = merge_suspicious_patches(patches)
    return filter_boxes_by_cluster_size(merged, min_cluster_patches)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate DME dataset anomaly thresholds.")
    parser.add_argument("--dataset-root", default=settings.dme_dataset_dir)
    parser.add_argument("--metadata-path", default=str(resolve_project_path(settings.metadata_dir) / "latest_reference.json"))
    parser.add_argument("--max-images-per-class", type=int, default=5)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--thresholds",
        default="0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.50,0.75",
    )
    parser.add_argument("--min-area-ratio", type=float, default=settings.good_area_threshold)
    parser.add_argument("--min-suspicious-patches", type=int, default=1)
    parser.add_argument("--min-cluster-patches", type=int, default=settings.min_defect_cluster_patches)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = evaluate_dme(
        dataset_root=resolve_project_path(args.dataset_root),
        metadata_path=Path(args.metadata_path),
        thresholds=parse_thresholds(args.thresholds),
        max_images_per_class=args.max_images_per_class,
        min_area_ratio=args.min_area_ratio,
        min_suspicious_patches=args.min_suspicious_patches,
        min_cluster_patches=args.min_cluster_patches,
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
    )

    version = report["preprocessing_version"]
    output_path = (
        Path(args.output)
        if args.output
        else resolve_project_path(f"storage/evaluation_results/{version}") / "dme_evaluation_latest.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    best = report["best_threshold"]
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "image_count": report["image_count"],
                "reference_version": report["reference_version"],
                "best_threshold": best,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
