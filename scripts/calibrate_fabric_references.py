import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_anomaly import (
    DenseReferenceDetector,
    CORE_LOGIC_VERSION,
    aggregate_image_score,
    localized_anomaly_score,
    localize_anomaly_map,
    resolve_pixel_threshold,
)
from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.fabric_decision import decide_fabric_inspection
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.fabric_registry import FabricRegistry


METHODS = [
    ("max", 1.0),
    ("top_k_mean", 0.001),
    ("top_k_mean", 0.0025),
    ("top_k_mean", 0.005),
    ("top_k_mean", 0.01),
    ("top_k_mean", 0.02),
    ("percentile_995", 0.005),
    ("max_median_contrast", 1.0),
    ("top_k_median_contrast", 0.001),
    ("top_k_median_contrast", 0.0025),
    ("top_k_median_contrast", 0.005),
    ("top_k_median_contrast", 0.01),
    ("top_k_median_contrast", 0.02),
    ("percentile_995_median_contrast", 0.005),
]

PIXEL_MAD_MULTIPLIERS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Calibrate per-fabric thresholds from validation only.")
    parser.add_argument("--manifest-dir", default="storage/metadata/company_demo_manifests")
    parser.add_argument("--fabric-id", action="append", default=None)
    parser.add_argument("--reference-version", type=int, default=1)
    args = parser.parse_args()

    registry = FabricRegistry()
    selected = args.fabric_id or [item["fabric_id"] for item in registry.list_fabrics()]
    backbone = registry.shared_backbone
    extractor = get_shared_dense_extractor(
        str(backbone["backbone_version"]),
        settings.device,
        bool(backbone.get("use_pretrained_weights", True)),
        str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
    )
    reports = []
    for fabric_id in selected:
        reference_id = f"{fabric_id}_ref_v{args.reference_version}"
        entry = registry.get_reference(fabric_id, reference_id)
        detector = DenseReferenceDetector(entry.metadata_path)
        rows = read_rows(resolve_project_path(args.manifest_dir) / f"{fabric_id}_validation.csv")
        scored = score_rows(rows, extractor, detector, int(backbone["image_max_size"]), fabric_id)
        thresholds, calibration = calibrate(scored)
        persist_calibration(entry.metadata_path, thresholds, calibration)
        reports.append({"fabric_id": fabric_id, "reference_id": reference_id, **calibration})
    update_registry(reports)
    print(json.dumps({"calibrated": reports}, indent=2))


def score_rows(rows, extractor, detector, image_max_size: int, fabric_id: str) -> list[dict]:
    scored = []
    for index, row in enumerate(rows, start=1):
        print(f"[{fabric_id}] validation {index}/{len(rows)} {Path(row['image_path']).name}", flush=True)
        dense = extractor.extract_path(Path(row["image_path"]), image_max_size=image_max_size)
        result = detector.score(
            dense.embeddings,
            dense.boxes,
            grid_rows=dense.grid_rows,
            grid_cols=dense.grid_cols,
            image_width=dense.processed_width,
            image_height=dense.processed_height,
            thresholds={"pixel_threshold": float("inf")},
        )
        scored.append(
            {
                "row": row,
                "patch_scores": result.patch_scores,
                "anomaly_map": result.anomaly_map,
                "processed_width": dense.processed_width,
                "processed_height": dense.processed_height,
            }
        )
    return scored


def calibrate(scored: list[dict]) -> tuple[dict, dict]:
    normal = [item for item in scored if item["row"]["label"] == "normal"]
    defective = [item for item in scored if item["row"]["label"] == "defective"]
    if not normal or not defective:
        raise RuntimeError("Calibration requires both normal and defective validation images")

    method_reports = []
    for method, fraction in METHODS:
        normal_scores = [aggregate_image_score(item["patch_scores"], method=method, top_fraction=fraction) for item in normal]
        defect_scores = [aggregate_image_score(item["patch_scores"], method=method, top_fraction=fraction) for item in defective]
        review = float(np.nextafter(min(defect_scores), -np.inf))
        fp = sum(score >= review for score in normal_scores)
        method_reports.append(
            {
                "method": method,
                "top_fraction": fraction,
                "review_threshold": review,
                "normal_scores": normal_scores,
                "defect_scores": defect_scores,
                "validation_fp": fp,
                "validation_fn": 0,
                "false_positive_rate": fp / len(normal_scores),
                "separation": min(defect_scores) - max(normal_scores),
            }
        )
    selected = sorted(
        method_reports,
        key=lambda item: (item["validation_fp"], -item["separation"], item["top_fraction"]),
    )[0]
    review = selected["review_threshold"]
    bad = choose_bad_threshold(selected["normal_scores"], selected["defect_scores"], review)

    pixel_candidates = [
        {"pixel_threshold_mode": "absolute", "pixel_threshold": value}
        for value in pixel_threshold_candidates(normal)
    ]
    pixel_candidates.extend(
        {
            "pixel_threshold_mode": "median_mad",
            "pixel_threshold": 0.0,
            "pixel_mad_multiplier": multiplier,
        }
        for multiplier in PIXEL_MAD_MULTIPLIERS
    )
    localization_reports = []
    for pixel_config in pixel_candidates:
        for minimum_area in (8, 16, 32, 64, 128):
            localization_thresholds = {**pixel_config, "minimum_defect_area": minimum_area}
            normal_boxes = [localize(item, localization_thresholds) for item in normal]
            defect_boxes = [localize(item, localization_thresholds) for item in defective]
            localization_reports.append(
                {
                    **localization_thresholds,
                    "defect_localization_recall": sum(bool(boxes) for boxes in defect_boxes) / len(defect_boxes),
                    "normal_localization_rate": sum(bool(boxes) for boxes in normal_boxes) / len(normal_boxes),
                    "mean_mask_iou": mean_mask_iou(defective, defect_boxes),
                }
            )
    localization = sorted(
        localization_reports,
        key=lambda item: (
            item["defect_localization_recall"],
            -item["normal_localization_rate"],
            item["mean_mask_iou"],
            item.get("pixel_mad_multiplier", 0.0),
            item["pixel_threshold"],
        ),
        reverse=True,
    )[0]
    normal_localized_scores = [localized_anomaly_score(item["patch_scores"]) for item in normal]
    defect_localized_scores = [localized_anomaly_score(item["patch_scores"]) for item in defective]
    localized_review = float(np.nextafter(max(normal_localized_scores), np.inf))
    defect_localized_boxes = [localize(item, localization) for item in defective]
    localization["localized_review_threshold"] = localized_review
    localization["localized_trigger_validation_recall"] = sum(
        bool(boxes) and score >= localized_review
        for score, boxes in zip(defect_localized_scores, defect_localized_boxes)
    ) / len(defective)
    thresholds = {
        "threshold_schema_version": 2,
        "core_logic_version": CORE_LOGIC_VERSION,
        "image_score_method": selected["method"],
        "image_score_top_fraction": selected["top_fraction"],
        "image_review_threshold": review,
        "image_bad_threshold": bad,
        "pixel_threshold_mode": localization["pixel_threshold_mode"],
        "pixel_threshold": localization["pixel_threshold"],
        "pixel_mad_multiplier": localization.get("pixel_mad_multiplier"),
        "minimum_defect_area": localization["minimum_defect_area"],
        "localized_review_threshold": localized_review,
    }
    validation_metrics = decision_metrics(scored, thresholds)
    calibration = {
        "status": "CALIBRATED",
        "calibrated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "selected_localization": localization,
        "image_score_candidates": method_reports,
        "localization_candidates": localization_reports,
    }
    return thresholds, calibration


def choose_bad_threshold(normal_scores: list[float], defect_scores: list[float], review: float) -> float:
    candidates = sorted(set([review, *normal_scores, *defect_scores]))
    reports = []
    for threshold in candidates:
        recall = sum(score >= threshold for score in defect_scores) / len(defect_scores)
        fpr = sum(score >= threshold for score in normal_scores) / len(normal_scores)
        if recall >= 0.8 and fpr <= 0.1:
            reports.append((recall, -fpr, threshold))
    if reports:
        return float(sorted(reports, reverse=True)[0][2])
    return float(max(review, np.median(defect_scores)))


def pixel_threshold_candidates(normal: list[dict]) -> list[float]:
    values = np.concatenate([item["patch_scores"] for item in normal])
    return sorted({float(np.quantile(values, q)) for q in (0.99, 0.995, 0.9975, 0.999, 0.9995)})


def localize(item: dict, thresholds: dict):
    pixel_threshold = resolve_pixel_threshold(item["anomaly_map"], thresholds)
    return localize_anomaly_map(
        item["anomaly_map"],
        image_width=item["processed_width"],
        image_height=item["processed_height"],
        pixel_threshold=pixel_threshold,
        minimum_defect_area=int(thresholds["minimum_defect_area"]),
    )


def decision_metrics(scored: list[dict], thresholds: dict) -> dict:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "good": 0, "needs_review": 0, "bad": 0}
    images = []
    for item in scored:
        image_score = aggregate_image_score(
            item["patch_scores"],
            method=thresholds["image_score_method"],
            top_fraction=thresholds["image_score_top_fraction"],
        )
        boxes = localize(item, thresholds)
        localized_score = localized_anomaly_score(item["patch_scores"])
        decision = decide_fabric_inspection(
            image_score=image_score,
            boxes=boxes,
            thresholds=thresholds,
            localized_score=localized_score,
        )
        expected = item["row"]["label"] == "defective"
        positive = decision.label != "GOOD"
        counts[decision.label.lower()] += 1
        if expected and positive:
            counts["tp"] += 1
        elif expected:
            counts["fn"] += 1
        elif positive:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
        images.append(
            {
                "image_path": item["row"]["image_path"],
                "label": item["row"]["label"],
                "result": decision.label,
                "reason": decision.reason,
                "image_score": image_score,
                "localized_score": localized_score,
                "box_count": len(boxes),
            }
        )
    counts["defect_recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    counts["false_positive_rate"] = counts["fp"] / max(1, counts["fp"] + counts["tn"])
    counts["false_negative_rate"] = counts["fn"] / max(1, counts["fn"] + counts["tp"])
    return {**counts, "images": images}


def mean_mask_iou(defective: list[dict], boxes_by_image: list[list]) -> float:
    values = []
    for item, boxes in zip(defective, boxes_by_image):
        mask_path = item["row"].get("mask_path")
        if not mask_path:
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (item["processed_width"], item["processed_height"]), interpolation=cv2.INTER_NEAREST) > 0
        predicted = np.zeros_like(mask, dtype=bool)
        for box in boxes:
            predicted[box.y : box.y + box.h, box.x : box.x + box.w] = True
        union = np.logical_or(mask, predicted).sum()
        values.append(float(np.logical_and(mask, predicted).sum() / union) if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def persist_calibration(metadata_path: Path, thresholds: dict, calibration: dict) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "CALIBRATED"
    metadata["thresholds"] = thresholds
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (metadata_path.parent / "thresholds.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def update_registry(reports: list[dict]) -> None:
    path = resolve_project_path(get_settings().fabric_registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["reference_id"]: item for item in reports}
    for fabric in registry["fabrics"].values():
        for reference in fabric.get("references", []):
            report = by_id.get(reference["reference_id"])
            if report:
                reference["status"] = "CALIBRATED"
                reference["thresholds"] = report["thresholds"]
    registry["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
