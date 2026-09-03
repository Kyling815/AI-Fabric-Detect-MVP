import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_anomaly import DenseReferenceDetector, aggregate_image_score, localize_anomaly_map
from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.fabric_decision import decide_fabric_inspection
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.fabric_registry import FabricRegistry


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate isolated fabric references without threshold tuning.")
    parser.add_argument("--manifest-dir", default="storage/metadata/company_demo_manifests")
    parser.add_argument("--split", choices=("validation", "locked_test"), default="locked_test")
    parser.add_argument("--fabric-id", action="append", default=None)
    parser.add_argument("--reference-version", type=int, default=1)
    parser.add_argument("--output-dir", default="storage/metadata/fabric_demo_evaluation")
    parser.add_argument("--activate-if-accepted", action="store_true")
    args = parser.parse_args()

    registry = FabricRegistry()
    backbone = registry.shared_backbone
    extractor = get_shared_dense_extractor(
        str(backbone["backbone_version"]),
        settings.device,
        bool(backbone.get("use_pretrained_weights", True)),
        str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
    )
    fabrics = args.fabric_id or [item["fabric_id"] for item in registry.list_fabrics()]
    reports = []
    for fabric_id in fabrics:
        reference_id = f"{fabric_id}_ref_v{args.reference_version}"
        entry = registry.get_reference(fabric_id, reference_id)
        if not entry.thresholds:
            raise RuntimeError(f"Reference is not calibrated: {reference_id}")
        rows = read_rows(resolve_project_path(args.manifest_dir) / f"{fabric_id}_{args.split}.csv")
        reports.append(evaluate(fabric_id, entry, rows, extractor, int(backbone["image_max_size"]), args.split))

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "split": args.split,
        "reports": reports,
        "all_accepted": all(report["accepted"] for report in reports),
    }
    (output_dir / f"{args.split}_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_dir / f"{args.split}_images.csv", reports)
    if args.activate_if_accepted:
        if args.split != "locked_test":
            raise RuntimeError("Activation requires locked_test evaluation")
        if not payload["all_accepted"]:
            raise RuntimeError("References were not activated because acceptance criteria failed")
        activate(reports)
    print(json.dumps(payload, indent=2))


def evaluate(fabric_id, entry, rows, extractor, image_max_size: int, split: str) -> dict:
    detector = DenseReferenceDetector(entry.metadata_path)
    thresholds = entry.thresholds
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "good": 0, "needs_review": 0, "bad": 0}
    images = []
    localization_hits = 0
    mask_ious = []
    durations = []
    for index, row in enumerate(rows, start=1):
        print(f"[{fabric_id}] {split} {index}/{len(rows)} {Path(row['image_path']).name}", flush=True)
        started = time.perf_counter()
        dense = extractor.extract_path(Path(row["image_path"]), image_max_size=image_max_size)
        raw = detector.score(
            dense.embeddings,
            dense.boxes,
            grid_rows=dense.grid_rows,
            grid_cols=dense.grid_cols,
            image_width=dense.processed_width,
            image_height=dense.processed_height,
            thresholds=thresholds,
        )
        durations.append(time.perf_counter() - started)
        decision = decide_fabric_inspection(
            image_score=raw.image_score,
            boxes=raw.boxes,
            thresholds=thresholds,
            localized_score=raw.localized_score,
        )
        expected = row["label"] == "defective"
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
        if expected and raw.boxes:
            localization_hits += 1
        iou = mask_iou(row.get("mask_path"), raw.boxes, dense.processed_width, dense.processed_height)
        if iou is not None:
            mask_ious.append(iou)
        images.append(
            {
                "image_path": row["image_path"],
                "origin": row["origin"],
                "expected": row["label"],
                "defect_type": row["defect_type"],
                "result": decision.label,
                "reason": decision.reason,
                "anomaly_score": raw.image_score,
                "localized_score": raw.localized_score,
                "box_count": len(raw.boxes),
                "mask_iou": iou,
            }
        )
    defect_count = counts["tp"] + counts["fn"]
    good_count = counts["tn"] + counts["fp"]
    recall = counts["tp"] / max(1, defect_count)
    fpr = counts["fp"] / max(1, good_count)
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accepted = counts["fn"] == 0 and recall == 1.0 and fpr <= 0.10
    return {
        "fabric_id": fabric_id,
        "reference_id": entry.reference_id,
        "reference_version": entry.reference_version,
        "split": split,
        "good_count": good_count,
        "defect_count": defect_count,
        **counts,
        "defect_recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": counts["fn"] / max(1, defect_count),
        "precision": precision,
        "f1": f1,
        "bbox_recall": localization_hits / max(1, defect_count),
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else None,
        "average_inference_seconds": float(np.mean(durations)) if durations else 0.0,
        "accepted": accepted,
        "thresholds": thresholds,
        "images": images,
    }


def mask_iou(mask_path: str | None, boxes, width: int, height: int) -> float | None:
    if not mask_path:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    expected = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    predicted = np.zeros_like(expected, dtype=bool)
    for box in boxes:
        predicted[box.y : box.y + box.h, box.x : box.x + box.w] = True
    union = np.logical_or(expected, predicted).sum()
    return float(np.logical_and(expected, predicted).sum() / union) if union else 0.0


def activate(reports: list[dict]) -> None:
    path = resolve_project_path(get_settings().fabric_registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    for report in reports:
        fabric = registry["fabrics"][report["fabric_id"]]
        fabric["status"] = "ACTIVE"
        fabric["active_reference"] = report["reference_id"]
        for reference in fabric["references"]:
            if reference["reference_id"] == report["reference_id"]:
                reference["status"] = "ACTIVE"
    registry["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, reports: list[dict]) -> None:
    rows = [{"fabric_id": report["fabric_id"], **image} for report in reports for image in report["images"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["fabric_id"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
