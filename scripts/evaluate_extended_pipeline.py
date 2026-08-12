import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.anomaly_detector import FaissAnomalyDetector
from backend.app.ai.feature_extractor import CNNFeatureExtractor
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.reference_registry import ReferenceRegistry, ReferenceRegistryEntry
from scripts.evaluate_pipeline import compute_threshold_metrics, parse_thresholds, score_image


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate registered references against extended dataset manifests.")
    parser.add_argument("--manifest-csv", default="storage/metadata/extended_dataset_manifests/extended_all.csv")
    parser.add_argument("--reference-id", action="append", default=None)
    parser.add_argument("--thresholds", default="0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.10,0.12,0.15")
    parser.add_argument("--max-images-per-group", type=int, default=50)
    parser.add_argument("--cross-max-images-per-group", type=int, default=10)
    parser.add_argument("--min-area-ratio", type=float, default=settings.good_area_threshold)
    parser.add_argument("--min-suspicious-patches", type=int, default=1)
    parser.add_argument("--min-cluster-patches", type=int, default=settings.min_defect_cluster_patches)
    parser.add_argument("--include-cross-reference", action="store_true")
    parser.add_argument("--update-reference-thresholds", action="store_true")
    parser.add_argument("--output-dir", default="storage/metadata/evaluation_results")
    args = parser.parse_args()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifest(resolve_project_path(args.manifest_csv))
    registry = ReferenceRegistry()
    entries = registry.list_latest_entries()
    if args.reference_id:
        entries = []
        for requested in dict.fromkeys(args.reference_id):
            try:
                entries.append(registry.get(requested))
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc
    thresholds = parse_thresholds(args.thresholds)

    reports = []
    print(
        f"matched_evaluation references={len(entries)} max_images_per_group={args.max_images_per_group}",
        flush=True,
    )
    for entry in entries:
        rows = matching_rows(entry, manifest_rows, max_images=args.max_images_per_group)
        if not rows:
            reports.append(empty_report(entry, "no_matching_manifest_rows"))
            continue
        print(f"evaluating reference={entry.reference_id} scenario=matched_reference images={len(rows)}", flush=True)
        reports.append(
            evaluate_reference(
                entry=entry,
                rows=rows,
                thresholds=thresholds,
                min_area_ratio=args.min_area_ratio,
                min_suspicious_patches=args.min_suspicious_patches,
                min_cluster_patches=args.min_cluster_patches,
                scenario="matched_reference",
            )
        )

    cross_reports = []
    if args.include_cross_reference:
        groups = group_eval_rows(manifest_rows, max_images=args.cross_max_images_per_group)
        scenario_count = sum(1 for entry in entries for group_key in groups if group_key != expected_group_key(entry))
        print(
            f"cross_reference_evaluation scenarios={scenario_count} cross_max_images_per_group={args.cross_max_images_per_group}",
            flush=True,
        )
        for entry in entries:
            for group_key, rows in groups.items():
                if group_key == expected_group_key(entry):
                    continue
                print(
                    f"evaluating reference={entry.reference_id} scenario=cross_reference:{group_key} images={len(rows)}",
                    flush=True,
                )
                cross_reports.append(
                    evaluate_reference(
                        entry=entry,
                        rows=rows,
                        thresholds=thresholds,
                        min_area_ratio=args.min_area_ratio,
                        min_suspicious_patches=args.min_suspicious_patches,
                        min_cluster_patches=args.min_cluster_patches,
                        scenario=f"cross_reference:{group_key}",
                    )
                )

    summary = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "manifest_csv": str(resolve_project_path(args.manifest_csv)),
        "reference_count": len(entries),
        "matched_references": reports,
        "cross_reference": cross_reports,
    }
    (output_dir / "extended_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "extended_evaluation_summary.csv", reports + cross_reports)
    write_markdown_report(output_dir / "extended_evaluation_report.md", summary)
    for report in reports:
        if report.get("status") == "evaluated":
            path = output_dir / f"{report['reference_id']}_threshold_tuning.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.update_reference_thresholds:
                update_reference_thresholds(report)

    print(
        json.dumps(
            {
                "summary_json": str(output_dir / "extended_evaluation_summary.json"),
                "summary_csv": str(output_dir / "extended_evaluation_summary.csv"),
                "report_md": str(output_dir / "extended_evaluation_report.md"),
                "references": [
                    {
                        "reference_id": item.get("reference_id"),
                        "status": item.get("status"),
                        "best_threshold": item.get("best_threshold"),
                    }
                    for item in reports
                ],
            },
            indent=2,
        )
    )


def evaluate_reference(
    entry: ReferenceRegistryEntry,
    rows: list[dict],
    thresholds: list[float],
    min_area_ratio: float,
    min_suspicious_patches: int,
    min_cluster_patches: int,
    scenario: str,
) -> dict:
    settings = get_settings()
    detector = FaissAnomalyDetector(metadata_path=entry.metadata_path, threshold=0.0)
    extractor = CNNFeatureExtractor(
        model_name=settings.model_name,
        device=settings.device,
        use_pretrained_weights=settings.use_pretrained_weights,
        model_input_size=settings.model_input_size,
    )

    image_scores = []
    total_seconds = 0.0
    skipped_images = []
    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(
                f"  scoring {scenario} {entry.reference_id}: {index}/{len(rows)}",
                flush=True,
            )
        start = time.perf_counter()
        try:
            image_scores.append(score_image(Path(row["image_path"]), row["label"] == "defective", extractor, detector))
            total_seconds += time.perf_counter() - start
        except RuntimeError as exc:
            skipped_images.append({"image_path": row["image_path"], "reason": str(exc)})
            continue

    if not image_scores:
        return {
            "status": "no_readable_images",
            "scenario": scenario,
            "reference_id": entry.reference_id,
            "reference_version": entry.reference_version,
            "dataset_name": entry.dataset_name,
            "fabric_group": expected_group_key(entry),
            "metadata_path": str(entry.metadata_path),
            "evaluated_image_count": 0,
            "skipped_image_count": len(skipped_images),
            "skipped_images": skipped_images[:50],
        }

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
    best = choose_recall_first(metrics)
    return {
        "status": "evaluated",
        "scenario": scenario,
        "reference_id": entry.reference_id,
        "reference_version": entry.reference_version,
        "dataset_name": entry.dataset_name,
        "fabric_group": expected_group_key(entry),
        "metadata_path": str(entry.metadata_path),
        "threshold_used": best.threshold,
        "average_inference_seconds": total_seconds / len(image_scores) if image_scores else 0.0,
        "evaluated_image_count": len(rows),
        "scored_image_count": len(image_scores),
        "skipped_image_count": len(skipped_images),
        "skipped_images": skipped_images[:50],
        "best_threshold": asdict(best),
        "thresholds": [asdict(item) for item in metrics],
    }


def choose_recall_first(metrics):
    return sorted(
        metrics,
        key=lambda item: (
            item.recall,
            -item.fn,
            item.f1,
            item.precision,
            -item.false_positive_rate,
            item.threshold,
        ),
        reverse=True,
    )[0]


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matching_rows(entry: ReferenceRegistryEntry, rows: list[dict], max_images: int) -> list[dict]:
    group_key = expected_group_key(entry)
    candidates = [
        row
        for row in rows
        if row["label"] in {"normal", "defective"}
        and row["split_source"] != "wfdd_train_good"
        and group_match(group_key, row)
    ]
    return balanced_rows(candidates, max_images)


def group_match(group_key: str, row: dict) -> bool:
    if group_key == "dme_plain":
        return row["dataset_name"] == "dme_20250129"
    if group_key.startswith("wfdd_"):
        return row["fabric_group"] == group_key
    if group_key.startswith("lusitano"):
        return row["dataset_name"] == "lusitano"
    return row["fabric_group"] == group_key


def expected_group_key(entry: ReferenceRegistryEntry) -> str:
    if entry.reference_id.startswith("dme_plain"):
        return "dme_plain"
    if entry.reference_id.startswith("wfdd_"):
        parts = entry.reference_id.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else entry.reference_id
    if entry.reference_id.startswith("lusitano"):
        return "lusitano_group_auto"
    if entry.fabric_type:
        return f"{entry.dataset_name}_{entry.fabric_type}"
    return entry.dataset_name


def group_eval_rows(rows: list[dict], max_images: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["label"] not in {"normal", "defective"} or row["split_source"] == "wfdd_train_good":
            continue
        key = row["fabric_group"]
        grouped.setdefault(key, []).append(row)
    return {key: balanced_rows(value, max_images) for key, value in grouped.items()}


def balanced_rows(rows: list[dict], max_images: int) -> list[dict]:
    if max_images <= 0:
        return []
    normal_rows = [row for row in rows if row["label"] == "normal"]
    defective_rows = [row for row in rows if row["label"] == "defective"]
    if not normal_rows or not defective_rows:
        return rows[:max_images]

    first_class_limit = max(1, max_images // 2)
    second_class_limit = max_images - first_class_limit
    selected = normal_rows[:first_class_limit] + defective_rows[:second_class_limit]
    if len(selected) < max_images:
        selected_paths = {row["image_path"] for row in selected}
        remainder = [row for row in rows if row["image_path"] not in selected_paths]
        selected.extend(remainder[: max_images - len(selected)])
    return selected[:max_images]


def empty_report(entry: ReferenceRegistryEntry, status: str) -> dict:
    return {
        "status": status,
        "reference_id": entry.reference_id,
        "reference_version": entry.reference_version,
        "dataset_name": entry.dataset_name,
        "fabric_group": expected_group_key(entry),
    }


def write_summary_csv(path: Path, reports: list[dict]) -> None:
    fields = [
        "scenario",
        "reference_id",
        "reference_version",
        "dataset_name",
        "fabric_group",
        "status",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "average_inference_seconds",
        "evaluated_image_count",
        "scored_image_count",
        "skipped_image_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            best = report.get("best_threshold") or {}
            writer.writerow(
                {
                    "scenario": report.get("scenario", ""),
                    "reference_id": report.get("reference_id", ""),
                    "reference_version": report.get("reference_version", ""),
                    "dataset_name": report.get("dataset_name", ""),
                    "fabric_group": report.get("fabric_group", ""),
                    "status": report.get("status", ""),
                    "threshold": best.get("threshold", ""),
                    "tp": best.get("tp", ""),
                    "fp": best.get("fp", ""),
                    "tn": best.get("tn", ""),
                    "fn": best.get("fn", ""),
                    "accuracy": best.get("accuracy", ""),
                    "precision": best.get("precision", ""),
                    "recall": best.get("recall", ""),
                    "f1": best.get("f1", ""),
                    "false_positive_rate": best.get("false_positive_rate", ""),
                    "false_negative_rate": best.get("false_negative_rate", ""),
                    "average_inference_seconds": report.get("average_inference_seconds", ""),
                    "evaluated_image_count": report.get("evaluated_image_count", ""),
                    "scored_image_count": report.get("scored_image_count", ""),
                    "skipped_image_count": report.get("skipped_image_count", ""),
                }
            )


def write_markdown_report(path: Path, summary: dict) -> None:
    lines = [
        "# Extended Evaluation Report",
        "",
        f"Created at: {summary['created_at']}",
        "",
        "## Reference Metrics",
        "",
        "| Reference | Scenario | Images | Threshold | TP | FP | TN | FN | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in summary["matched_references"]:
        best = report.get("best_threshold") or {}
        lines.append(
            "| {reference} | {scenario} | {images} | {threshold} | {tp} | {fp} | {tn} | {fn} | {precision} | {recall} | {f1} |".format(
                reference=report.get("reference_id", ""),
                scenario=report.get("scenario", ""),
                images=report.get("evaluated_image_count", ""),
                threshold=best.get("threshold", ""),
                tp=best.get("tp", ""),
                fp=best.get("fp", ""),
                tn=best.get("tn", ""),
                fn=best.get("fn", ""),
                precision=round(best.get("precision", 0), 4) if best else "",
                recall=round(best.get("recall", 0), 4) if best else "",
                f1=round(best.get("f1", 0), 4) if best else "",
            )
        )
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- Results depend on the selected reference library matching the inspected fabric style.",
            "- Lusitano evaluation may be normal-only if no defect folders are available.",
            "- Cross-reference results are intended to confirm style mismatch behavior, not final production thresholds.",
            "",
            "## Next Recommendations",
            "",
            "- Prefer per-fabric thresholds over a single global threshold.",
            "- Keep the old DME baseline active until extended evaluation shows equal or better DME recall.",
            "- Register new references as optional before switching the default reference id.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_reference_thresholds(report: dict) -> None:
    best = report.get("best_threshold") or {}
    thresholds = {
        "anomaly_threshold": best.get("threshold"),
        "good_area_threshold": get_settings().good_area_threshold,
        "bad_area_threshold": get_settings().bad_area_threshold,
        "min_defect_cluster_patches": get_settings().min_defect_cluster_patches,
    }
    metadata_path = Path(report["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["thresholds"] = thresholds
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_path = resolve_project_path(get_settings().reference_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for item in registry.get("references", []):
        if item.get("reference_id") == report["reference_id"]:
            item["thresholds"] = thresholds
    registry["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
