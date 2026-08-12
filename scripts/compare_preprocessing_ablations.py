"""Create ablation configs and aggregate their evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.preprocessing.config import ABLATION_VARIANTS, PreprocessConfig, config_for_ablation


FIELDS = (
    "variant",
    "preprocessing_version",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "normal_mean_score",
    "defective_mean_score",
    "processing_time_ms",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/compare preprocessing ablation variants.")
    parser.add_argument("--reports", type=Path, default=Path("storage/evaluation_results/ablations"))
    parser.add_argument("--output", type=Path, default=Path("storage/evaluation_results/ablations/comparison.csv"))
    parser.add_argument("--configs-output", type=Path, default=Path("config/ablations"))
    parser.add_argument("--base-config", type=Path)
    args = parser.parse_args()
    base = PreprocessConfig.load(args.base_config) if args.base_config else PreprocessConfig()
    args.configs_output.mkdir(parents=True, exist_ok=True)
    for variant in ABLATION_VARIANTS:
        config = config_for_ablation(variant, base)
        (args.configs_output / f"{variant}.json").write_text(
            json.dumps(config.to_dict(), indent=2),
            encoding="utf-8",
        )
    rows = []
    for path in sorted(args.reports.glob("P*/**/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        best = report.get("best_threshold") or {}
        images = report.get("images") or []
        normal = [item["mean_anomaly_score"] for item in images if not item["expected_defective"]]
        defective = [item["mean_anomaly_score"] for item in images if item["expected_defective"]]
        rows.append(
            {
                "variant": next((part for part in path.parts if part.upper() in ABLATION_VARIANTS), path.parent.name).upper(),
                "preprocessing_version": report.get("preprocessing_version", "v1"),
                **{key: best.get(key, 0) for key in ("accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn")},
                "normal_mean_score": statistics.fmean(normal) if normal else 0.0,
                "defective_mean_score": statistics.fmean(defective) if defective else 0.0,
                "processing_time_ms": report.get("processing_time_ms", 0.0),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"configs": str(args.configs_output), "comparison": str(args.output), "report_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
