"""Read-only period diagnostic. Does not run or configure CNN/FAISS inspection."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.pattern_period import PatternPeriodEstimator
from backend.app.core.pattern import PatternEstimatorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = PatternEstimatorConfig()
    if args.config:
        config = PatternEstimatorConfig.model_validate(json.loads(args.config.read_text(encoding="utf-8")))
    result = PatternPeriodEstimator().estimate_path(args.image, config)
    print(json.dumps({"mode": config.mode, **asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
