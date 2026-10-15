"""Run ONLY in .venv-tensorrt. Offline full research model; no activation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-local", action="store_true", required=True)
    parser.add_argument("--allow-untrained-probe", action="store_true", required=True)
    args = parser.parse_args()
    try:
        from backend.app.ai.tensorrt_benchmark import benchmark
        result = benchmark(args.candidate_dir, args.manifest_sha256, args.output_dir,
                           trusted_local=args.trusted_local, allow_untrained_probe=args.allow_untrained_probe)
    except Exception as exc:
        print(json.dumps({"phase": 17, "passed": False, "error_type": type(exc).__name__,
                          "message": str(exc), "status": "FAILED_STOP_NO_FALLBACK", "phase18_started": False}),
              file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "report": str(args.output_dir / "benchmark_report.json"),
                      "selected_research_configuration": result["selected_research_configuration"],
                      "production_approved": False, "phase18_started": False}, indent=2))
    return 0 if result["at_least_one_engine_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
