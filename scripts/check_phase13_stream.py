"""Finite read-only four-lane demonstration with explicit bounded-queue backpressure."""

import argparse
import json
from pathlib import Path
from queue import Empty
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.line_scan import LineScanConfig
from backend.app.services.line_scan_runtime import SimulatedStreamRuntime


def demo():
    config = LineScanConfig.model_validate_json((PROJECT_ROOT / "config/line_scan_simulation_example.json").read_text(encoding="utf-8"))
    runtime, bundles = SimulatedStreamRuntime(config), []
    # Deliberately delay consumption until bounded queue pressure or source completion.
    for _ in range(config.total_lines * 4 + 20):
        status = runtime.step()
        if status.last_event == "backpressure" or status.state != "running":
            while True:
                try:
                    bundles.append(runtime.take())
                except Empty:
                    break
        if runtime.snapshot().state in {"stopped", "failed"}:
            break
    else:
        raise AssertionError("Finite stream demo did not terminate")
    status = runtime.snapshot()
    if (status.state != "stopped" or status.delivered_line_ticks != config.total_lines or status.backpressure_events == 0
            or status.created_bundles != 3 or len(bundles) != 3):
        raise AssertionError("Stream demo delivery/accounting mismatch")
    return {"phase": 13, "status": "HYPOTHETICAL_SOFTWARE_CHECK_ONLY", "camera_count": 4,
        "config": config.model_dump(mode="json"), "bundle_count": len(bundles), "chunk_count": len(bundles) * 4,
        "chunks": [chunk.metadata() for bundle in bundles for chunk in bundle.chunks], "runtime": status.model_dump(mode="json"),
        "queue_policy": "Bounded atomic bundles + one retained pending bundle; cooperative producer pauses until delivery can advance.",
        "production_api_integrated": False, "models_trained_or_loaded": False, "vendor_sdk_integrated": False,
        "physical_calibration_inferred": False, "writes_performed": False, "seam_merging_implemented": False,
        "note": "Synthetic RGB/virtual clock, not industrial throughput or sensor synchronization proof. take() transfers ownership to caller, not inference completion or durable acknowledgement. Phase 14/15 not started."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    if parser.parse_args().demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
