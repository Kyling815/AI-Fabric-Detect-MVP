from pathlib import Path
from queue import Empty

from backend.app.core.line_scan import LineScanConfig


def scan_config(**updates):
    root = Path(__file__).resolve().parents[1]
    fields = LineScanConfig.model_validate_json((root / "config/line_scan_simulation_example.json").read_text(encoding="utf-8")).model_dump()
    fields.update(updates)
    return LineScanConfig.model_validate(fields)


def run_to_end(runtime, *, eager=False):
    results = []
    for _ in range(runtime.config.total_lines * 5 + 30):
        status = runtime.step()
        if eager or status.last_event == "backpressure" or status.state != "running":
            while True:
                try:
                    results.append(runtime.take())
                except Empty:
                    break
        if runtime.snapshot().state in {"stopped", "failed"}:
            return results
    raise AssertionError("Runtime failed to terminate within deterministic step bound")
