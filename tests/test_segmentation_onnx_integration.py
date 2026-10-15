import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_onnx_module_is_opt_in_and_http_remains_golden(phase1_runtime, case):
    from backend.app.ai import segmentation_onnx
    phase1_runtime.use_case(case)
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == {**baseline()["common"], **baseline()["cases"][case]["post_overrides"]}
    assert phase1_runtime.client.get("/openapi.json").json()["components"]["schemas"]["InspectionCreateResponse"] == baseline()["response_schema"]


def test_demo_cli_exports_reloads_infers_and_reports_actual_parity():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "scripts/export_segmentation_onnx.py", "--demo"], cwd=root,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["all_passed"] and result["fixed_shape_exports"] == 3 and result["parity_cases"] == 21
    assert not result["permanent_model_artifacts_written"] and not result["production_registry_modified"]
    assert not result["anomaly_converted"] and not result["next_phase_started"]
    for export in result["exports"]:
        assert export["reload_inference"]["passed"]
        assert export["manifest"]["synthetic"] and not export["manifest"]["production_approved"]
        assert export["parity"]["atol"] == export["parity"]["rtol"] == 1e-4
        for case in export["parity"]["cases"].values():
            assert case["mask"]["exact"]


@pytest.mark.parametrize("arguments", [["--checkpoint", "missing.pt"], ["--demo", "--output-dir", "forbidden"],
    ["--checkpoint", "missing.pt", "--expected-sha256", "a" * 64, "--model-version", "missing", "--output-dir", "unused", "--trusted-local"]])
def test_cli_invalid_arguments_or_missing_checkpoint_exit_nonzero_without_output(tmp_path, arguments):
    script = Path(__file__).resolve().parents[1] / "scripts/export_segmentation_onnx.py"
    completed = subprocess.run([sys.executable, str(script), *arguments], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert not list(tmp_path.iterdir())
