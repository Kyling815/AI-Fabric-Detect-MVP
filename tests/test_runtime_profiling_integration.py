from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backend.app.ai.dense_feature_extractor import SharedDenseResNet18
from backend.app.core.config import Settings, get_settings
from backend.app.core.fusion import FusionPolicy
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.runtime_profiling import RuntimeProfiler
from backend.app.services.inspection_service import InspectionService
from backend.app.services.runtime_scheduler import RuntimeScheduler
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


def test_profiling_default_off():
    assert Settings(_env_file=None).runtime_profiling_enabled is False


@pytest.mark.parametrize("case", list(baseline()["cases"]))
@pytest.mark.parametrize("enabled", [False, True])
def test_profile_flag_preserves_http_and_persisted_golden(phase1_runtime, monkeypatch, case, enabled):
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", enabled)
    phase1_runtime.use_case(case)
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == {**baseline()["common"], **baseline()["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get("/api/inspections/" + response.json()["inspection_id"])
    assert normalize_response(saved.json()) == {**baseline()["common"], **baseline()["cases"][case]["get_overrides"]}
    assert "runtime_profile" not in response.json()
    assert phase1_runtime.client.get("/openapi.json").json()["components"]["schemas"]["InspectionCreateResponse"] == baseline()["response_schema"]


def test_real_extractor_batch_one_instrumentation_preserves_features():
    import torch
    from PIL import Image
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = SharedDenseResNet18(device="cpu", use_pretrained_weights=False)
        image = Image.new("RGB", (48, 32), (100, 110, 120))
        expected = model.extract_image(image, image_max_size=48)
        with RuntimeProfiler() as profiler:
            actual = model.extract_image(image, image_max_size=48)
        np.testing.assert_array_equal(actual.embeddings, expected.embeddings)
        assert actual.boxes == expected.boxes
        value = profiler.profile.to_dict()
        assert value["stage_calls"]["preprocessing"] == 1
        assert value["stage_calls"]["anomaly"] == 1
        assert value["stage_calls"]["postprocessing"] == 1
        assert sum(value[name + "_ms"] for name in value["stage_calls"]) <= value["total_ms"]
    finally:
        torch.set_num_threads(previous)


def test_internal_profile_disabled_branch_is_not_measured_as_executed(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", True)
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    result = service._inspect_fabric(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)
    profile = result.runtime_profile.to_dict()
    assert profile["stage_calls"]["segmentation"] == profile["stage_calls"]["fusion"] == 0
    assert profile["queue_wait_ms"] == 0
    assert result.decision.label == baseline()["common"]["final_result"]


def test_profile_log_failure_does_not_fail_or_undo_persistence(phase1_runtime, monkeypatch):
    from backend.app.services.inspection_service import runtime_audit_logger
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", True)
    def fail(*args, **kwargs):
        raise OSError("Injected log failure")
    monkeypatch.setattr(runtime_audit_logger, "info", fail)
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert phase1_runtime.client.get("/api/inspections/" + response.json()["inspection_id"]).status_code == 200


def test_profile_audit_has_timings_but_no_input_paths(phase1_runtime, monkeypatch, caplog):
    from backend.app.services.inspection_service import runtime_audit_logger
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", True)
    with caplog.at_level("INFO", logger=runtime_audit_logger.name):
        assert phase1_runtime.upload().status_code == 201
    records = [record for record in caplog.records if record.name == runtime_audit_logger.name]
    assert len(records) == 1
    assert records[0].runtime_profile["total_ms"] > 0
    assert str(phase1_runtime.root) not in records[0].getMessage()


def test_scheduler_profiles_multiple_service_tiles_without_sharing_db(phase1_runtime):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    # Pure evidence/QC only. No upload, DB session or storage service mutation.
    with RuntimeScheduler(workers=2, max_pending=4) as scheduler:
        futures = [scheduler.submit(lambda: service._inspect_fabric(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)) for _ in range(4)]
        results = [future.result(timeout=10) for future in futures]
    assert len(results) == 4
    assert all(row.succeeded and row.value.decision.label == baseline()["common"]["final_result"] for row in results)
    assert all(row.profile.anomaly_ms > 0 for row in results)


@pytest.mark.parametrize("failure", ["encoder", "native_index", "persistence"])
def test_profile_enabled_preserves_errors_and_upload_cleanup(phase1_runtime, monkeypatch, failure):
    from backend.app.ai import current_anomaly
    from backend.app.services import inspection_service
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", True)
    def fail(*args, **kwargs):
        raise RuntimeError("Injected failure")
    if failure == "encoder":
        phase1_runtime.factory.side_effect = fail
    elif failure == "native_index":
        monkeypatch.setattr(current_anomaly, "DenseReferenceDetector", fail)
    else:
        monkeypatch.setattr(inspection_service.crud, "create_inspection_with_defects", fail)
    assert phase1_runtime.upload().status_code == 500
    assert not list((phase1_runtime.root / "uploads").iterdir())
    assert not list((phase1_runtime.root / "annotated").iterdir())


def test_handled_segmentation_failure_retains_anomaly_and_diagnostic(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "runtime_profiling_enabled", True)
    service = InspectionService(None)
    context = replace(service._resolve_fabric_context("fixture_fabric"), supervised_shadow_error="test_configuration_error",
                      fusion_policy=FusionPolicy())
    result = service._inspect_fabric(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)
    assert result.known_defects_shadow.status == "configuration_error"
    assert result.defect_evidence.supervised_status == "failed"
    assert result.decision.label == baseline()["common"]["final_result"]
    assert result.runtime_profile.to_dict()["stage_calls"]["segmentation"] == 1
    assert result.runtime_profile.to_dict()["stage_calls"]["fusion"] == 1
    # Errors intentionally converted to diagnostic values are not Python exceptions.
    assert result.runtime_profile.to_dict()["stage_errors"]["segmentation"] == 0


def test_measured_benchmark_cli_cpu_multiple_tiles_failure_and_pressure():
    import json
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/check_phase15_runtime.py"), "--benchmark",
               "--iterations", "2", "--warmups", "1", "--tiles", "3", "--size", "32"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["parameters"]["batch_size"] == 1
    assert result["output_parity"] == "PASS"
    assert result["gpu_contention"]["status"] == "NOT_MEASURED"
    assert result["multiple_tile_scheduler"]["snapshot"]["completed"] == 3
    assert result["queue_pressure_and_failure"]["admission_rejected"]
    assert result["queue_pressure_and_failure"]["failed_job_error_type"] == "RuntimeError"
    assert result["cpu_contention"]["summary"]["concurrent_pair_ms"]["count"] == 2
    assert not result["production_activation"] and not result["next_phase_started"]
