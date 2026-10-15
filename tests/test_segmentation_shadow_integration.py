from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.segmentation_shadow import SupervisedShadowObservation
from backend.app.services import inspection_service
from backend.app.services.inspection_service import InspectionService
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.segmentation_shadow_fixtures import enable_shadow, observed_fixture, shadow_checkpoint, shadow_config


def test_default_off_skips_config_model_worker_and_logging(phase1_runtime, monkeypatch, caplog):
    monkeypatch.setattr(get_settings(), "supervised_shadow_config_path", "does_not_exist.json")
    runner = Mock(spec=ProcessSupervisedShadowRunner)
    service = InspectionService(None, supervised_shadow=runner)
    context = service._resolve_fabric_context("fixture_fabric")
    result = service._inspect_fabric(InspectionInput(Path("unused.png"), "fixture_fabric"), context)
    assert context.supervised_shadow_config is None and result.known_defects_shadow is None
    runner.observe.assert_not_called()
    with patch.object(ProcessSupervisedShadowRunner, "observe", side_effect=AssertionError("OFF must not call")):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]
    assert not [record for record in caplog.records if hasattr(record, "known_defects_shadow")]


@pytest.mark.parametrize("case", list(baseline()["cases"]))
@pytest.mark.parametrize("state", ["observed", "unavailable", "timeout", "error", "busy", "raises"])
def test_all_shadow_states_preserve_golden_http_sqlite_and_schema(phase1_runtime, case, state, caplog):
    phase1_runtime.use_case(case)
    config = shadow_config()
    enable_shadow(phase1_runtime, config)
    observation = observed_fixture(config) if state == "observed" else SupervisedShadowObservation(
                    status="error" if state == "raises" else state, reason="fixture_status", requested_model_version=config.expected_model_version)
    with caplog.at_level(logging.INFO), patch.object(ProcessSupervisedShadowRunner, "observe", return_value=observation,
                    side_effect=RuntimeError("private fixture error") if state == "raises" else None):
        response = phase1_runtime.upload()
    snapshot = baseline()
    assert response.status_code == 201, response.text
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    assert schema["components"]["schemas"]["Body_inspect_image_api_inspect_post"] == snapshot["request_schema"]
    assert "known_defects_shadow" not in response.json() and "known_defects_shadow" not in saved.json()
    records = [record for record in caplog.records if hasattr(record, "known_defects_shadow")]
    assert len(records) == 1
    payload = records[0].known_defects_shadow
    assert payload["inspection_id"] == response.json()["inspection_id"] and payload["anomaly_reference_id"] == "fixture_ref_v1"
    assert payload["status"] == ("error" if state == "raises" else state) and not payload["affects_qc"]
    assert "mask_rle" not in payload and "private fixture error" not in records[0].getMessage()
    if state == "observed":
        assert payload["model_version"] == config.expected_model_version
        assert payload["classes"][1]["predicted_pixels"] == 1024
        assert payload["classes"][1]["mean_winning_confidence"] == 0.99
    else:
        assert payload["model_version"] is None and not payload["classes"]


def test_shadow_is_post_qc_internal_only_and_cannot_mutate_anomaly_evidence(phase1_runtime, monkeypatch):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(Path("unused.png"), "fixture_fabric")
    off = service._inspect_fabric(value, context)
    order, decide = [], inspection_service.decide_fabric_inspection
    def tracked_decide(**kwargs):
        order.append("decision")
        return decide(**kwargs)
    monkeypatch.setattr(inspection_service, "decide_fabric_inspection", tracked_decide)
    runner = Mock(spec=ProcessSupervisedShadowRunner)
    def observe(*args):
        order.append("shadow")
        return observed_fixture()
    runner.observe.side_effect = observe
    on = InspectionService(None, supervised_shadow=runner)._inspect_fabric(value, replace(context, supervised_shadow_config=shadow_config()))
    assert order == ["decision", "shadow"] and on.known_defects_shadow.status == "observed"
    assert on.decision == off.decision and on.candidates == off.candidates and on.defect_area_ratio == off.defect_area_ratio
    np.testing.assert_array_equal(on.evidence.features.embeddings, off.evidence.features.embeddings)
    np.testing.assert_array_equal(on.evidence.anomaly.patch_scores, off.evidence.anomaly.patch_scores)
    assert on.evidence.extractor_info == off.evidence.extractor_info and on.evidence.anomaly.image_score == off.evidence.anomaly.image_score
    assert on.evidence.anomaly.localized_score == off.evidence.anomaly.localized_score


@pytest.mark.parametrize("failure", ["missing_file", "corrupt_file", "nonobject_file", "invalid_mode", "invalid_flag", "nonobject_override", "missing_pins"])
def test_bad_shadow_configuration_never_rejects_valid_inspection(phase1_runtime, monkeypatch, failure, caplog):
    enable_shadow(phase1_runtime)
    if failure.endswith("file"):
        path = phase1_runtime.root / "shadow.json"
        if failure != "missing_file":
            path.write_text("{bad" if failure == "corrupt_file" else "[]", encoding="utf-8")
        monkeypatch.setattr(get_settings(), "supervised_shadow_config_path", str(path))
    elif failure == "nonobject_override":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["supervised"] = []
        phase1_runtime.save_registry()
    elif failure == "missing_pins":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["supervised"] = {"enabled": True, "checkpoint_path": "new.pt"}
        phase1_runtime.save_registry()
    else:
        enable_shadow(phase1_runtime, **({"mode": "active"} if failure == "invalid_mode" else {"enabled": "true"}))
    context = InspectionService(None)._resolve_fabric_context("fixture_fabric")
    assert context.supervised_shadow_error == "configuration_error"
    with caplog.at_level(logging.INFO), patch.object(ProcessSupervisedShadowRunner, "observe", side_effect=AssertionError("invalid policy must not launch")):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]
    record = next(record for record in caplog.records if hasattr(record, "known_defects_shadow"))
    assert record.known_defects_shadow["status"] == "configuration_error"


def test_global_config_per_fabric_override_and_off_fabric_skips_missing_file(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "supervised_shadow_enabled", True)
    path = phase1_runtime.root / "shadow.json"
    path.write_text(shadow_config(timeout_seconds=12).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "supervised_shadow_config_path", str(path))
    service = InspectionService(None)
    assert service._resolve_fabric_context("fixture_fabric").supervised_shadow_config.timeout_seconds == 12
    phase1_runtime.registry["fabrics"]["fixture_fabric"]["supervised"] = {"enabled": True, "timeout_seconds": 2}
    other = dict(phase1_runtime.registry["fabrics"]["fixture_fabric"])
    other["supervised"] = {"enabled": False}
    phase1_runtime.registry["fabrics"]["other_fabric"] = other
    phase1_runtime.save_registry()
    assert service._resolve_fabric_context("fixture_fabric").supervised_shadow_config.timeout_seconds == 2
    monkeypatch.setattr(get_settings(), "supervised_shadow_config_path", "does_not_exist.json")
    context = service._resolve_fabric_context("other_fabric")
    assert context.supervised_shadow_config is None and context.supervised_shadow_error is None


@pytest.mark.parametrize("failure", ["quality", "encoder", "registry", "corrupt_image"])
def test_prior_anomaly_acquisition_failures_skip_shadow_and_keep_failure_response(phase1_runtime, failure, monkeypatch):
    enable_shadow(phase1_runtime)
    content = None
    if failure == "quality":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True}
        phase1_runtime.save_registry()
    elif failure == "encoder":
        phase1_runtime.factory.side_effect = RuntimeError("encoder fixture failure")
    elif failure == "registry":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["status"] = "INACTIVE"
        phase1_runtime.save_registry()
    else:
        content = b"bad capture"
    with patch.object(ProcessSupervisedShadowRunner, "observe", side_effect=AssertionError("prior failures must skip shadow")) as runner:
        response = phase1_runtime.upload(content=content)
    assert response.status_code == {"quality": 400, "encoder": 500, "registry": 409, "corrupt_image": 400}[failure]
    runner.assert_not_called()
    assert not list((phase1_runtime.root / "uploads").iterdir()) and not list((phase1_runtime.root / "annotated").iterdir())


def test_bad_injected_observer_payload_isolated(phase1_runtime):
    runner = Mock(spec=ProcessSupervisedShadowRunner)
    runner.observe.return_value = observed_fixture().model_copy(update={"affects_qc": True})
    service = InspectionService(None, supervised_shadow=runner)
    context = replace(service._resolve_fabric_context("fixture_fabric"), supervised_shadow_config=shadow_config())
    result = service._inspect_fabric(InspectionInput(Path("unused.png"), "fixture_fabric"), context)
    assert result.known_defects_shadow.status == "error" and result.decision.label == "GOOD"


def test_optional_log_sink_failure_cannot_turn_success_into_http_error(phase1_runtime):
    enable_shadow(phase1_runtime)
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=observed_fixture()), patch.object(inspection_service.shadow_audit_logger, "info", side_effect=RuntimeError("sink fixture")):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]


@pytest.mark.parametrize("case", ["good", "bad"])
def test_real_isolated_model_with_extreme_foreground_keeps_anomaly_decision_and_images(phase1_runtime, shadow_checkpoint, case, caplog):
    _, config = shadow_checkpoint
    phase1_runtime.use_case(case)
    off = phase1_runtime.upload()
    enable_shadow(phase1_runtime, config)
    with caplog.at_level(logging.INFO):
        on = phase1_runtime.upload()
    assert off.status_code == on.status_code == 201
    assert normalize_response(off.json()) == normalize_response(on.json())
    for key in ("original_image_url", "annotated_image_url"):
        assert hashlib.sha256(phase1_runtime.client.get(off.json()[key]).content).digest() == hashlib.sha256(phase1_runtime.client.get(on.json()[key]).content).digest()
    record = next(record for record in caplog.records if hasattr(record, "known_defects_shadow"))
    assert record.known_defects_shadow["status"] == "observed"
    assert record.known_defects_shadow["classes"][1]["predicted_pixels"] == 1024
    assert record.known_defects_shadow["classes"][1]["mean_winning_confidence"] > 0.99
    assert record.known_defects_shadow["model_version"] == config.expected_model_version


def test_real_subprocess_timeout_preserves_api_success(phase1_runtime, shadow_checkpoint, caplog):
    _, config = shadow_checkpoint
    enable_shadow(phase1_runtime, config, timeout_seconds=0.05)
    with caplog.at_level(logging.INFO):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]
    record = next(record for record in caplog.records if hasattr(record, "known_defects_shadow"))
    assert record.known_defects_shadow["status"] == "timeout" and not record.known_defects_shadow["classes"]


def test_cli_temporary_shadow_demo_creates_no_permanent_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / "scripts/check_phase9_supervised_shadow.py"), "--demo"],
                         cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    report = json.loads(run.stdout)
    assert report["observed"]["status"] == "observed" and report["deterministic_mask_across_workers"]
    assert report["timeout"]["status"] == "timeout" and report["unavailable"]["status"] == "unavailable"
    assert not report["artifacts_persisted"] and not report["qc_integration"] and not report["production_accuracy_evaluated"]
    assert list(tmp_path.iterdir()) == []
