from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.segmentation_shadow import SupervisedShadowObservation
from backend.app.services import inspection_service
from backend.app.services.inspection_service import InspectionService
from backend.app.services.qc_decision import QCDecisionEngine
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.qc_fixtures import qc_policy
from tests.segmentation_shadow_fixtures import enable_shadow, observed_fixture


def shadow(pixels=128, confidence=0.99):
    fields = observed_fixture().model_dump(mode="json")
    background = 1024 - pixels
    runs = ([[0, background]] if background else []) + ([[1, pixels]] if pixels else [])
    raw = b"\x00\x00\x00\x00" * background + b"\x01\x00\x00\x00" * pixels
    fields.update(mask_rle=runs, mask_sha256=hashlib.sha256(raw).hexdigest(), classes=[
        {"name": "background", "predicted_pixels": background, "mean_class_probability": (background * confidence + pixels * (1 - confidence)) / 1024,
         "mean_winning_confidence": confidence if background else None},
        {"name": "defect", "predicted_pixels": pixels, "mean_class_probability": (pixels * confidence + background * (1 - confidence)) / 1024,
         "mean_winning_confidence": confidence if pixels else None}])
    return SupervisedShadowObservation.model_validate(fields)


def enable_qc(runtime, *, mode="observe_only", dependencies=True, **updates):
    values = qc_policy().model_dump(mode="json")
    values.update(updates)
    record = runtime.registry["fabrics"]["fixture_fabric"]
    record["qc"] = {"enabled": True, "mode": mode, **values}
    if dependencies:
        from tests.fusion_fixtures import policy
        record["fusion"] = {"enabled": True, **policy(anomaly_high_threshold=2.0, anomaly_review_threshold=1.5,
            anomaly_score_method=runtime.reference["thresholds"]["image_score_method"]).model_dump(mode="json")}
        record["quality_gate"] = {"enabled": True, "min_width": 32, "min_height": 32, "blur_threshold": 0, "max_dark_ratio": 1, "max_bright_ratio": 1}
        enable_shadow(runtime)
    runtime.save_registry()


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_default_off_preserves_all_golden_http_and_schema(phase1_runtime, monkeypatch, case):
    phase1_runtime.use_case(case)
    monkeypatch.setattr(get_settings(), "hybrid_qc_config_path", "missing.json")
    with patch.object(QCDecisionEngine, "evaluate", side_effect=AssertionError("OFF must not evaluate")) as engine:
        response = phase1_runtime.upload()
    snapshot = baseline()
    assert response.status_code == 201
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    engine.assert_not_called()


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_observe_policy_can_reject_while_legacy_http_and_sqlite_stay_exact(phase1_runtime, case, caplog):
    phase1_runtime.use_case(case)
    enable_qc(phase1_runtime)
    with caplog.at_level(logging.INFO), patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow(512)):
        response = phase1_runtime.upload()
    snapshot = baseline()
    assert response.status_code == 201
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    record = next(record.qc_evaluation for record in caplog.records if hasattr(record, "qc_evaluation"))
    assert record["qc_evaluation"]["label"] == "BAD" and not record["qc_evaluation"]["applied"]
    assert record["selected_qc_label"] == response.json()["final_result"]


@pytest.mark.parametrize("pixels,label,reason", [(128, "GOOD", "QC_KNOWN_SEVERITY_ALLOWED"),
                 (300, "NEEDS_REVIEW", "QC_KNOWN_SEVERITY_BORDERLINE_OR_UNMAPPED"), (512, "BAD", "QC_KNOWN_SEVERITY_REJECT")])
def test_enforce_explicit_business_severity_changes_labels_and_persists_them(phase1_runtime, caplog, pixels, label, reason):
    enable_qc(phase1_runtime, mode="enforce")
    with caplog.at_level(logging.INFO), patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow(pixels)):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and response.json()["final_result"] == label and response.json()["reason"] == reason
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert saved.json()["final_result"] == label and saved.json()["reason"] == reason
    assert response.json()["reference_id"] == "fixture_ref_v1" and response.json()["confidence"] is None
    assert "qc_evaluation" not in response.json()
    audit = next(record.qc_evaluation for record in caplog.records if hasattr(record, "qc_evaluation"))
    assert audit["qc_evaluation"]["applied"] and audit["qc_evaluation"]["runtime_mode"] == "enforce"
    fused = next(record.defect_evidence for record in caplog.records if hasattr(record, "defect_evidence"))
    assert fused["hybrid_qc_active"] and fused["qc_source"] == "hybrid_qc"
    assert audit["legacy_qc_label"] == "GOOD"


def test_unknown_high_can_reject_only_by_explicit_qc_rule(phase1_runtime):
    enable_qc(phase1_runtime, mode="enforce", unknown_high_action="BAD")
    record = phase1_runtime.registry["fabrics"]["fixture_fabric"]
    record["fusion"].update(anomaly_high_threshold=0.7, anomaly_review_threshold=0.3)
    phase1_runtime.save_registry()
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow(confidence=0.6)):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and response.json()["final_result"] == "BAD"
    assert response.json()["reason"] == "QC_UNKNOWN_HIGH_ANOMALY"


@pytest.mark.parametrize("failure", ["no_fusion", "timeout", "busy", "worker_error", "uncalibrated", "synthetic", "unmapped", "engine_error"])
def test_enforce_uncertainty_never_silently_falls_back_to_legacy_good(phase1_runtime, failure):
    enable_qc(phase1_runtime, mode="enforce", dependencies=failure != "no_fusion")
    observation = shadow()
    if failure in {"timeout", "busy", "worker_error"}:
        observation = SupervisedShadowObservation(status="error" if failure == "worker_error" else failure, reason="fixture",
                                                   requested_model_version="fixture_shadow_v1")
    elif failure in {"uncalibrated", "synthetic", "unmapped"}:
        key, value = {"uncalibrated": ("allow_uncalibrated_supervised", False), "synthetic": ("allow_synthetic_evidence", False), "unmapped": ("known_rules", [])}[failure]
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["qc"][key] = value
        phase1_runtime.save_registry()
    engine_override = patch.object(QCDecisionEngine, "evaluate", side_effect=RuntimeError("fixture")) if failure == "engine_error" else nullcontext()
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=observation), engine_override:
        response = phase1_runtime.upload()
    assert response.status_code == 201 and response.json()["final_result"] == "NEEDS_REVIEW", response.text
    expected_reason = {"uncalibrated": "QC_SUPERVISED_CONFIDENCE_UNCALIBRATED", "synthetic": "QC_SYNTHETIC_OR_UNKNOWN_PROVENANCE",
                       "unmapped": "QC_KNOWN_SEVERITY_BORDERLINE_OR_UNMAPPED", "engine_error": "QC_EVALUATION_ERROR"}
    if failure in expected_reason:
        assert response.json()["reason"] == expected_reason[failure]


@pytest.mark.parametrize("failure", ["missing_file", "corrupt_file", "nonobject_file", "wrong_rule_unit", "empty_scope", "wrong_scope", "invalid_flag", "invalid_mode"])
def test_bad_enforcement_config_is_fail_closed_review(phase1_runtime, monkeypatch, failure):
    enable_qc(phase1_runtime, mode="enforce")
    if failure.endswith("file"):
        path = phase1_runtime.root / "qc.json"
        if failure != "missing_file":
            path.write_text("{bad" if failure == "corrupt_file" else "[]", encoding="utf-8")
        monkeypatch.setattr(get_settings(), "hybrid_qc_config_path", str(path))
    else:
        record = phase1_runtime.registry["fabrics"]["fixture_fabric"]["qc"]
        if failure == "wrong_rule_unit":
            record["known_rules"][0]["unit"] = "mm2"
        else:
            key, value = {"empty_scope": ("allowed_fabric_ids", []), "wrong_scope": ("allowed_fabric_ids", ["other_fabric"]),
                          "invalid_flag": ("enabled", "true"), "invalid_mode": ("mode", [])}[failure]
            record[key] = value
        phase1_runtime.save_registry()
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow()):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and response.json()["final_result"] == "NEEDS_REVIEW", response.text


def test_invalid_capture_keeps_existing_http_400_not_bad_fabric(phase1_runtime):
    enable_qc(phase1_runtime, mode="enforce")
    phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True}
    phase1_runtime.save_registry()
    with patch.object(QCDecisionEngine, "evaluate", side_effect=AssertionError("invalid capture should reject before inference")) as engine:
        response = phase1_runtime.upload()
    assert response.status_code == 400
    engine.assert_not_called()
    assert not list((phase1_runtime.root / "uploads").iterdir())


def test_global_policy_mode_and_explicit_other_fabric_disable(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "hybrid_qc_enabled", True)
    monkeypatch.setattr(get_settings(), "hybrid_qc_mode", "enforce")
    path = phase1_runtime.root / "qc.json"
    path.write_text(qc_policy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "hybrid_qc_config_path", str(path))
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    assert context.hybrid_qc_mode == "enforce" and context.qc_policy == qc_policy()
    other = dict(phase1_runtime.registry["fabrics"]["fixture_fabric"])
    other["qc"] = {"enabled": False}
    phase1_runtime.registry["fabrics"]["other_fabric"] = other
    phase1_runtime.save_registry()
    monkeypatch.setattr(get_settings(), "hybrid_qc_config_path", "missing.json")
    context = service._resolve_fabric_context("other_fabric")
    assert context.qc_policy is None and context.qc_policy_error is None and context.hybrid_qc_mode == "observe_only"


def test_enforcement_does_not_mutate_evidence_weights_or_annotation_geometry(phase1_runtime):
    off = phase1_runtime.upload()
    enable_qc(phase1_runtime, mode="enforce")
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow(512)):
        on = phase1_runtime.upload()
    assert off.json()["final_result"] == "GOOD" and on.json()["final_result"] == "BAD"
    assert off.json()["anomaly_score"] == on.json()["anomaly_score"] and off.json()["boxes"] == on.json()["boxes"]
    for key in ("original_image_url", "annotated_image_url"):
        assert hashlib.sha256(phase1_runtime.client.get(off.json()[key]).content).digest() == hashlib.sha256(phase1_runtime.client.get(on.json()[key]).content).digest()


def test_qc_log_sink_failure_cannot_undo_enforced_rejection(phase1_runtime):
    enable_qc(phase1_runtime, mode="enforce")
    with patch.object(ProcessSupervisedShadowRunner, "observe", return_value=shadow(512)), patch.object(inspection_service.qc_audit_logger, "info", side_effect=RuntimeError("sink")):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and response.json()["final_result"] == "BAD"


def test_cli_read_only_hypothetical_rules_do_not_modify_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / "scripts/check_phase11_qc.py"), "--demo"], cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(run.stdout)
    assert report["case_count"] == 9 and not report["production_policy_activated"] and not report["writes_performed"]
    assert not report["models_trained_or_loaded"] and list(tmp_path.iterdir()) == []
