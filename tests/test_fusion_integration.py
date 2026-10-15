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

from backend.app.ai.fusion import FusionEngine
from backend.app.ai.preprocessing.quality import QualityMetrics
from backend.app.ai.quality_gate import QualityGateResult
from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.segmentation_shadow import SupervisedShadowObservation
from backend.app.services import inspection_service
from backend.app.services.inspection_service import InspectionService
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner
from tests.fusion_fixtures import policy
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.segmentation_shadow_fixtures import enable_shadow, observed_fixture


def enable(runtime, **updates):
    values = policy().model_dump(mode="json")
    values.update(updates)
    runtime.registry["fabrics"]["fixture_fabric"]["fusion"] = {"enabled": True, **values}
    runtime.save_registry()


def valid_capture_gate(runtime):
    runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True, "min_width": 32, "min_height": 32,
            "blur_threshold": 0, "max_dark_ratio": 1, "max_bright_ratio": 1}
    runtime.save_registry()


def test_default_off_skips_engine_adapter_optional_config_and_logging(phase1_runtime, monkeypatch, caplog):
    monkeypatch.setattr(get_settings(), "fusion_config_path", "missing_file.json")
    with patch.object(FusionEngine, "fuse", side_effect=AssertionError("OFF must not fuse")) as engine:
        service = InspectionService(None)
        context = service._resolve_fabric_context("fixture_fabric")
        result = service._inspect_fabric(InspectionInput(Path("unused.png"), "fixture_fabric"), context)
        response = phase1_runtime.upload()
    assert result.defect_evidence is None and response.status_code == 201
    engine.assert_not_called()
    assert not [record for record in caplog.records if hasattr(record, "defect_evidence")]


@pytest.mark.parametrize("case", list(baseline()["cases"]))
@pytest.mark.parametrize("state", ["known", "low", "missing", "timeout", "failed"])
def test_observe_only_fusion_preserves_golden_http_sqlite_and_schema_with_explicit_audit(phase1_runtime, case, state, caplog):
    phase1_runtime.use_case(case)
    enable(phase1_runtime)
    enable_shadow(phase1_runtime)
    valid_capture_gate(phase1_runtime)
    observed = observed_fixture()
    if state == "low":
        values = observed.model_dump(mode="json")
        values["classes"][1]["mean_winning_confidence"] = 0.2
        observed = SupervisedShadowObservation.model_validate(values)
    elif state in {"missing", "timeout", "failed"}:
        observed = SupervisedShadowObservation(status={"missing": "unavailable", "timeout": "timeout", "failed": "error"}[state],
                     reason="fixture_status", requested_model_version="fixture_shadow_v1")
    with caplog.at_level(logging.INFO), patch.object(ProcessSupervisedShadowRunner, "observe", return_value=observed):
        response = phase1_runtime.upload()
    assert response.status_code == 201, response.text
    snapshot = baseline()
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    assert schema["components"]["schemas"]["Body_inspect_image_api_inspect_post"] == snapshot["request_schema"]
    assert "defect_evidence" not in response.json()
    records = [record for record in caplog.records if hasattr(record, "defect_evidence")]
    assert len(records) == 1
    log = records[0].defect_evidence
    result = log["defect_evidence"]
    assert log["inspection_id"] == response.json()["inspection_id"] and log["qc_source"] == "legacy_anomaly"
    assert not log["hybrid_qc_active"] and not result["affects_qc"]
    if state == "known":
        assert result["kind"] == "KNOWN_DEFECT" and result["supervised_model_version"] == observed.model_version
    if state in {"missing", "timeout", "failed"}:
        assert result["review_required"] and result["kind"] != "NORMAL"
    assert result["policy"]["policy_version"] and result["trace"]


def test_post_qc_fusion_does_not_mutate_dense_evidence_candidates_or_decision(phase1_runtime, monkeypatch):
    value = InspectionInput(Path("unused.png"), "fixture_fabric")
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    off = service._inspect_fabric(value, context)
    order, decide = [], inspection_service.decide_fabric_inspection
    def qc(**kwargs):
        order.append("qc")
        return decide(**kwargs)
    monkeypatch.setattr(inspection_service, "decide_fabric_inspection", qc)
    runner = Mock(spec=ProcessSupervisedShadowRunner)
    runner.observe.return_value = observed_fixture()
    engine = FusionEngine(policy())
    fuse = engine.fuse
    def tracked(*args, **kwargs):
        order.append("fusion")
        return fuse(*args, **kwargs)
    engine.fuse = tracked
    from tests.segmentation_shadow_fixtures import shadow_config
    context = replace(context, fusion_policy=policy(), supervised_shadow_config=shadow_config())
    quality = QualityGateResult(QualityMetrics(decode_success=True))
    on = InspectionService(None, supervised_shadow=runner, fusion_engine=engine)._inspect_fabric(value, context, quality_gate_result=quality)
    assert order == ["qc", "fusion"] and on.defect_evidence.kind == "KNOWN_DEFECT"
    assert on.decision == off.decision and on.candidates == off.candidates and on.defect_area_ratio == off.defect_area_ratio
    np.testing.assert_array_equal(on.evidence.features.embeddings, off.evidence.features.embeddings)
    np.testing.assert_array_equal(on.evidence.anomaly.patch_scores, off.evidence.anomaly.patch_scores)


def test_fusion_without_shadow_or_quality_produces_review_not_normal(phase1_runtime, caplog):
    enable(phase1_runtime)
    with caplog.at_level(logging.INFO):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]
    result = next(record.defect_evidence for record in caplog.records if hasattr(record, "defect_evidence"))["defect_evidence"]
    assert result["kind"] == "NEEDS_REVIEW" and result["review_required"]
    assert "capture_unchecked" in result["uncertainties"] and "supervised_disabled" in result["uncertainties"]


@pytest.mark.parametrize("failure", ["missing_file", "corrupt_file", "nonobject_file", "invalid_thresholds", "invalid_mode", "invalid_flag", "nonobject_override"])
def test_bad_fusion_configuration_is_review_diagnostic_not_accepted_normal(phase1_runtime, monkeypatch, caplog, failure):
    enable(phase1_runtime)
    if failure.endswith("file"):
        path = phase1_runtime.root / "fusion.json"
        if failure != "missing_file":
            path.write_text("{bad" if failure == "corrupt_file" else "[]", encoding="utf-8")
        monkeypatch.setattr(get_settings(), "fusion_config_path", str(path))
    elif failure == "nonobject_override":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["fusion"] = []
        phase1_runtime.save_registry()
    else:
        enable(phase1_runtime, **{"invalid_thresholds": {"anomaly_high_threshold": 0.1}, "invalid_mode": {"mode": "active"}, "invalid_flag": {"enabled": "true"}}[failure])
    with caplog.at_level(logging.INFO):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]
    result = next(record.defect_evidence for record in caplog.records if hasattr(record, "defect_evidence"))["defect_evidence"]
    assert result["kind"] == "NEEDS_REVIEW" and result["review_required"] and result["rule_id"] == "fusion_configuration_error"


def test_global_config_override_and_other_fabric_off_isolation(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "fusion_enabled", True)
    path = phase1_runtime.root / "fusion.json"
    path.write_text(policy(known_threshold=0.95).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "fusion_config_path", str(path))
    service = InspectionService(None)
    assert service._resolve_fabric_context("fixture_fabric").fusion_policy.known_threshold == 0.95
    enable(phase1_runtime, known_threshold=0.8)
    other = dict(phase1_runtime.registry["fabrics"]["fixture_fabric"])
    other["fusion"] = {"enabled": False}
    phase1_runtime.registry["fabrics"]["other_fabric"] = other
    phase1_runtime.save_registry()
    assert service._resolve_fabric_context("fixture_fabric").fusion_policy.known_threshold == 0.8
    monkeypatch.setattr(get_settings(), "fusion_config_path", "missing_file.json")
    assert service._resolve_fabric_context("other_fabric").fusion_policy is None
    assert service._resolve_fabric_context("other_fabric").fusion_error is None


@pytest.mark.parametrize("failure", ["quality", "encoder", "registry", "decode"])
def test_existing_prior_failure_statuses_and_cleanup_skip_fusion(phase1_runtime, failure):
    enable(phase1_runtime)
    content = None
    if failure == "quality":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True}
        phase1_runtime.save_registry()
    elif failure == "encoder":
        phase1_runtime.factory.side_effect = RuntimeError("encoder fixture")
    elif failure == "registry":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["status"] = "INACTIVE"
        phase1_runtime.save_registry()
    else:
        content = b"invalid image"
    with patch.object(FusionEngine, "fuse", side_effect=AssertionError("prior failure must not fuse")) as engine:
        response = phase1_runtime.upload(content=content)
    assert response.status_code == {"quality": 400, "encoder": 500, "registry": 409, "decode": 400}[failure]
    engine.assert_not_called()
    assert not list((phase1_runtime.root / "uploads").iterdir()) and not list((phase1_runtime.root / "annotated").iterdir())


def test_engine_or_log_sink_failure_is_observable_review_not_qc_mutation(phase1_runtime, caplog):
    enable(phase1_runtime)
    with caplog.at_level(logging.INFO), patch.object(FusionEngine, "fuse", side_effect=RuntimeError("fixture private error")):
        response = phase1_runtime.upload()
    result = next(record.defect_evidence for record in caplog.records if hasattr(record, "defect_evidence"))["defect_evidence"]
    assert result["kind"] == "NEEDS_REVIEW" and result["review_required"] and result["rule_id"] == "fusion_engine_error"
    assert response.status_code == 201
    with patch.object(inspection_service.fusion_audit_logger, "info", side_effect=RuntimeError("sink fixture")):
        response = phase1_runtime.upload()
    assert response.status_code == 201 and normalize_response(response.json()) == baseline()["common"]


def test_observation_on_off_preserves_annotated_and_original_image_hashes(phase1_runtime):
    off = phase1_runtime.upload()
    enable(phase1_runtime)
    on = phase1_runtime.upload()
    assert normalize_response(off.json()) == normalize_response(on.json())
    for key in ("original_image_url", "annotated_image_url"):
        assert hashlib.sha256(phase1_runtime.client.get(off.json()[key]).content).digest() == hashlib.sha256(phase1_runtime.client.get(on.json()[key]).content).digest()


def test_read_only_cli_truth_table_from_other_cwd_creates_no_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / "scripts/check_phase10_fusion.py"), "--demo"], cwd=tmp_path,
                         capture_output=True, text=True, check=True, timeout=30)
    report = json.loads(run.stdout)
    assert report["case_count"] == 13 and not report["models_trained_or_loaded"] and not report["hybrid_qc_active"]
    assert all(not item["result"]["affects_qc"] for item in report["cases"])
    assert list(tmp_path.iterdir()) == []
