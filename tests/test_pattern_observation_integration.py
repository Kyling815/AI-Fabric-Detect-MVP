from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from backend.app.ai.pattern_period import PatternPeriodEstimator, PatternPeriodResult
from backend.app.ai.quality_gate import InvalidCaptureError
from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.pattern import PatternEstimatorConfig
from backend.app.services import inspection_service
from backend.app.services.inspection_service import InspectionService
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.pattern_period_fixtures import png_bytes


def enable(runtime, **values):
    runtime.registry["fabrics"]["fixture_fabric"]["pattern_estimator"] = {"enabled": True, **values}
    runtime.save_registry()


def test_default_off_skips_observer_and_missing_configuration(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "pattern_estimator_config_path", "does_not_exist.json")
    observer = Mock(spec=PatternPeriodEstimator)
    service = InspectionService(None, pattern_estimator=observer)
    context = service._resolve_fabric_context("fixture_fabric")
    result = service._inspect_fabric(InspectionInput(Path("unused.png"), "fixture_fabric"), context)
    assert result.pattern_period_result is None
    observer.estimate_path.assert_not_called()


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_observer_on_preserves_all_frozen_http_and_persisted_decisions(phase1_runtime, case):
    phase1_runtime.use_case(case)
    enable(phase1_runtime)
    snapshot = baseline()
    response = phase1_runtime.upload()
    assert response.status_code == 201, response.text
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    assert schema["components"]["schemas"]["Body_inspect_image_api_inspect_post"] == snapshot["request_schema"]


@pytest.mark.parametrize("kind", ["checker", "stripes", "dots", "uniform", "noise", "stretch", "lighting", "rotate"])
def test_real_observer_on_off_preserves_http_and_annotated_image(phase1_runtime, kind):
    content = png_bytes(kind)
    off = phase1_runtime.upload(content=content)
    enable(phase1_runtime)
    on = phase1_runtime.upload(content=content)
    assert off.status_code == on.status_code == 201
    assert normalize_response(off.json()) == normalize_response(on.json())
    for key in ("original_image_url", "annotated_image_url"):
        off_bytes = phase1_runtime.client.get(off.json()[key]).content
        on_bytes = phase1_runtime.client.get(on.json()[key]).content
        assert hashlib.sha256(off_bytes).digest() == hashlib.sha256(on_bytes).digest()


def test_observation_is_internal_and_does_not_modify_dense_evidence(phase1_runtime, monkeypatch):
    path = phase1_runtime.root / "capture.png"
    path.write_bytes(png_bytes())
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(path, "fixture_fabric")
    off = service._inspect_fabric(value, context)
    order = []
    decision_function = inspection_service.decide_fabric_inspection
    def decide(**kwargs):
        order.append("decision")
        return decision_function(**kwargs)
    monkeypatch.setattr(inspection_service, "decide_fabric_inspection", decide)
    observer = Mock(spec=PatternPeriodEstimator)
    def observe(*args):
        order.append("observe")
        return PatternPeriodEstimator().estimate_path(*args)
    observer.estimate_path.side_effect = observe
    on_service = InspectionService(None, pattern_estimator=observer)
    on_context = replace(context, pattern_estimator_config=PatternEstimatorConfig())
    on = on_service._inspect_fabric(value, on_context)
    assert order == ["decision", "observe"]
    observer.estimate_path.assert_called_once_with(path, on_context.pattern_estimator_config)
    assert (on.pattern_period_result.period_x_px, on.pattern_period_result.period_y_px) == (16, 24)
    assert on.decision == off.decision and on.candidates == off.candidates
    assert on.defect_area_ratio == off.defect_area_ratio
    np.testing.assert_array_equal(on.evidence.features.embeddings, off.evidence.features.embeddings)
    np.testing.assert_array_equal(on.evidence.anomaly.patch_scores, off.evidence.anomaly.patch_scores)
    assert on.evidence.features.boxes == off.evidence.features.boxes
    assert on.evidence.anomaly.image_score == off.evidence.anomaly.image_score
    assert on.evidence.anomaly.localized_score == off.evidence.anomaly.localized_score
    assert on.evidence.extractor_info == off.evidence.extractor_info


def test_observer_failure_cannot_change_qc_or_http(phase1_runtime, monkeypatch):
    enable(phase1_runtime)
    observer = Mock(spec=PatternPeriodEstimator)
    observer.estimate_path.side_effect = RuntimeError("observer failure")
    service = InspectionService(None, pattern_estimator=observer)
    context = service._resolve_fabric_context("fixture_fabric")
    result = service._inspect_fabric(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)
    assert result.pattern_period_result == PatternPeriodResult(reason="observer_error")
    assert result.decision.label == "GOOD"
    monkeypatch.setattr(PatternPeriodEstimator, "estimate_path", observer.estimate_path)
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == baseline()["common"]


@pytest.mark.parametrize("case", ["missing_file", "corrupt_file", "nonobject_file", "invalid_config",
                                  "invalid_mode", "invalid_flag", "nonobject_override"])
def test_observer_configuration_failure_never_rejects_inspection(phase1_runtime, monkeypatch, case):
    enable(phase1_runtime)
    if case.endswith("file"):
        path = phase1_runtime.root / "observation.json"
        if case != "missing_file":
            path.write_text("{bad" if case == "corrupt_file" else "[]", encoding="utf-8")
        monkeypatch.setattr(get_settings(), "pattern_estimator_config_path", str(path))
    elif case == "nonobject_override":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["pattern_estimator"] = []
        phase1_runtime.save_registry()
    else:
        enable(phase1_runtime, **{"invalid_config": {"band_count": 0}, "invalid_mode": {"mode": "adaptive"},
                                "invalid_flag": {"enabled": "true"}}[case])
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    assert context.pattern_observation_error == "configuration_error"
    result = service._inspect_fabric(InspectionInput(Path("unused.png"), "fixture_fabric"), context)
    assert result.pattern_period_result.reason == "configuration_error"
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == baseline()["common"]


def test_global_config_per_fabric_override_and_other_fabric_isolation(phase1_runtime, monkeypatch):
    monkeypatch.setattr(get_settings(), "pattern_estimator_enabled", True)
    path = phase1_runtime.root / "observation.json"
    path.write_text(json.dumps({"min_period_px": 8, "max_period_px": 100}), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "pattern_estimator_config_path", str(path))
    service = InspectionService(None)
    assert service._resolve_fabric_context("fixture_fabric").pattern_estimator_config.min_period_px == 8
    enable(phase1_runtime, min_period_px=4)
    other = dict(phase1_runtime.registry["fabrics"]["fixture_fabric"])
    other["pattern_estimator"] = {"enabled": False}
    phase1_runtime.registry["fabrics"]["other_fabric"] = other
    phase1_runtime.save_registry()
    assert service._resolve_fabric_context("fixture_fabric").pattern_estimator_config.min_period_px == 4
    assert service._resolve_fabric_context("other_fabric").pattern_estimator_config is None
    monkeypatch.setattr(get_settings(), "pattern_estimator_config_path", "missing.json")
    assert service._resolve_fabric_context("other_fabric").pattern_observation_error is None


@pytest.mark.parametrize("case", ["quality", "encoder", "registry", "corrupt_image"])
def test_prior_failures_keep_status_and_skip_observer(phase1_runtime, monkeypatch, case):
    enable(phase1_runtime)
    observer = Mock(return_value=PatternPeriodResult())
    monkeypatch.setattr(PatternPeriodEstimator, "estimate_path", observer)
    content = None
    if case == "quality":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True}
        phase1_runtime.save_registry()  # Existing 48x32 capture rejected by Phase 2.
    elif case == "encoder":
        phase1_runtime.factory.side_effect = RuntimeError("encoder failure")
    elif case == "registry":
        phase1_runtime.registry["fabrics"]["fixture_fabric"]["status"] = "INACTIVE"
        phase1_runtime.save_registry()
    else:
        content = b"not an image"
    response = phase1_runtime.upload(content=content)
    assert response.status_code == {"quality": 400, "encoder": 500, "registry": 409, "corrupt_image": 400}[case]
    observer.assert_not_called()
    assert not list((phase1_runtime.root / "uploads").iterdir())
    assert not list((phase1_runtime.root / "annotated").iterdir())


def test_quality_error_precedes_observation_in_internal_path(phase1_runtime):
    observer = Mock(spec=PatternPeriodEstimator)
    service = InspectionService(None, pattern_estimator=observer)
    phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True}
    enable(phase1_runtime)
    path = phase1_runtime.root / "small.png"
    path.write_bytes(phase1_runtime.png)
    with pytest.raises(InvalidCaptureError):
        service._inspect_fabric(InspectionInput(path, "fixture_fabric"), service._resolve_fabric_context("fixture_fabric"))
    observer.estimate_path.assert_not_called()
