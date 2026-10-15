from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy import func, select

from backend.app.ai.current_anomaly import CurrentAnomalyDetector
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.inspection_contracts import InspectionInput, InspectionResult
from backend.app.db.models import Inspection
from backend.app.services import inspection_service
from backend.app.services.inspection_service import InspectionService
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_post_and_persisted_response_match_pre_refactor_snapshot(phase1_runtime, case: str) -> None:
    phase1_runtime.use_case(case)
    snapshot = baseline()
    response = phase1_runtime.upload()
    assert response.status_code == 201, response.text
    value = response.json()
    assert normalize_response(value) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{value['inspection_id']}")
    assert saved.status_code == 200
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    for field in ("original_image_url", "annotated_image_url"):
        assert phase1_runtime.client.get(value[field]).status_code == 200
    with phase1_runtime.sessions() as db:
        stored = db.get(Inspection, value["inspection_id"])
        assert stored.fabric_id == "fixture_fabric"
        assert stored.decision_reason == value["reason"]
        assert [(box.x, box.y, box.w, box.h, box.score) for box in stored.defects] == [
            (box["x"], box["y"], box["w"], box["h"], box["score"]) for box in value["boxes"]
        ]


def test_openapi_request_response_contract_matches_baseline(phase1_runtime) -> None:
    schema = phase1_runtime.client.get("/openapi.json").json()
    snapshot = baseline()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    assert schema["components"]["schemas"]["Body_inspect_image_api_inspect_post"] == snapshot["request_schema"]


def test_service_orchestration_can_use_callable_component_without_http_or_db(phase1_runtime) -> None:
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    evidence = CurrentAnomalyDetector().inspect(InspectionInput(Path("input.png"), "fixture_fabric"), context)
    component = Mock(spec=CurrentAnomalyDetector)
    component.inspect.return_value = evidence
    service = InspectionService(None, anomaly_detector=component)
    value = InspectionInput(Path("input.png"), "fixture_fabric")
    result = service._inspect_fabric(value, context)
    component.inspect.assert_called_once_with(value, context)
    assert isinstance(result, InspectionResult)
    assert result.decision.label == baseline()["common"]["final_result"]
    assert [(b.x, b.y, b.w, b.h) for b in result.candidates] == [
        (b["x"], b["y"], b["w"], b["h"]) for b in baseline()["common"]["boxes"]
    ]
    assert result.defect_area_ratio == baseline()["common"]["defect_area_ratio"]


@pytest.mark.parametrize("case", ["unknown", "inactive", "missing_active", "inactive_reference", "backbone_mismatch",
                                  "missing_metadata", "missing_bank", "missing_index", "incomplete_thresholds"])
def test_registry_failure_is_rejected_before_upload_and_inference(phase1_runtime, case: str) -> None:
    fabric = phase1_runtime.registry["fabrics"]["fixture_fabric"]
    if case == "inactive":
        fabric["status"] = "INACTIVE"
    elif case == "missing_active":
        fabric["active_reference"] = None
    elif case == "inactive_reference":
        phase1_runtime.reference["status"] = "CALIBRATED"
    elif case == "backbone_mismatch":
        phase1_runtime.reference["backbone_version"] = "other_backbone"
    elif case.startswith("missing_"):
        key = {"missing_metadata": "metadata_path", "missing_bank": "memory_bank_path", "missing_index": "faiss_index_path"}[case]
        Path(phase1_runtime.reference[key]).unlink()
    elif case == "incomplete_thresholds":
        phase1_runtime.reference["thresholds"].pop("pixel_threshold")
    phase1_runtime.save_registry()
    response = phase1_runtime.upload("missing_fabric" if case == "unknown" else "fixture_fabric")
    assert response.status_code == (400 if case == "unknown" else 409)
    phase1_runtime.factory.assert_not_called()
    assert not list((phase1_runtime.root / "uploads").iterdir())


@pytest.mark.parametrize("case, expected_status", [("corrupt_image", 400), ("corrupt_metadata", 409),
                                                    ("wrong_dimension", 409), ("corrupt_faiss", 500),
                                                    ("encoder_failure", 500), ("persistence_failure", 500)])
def test_failed_inspection_preserves_status_and_cleanup(phase1_runtime, monkeypatch: pytest.MonkeyPatch,
                                                      case: str, expected_status: int) -> None:
    content = None
    if case == "corrupt_image":
        content = b"not an image"
    elif case == "corrupt_metadata":
        Path(phase1_runtime.reference["metadata_path"]).write_text("{invalid", encoding="utf-8")
    elif case == "wrong_dimension":
        FaissIndexStore().save(np.array([[1, 0, 0]], dtype=np.float32), Path(phase1_runtime.reference["faiss_index_path"]))
    elif case == "corrupt_faiss":
        Path(phase1_runtime.reference["faiss_index_path"]).write_bytes(b"invalid native index")
    elif case == "encoder_failure":
        phase1_runtime.factory.side_effect = ValueError("encoder failed")
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("persistence failed")
        monkeypatch.setattr(inspection_service.crud, "create_inspection_with_defects", fail)
    response = phase1_runtime.upload(content=content)
    assert response.status_code == expected_status
    assert not list((phase1_runtime.root / "uploads").iterdir())
    assert not list((phase1_runtime.root / "annotated").iterdir())
    with phase1_runtime.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Inspection)) == 0
    if case == "corrupt_image":
        phase1_runtime.factory.assert_not_called()
