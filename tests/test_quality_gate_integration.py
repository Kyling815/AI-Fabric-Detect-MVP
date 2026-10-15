from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock

from fastapi import HTTPException, UploadFile
import pytest
from sqlalchemy import func, select
from starlette.datastructures import Headers

from backend.app.ai.current_anomaly import CurrentAnomalyDetector
from backend.app.ai.quality_gate import CaptureStatus, ImageQualityGate, InvalidCaptureError, QualityGate
from backend.app.core.acquisition import AcquisitionMetadata
from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.db.models import Inspection
from backend.app.services.inspection_service import InspectionService
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.quality_gate_fixtures import corrupt_png_checksum, png_bytes, write_capture


def enable_fixture(runtime, **overrides) -> None:
    runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": True, **overrides}
    runtime.save_registry()


def assert_no_side_effects(runtime) -> None:
    runtime.factory.assert_not_called()
    assert not list((runtime.root / "uploads").iterdir())
    assert not list((runtime.root / "annotated").iterdir())
    with runtime.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Inspection)) == 0


@pytest.mark.parametrize("kind", ["black", "white", "blurred", "small", "corrupt", "bad_checksum"])
def test_enabled_gate_rejects_invalid_capture_before_ai_and_qc(phase1_runtime, kind: str) -> None:
    enable_fixture(phase1_runtime)
    if kind == "small":
        content = png_bytes("valid", 16, 16)
    elif kind == "corrupt":
        content = b"corrupted image"
    elif kind == "bad_checksum":
        content = corrupt_png_checksum(png_bytes("valid"))
    else:
        content = png_bytes(kind)
    response = phase1_runtime.upload(content=content)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("INVALID_CAPTURE: ")
    assert "final_result" not in response.json()
    assert_no_side_effects(phase1_runtime)


def test_enabled_gate_accepts_valid_capture_with_existing_response_schema(phase1_runtime) -> None:
    enable_fixture(phase1_runtime)
    response = phase1_runtime.upload(content=png_bytes("valid"))
    assert response.status_code == 201
    assert set(response.json()) == set(baseline()["common"])
    assert response.json()["final_result"] == "GOOD"
    phase1_runtime.factory.assert_called_once()


def test_flag_off_preserves_baseline_bad_checksum_failure(phase1_runtime) -> None:
    response = phase1_runtime.upload(content=corrupt_png_checksum(png_bytes("valid")))
    assert response.status_code == 500
    assert_no_side_effects(phase1_runtime)


def test_flag_off_skips_quality_even_for_bad_pixels_and_invalid_metadata(phase1_runtime) -> None:
    gate = Mock(spec=QualityGate)
    gate.evaluate.side_effect = AssertionError("disabled gate was called")
    service = InspectionService(None, quality_gate=gate)
    context = service._resolve_fabric_context("fixture_fabric")
    path = write_capture(phase1_runtime.root, "black")
    result = service._inspect_fabric(InspectionInput(path, "fixture_fabric", AcquisitionMetadata(encoder_position=-1)), context)
    gate.evaluate.assert_not_called()
    assert result.quality_gate_result is None
    assert result.decision.label == "GOOD"


def test_per_fabric_off_overrides_global_on_and_does_not_read_config(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "quality_gate_enabled", True)
    monkeypatch.setattr(get_settings(), "quality_gate_config_path", str(phase1_runtime.root / "missing_config.json"))
    phase1_runtime.registry["fabrics"]["fixture_fabric"]["quality_gate"] = {"enabled": False}
    phase1_runtime.save_registry()
    response = phase1_runtime.upload()
    assert response.status_code == 201
    snapshot = baseline()
    assert normalize_response(response.json()) == snapshot["common"]


def test_global_flag_and_profile_work_without_registry_edits(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "quality_gate_enabled", True)
    path = phase1_runtime.root / "profile.json"
    path.write_text(json.dumps({"min_width": 32, "min_height": 32, "blur_threshold": 0}), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "quality_gate_config_path", str(path))
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == baseline()["common"]


def test_per_fabric_overrides_profile_and_does_not_affect_other_fabric(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    path = phase1_runtime.root / "profile.json"
    path.write_text(json.dumps({"min_width": 256}), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "quality_gate_config_path", str(path))
    enable_fixture(phase1_runtime, min_width=64)
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    assert context.quality_gate_config.min_width == 64
    other = dict(phase1_runtime.registry["fabrics"]["fixture_fabric"])
    other.pop("quality_gate")
    other["display_name"] = "Other Fabric"
    phase1_runtime.registry["fabrics"]["other_fabric"] = other
    phase1_runtime.save_registry()
    assert service._resolve_fabric_context("other_fabric").quality_gate_config is None


@pytest.mark.parametrize("case", ["missing_file", "corrupt_file", "nonobject_file", "invalid_threshold", "invalid_flag"])
def test_invalid_enabled_configuration_fails_before_upload(phase1_runtime, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    enable_fixture(phase1_runtime)
    if case in {"missing_file", "corrupt_file", "nonobject_file"}:
        path = phase1_runtime.root / "bad_config.json"
        if case != "missing_file":
            path.write_text("{bad" if case == "corrupt_file" else "[]", encoding="utf-8")
        monkeypatch.setattr(get_settings(), "quality_gate_config_path", str(path))
    else:
        enable_fixture(phase1_runtime, **({"min_width": 0} if case == "invalid_threshold" else {"enabled": "false"}))
    response = phase1_runtime.upload()
    assert response.status_code == 409
    assert response.json()["detail"].startswith("Invalid quality gate configuration:")
    assert_no_side_effects(phase1_runtime)


def test_gate_failure_never_silently_becomes_good(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    enable_fixture(phase1_runtime)
    def fail(*args, **kwargs):
        raise RuntimeError("quality module failed")
    monkeypatch.setattr(ImageQualityGate, "evaluate", fail)
    response = phase1_runtime.upload()
    assert response.status_code == 500
    assert_no_side_effects(phase1_runtime)


def test_metadata_and_quality_evidence_flow_through_internal_orchestration(phase1_runtime) -> None:
    enable_fixture(phase1_runtime, check_metadata=True, required_metadata=["camera_id", "timestamp"], expected_camera_id="CAM_2")
    component = CurrentAnomalyDetector()
    detector = Mock(spec=CurrentAnomalyDetector)
    detector.inspect.side_effect = component.inspect
    service = InspectionService(None, anomaly_detector=detector)
    metadata = AcquisitionMetadata("LOT_1", "ROLL_1", "CAM_2", "FRAME_1", datetime(2026, 10, 13, tzinfo=timezone.utc), 100.5)
    value = InspectionInput(write_capture(phase1_runtime.root, "valid"), "fixture_fabric", metadata)
    context = service._resolve_fabric_context("fixture_fabric")
    result = service._inspect_fabric(value, context)
    detector.inspect.assert_called_once_with(value, context)
    assert detector.inspect.call_args.args[0].metadata is metadata
    assert result.quality_gate_result.status == CaptureStatus.VALID_CAPTURE


def test_invalid_metadata_blocks_internal_model_with_capture_error(phase1_runtime) -> None:
    enable_fixture(phase1_runtime, check_metadata=True, expected_camera_id="CAM_2")
    service = InspectionService(None)
    value = InspectionInput(write_capture(phase1_runtime.root, "valid"), "fixture_fabric", AcquisitionMetadata(camera_id="CAM_1"))
    with pytest.raises(InvalidCaptureError) as error:
        service._inspect_fabric(value, service._resolve_fabric_context("fixture_fabric"))
    assert error.value.result.status == CaptureStatus.INVALID_CAPTURE
    phase1_runtime.factory.assert_not_called()


def test_direct_upload_service_accepts_optional_metadata_without_api_changes(phase1_runtime) -> None:
    enable_fixture(phase1_runtime, check_metadata=True, required_metadata=["camera_id"], expected_camera_id="CAM_2")
    upload = UploadFile(BytesIO(png_bytes("valid")), filename="capture.png", headers=Headers({"content-type": "image/png"}))
    with phase1_runtime.sessions() as db:
        result = InspectionService(db).create_fabric_inspection(upload, "fixture_fabric", metadata=AcquisitionMetadata(camera_id="CAM_2"))
    assert result.final_result == "GOOD"
    assert set(result.model_dump()) == set(baseline()["common"])


def test_direct_service_invalid_capture_is_http_error_with_internal_evidence(phase1_runtime) -> None:
    enable_fixture(phase1_runtime)
    upload = UploadFile(BytesIO(png_bytes("black")), filename="capture.png", headers=Headers({"content-type": "image/png"}))
    with pytest.raises(HTTPException) as error:
        InspectionService(None).create_fabric_inspection(upload, "fixture_fabric")
    assert error.value.status_code == 400
    assert isinstance(error.value.__cause__, InvalidCaptureError)
    assert error.value.__cause__.result.status == CaptureStatus.INVALID_CAPTURE
    assert_no_side_effects(phase1_runtime)


def test_optional_roi_metadata_is_checked_on_upload_service(phase1_runtime) -> None:
    enable_fixture(phase1_runtime)
    upload = UploadFile(BytesIO(png_bytes("valid")), filename="capture.png", headers=Headers({"content-type": "image/png"}))
    with pytest.raises(HTTPException) as error:
        InspectionService(None).create_fabric_inspection(upload, "fixture_fabric", metadata=AcquisitionMetadata(roi=(-1, 0, 8, 8)))
    assert error.value.status_code == 400
    assert "invalid_roi" in error.value.__cause__.result.reasons
    assert_no_side_effects(phase1_runtime)


def test_quality_gate_runs_once_per_valid_uploaded_image(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    enable_fixture(phase1_runtime)
    calls = []
    evaluate = ImageQualityGate.evaluate
    def spy(self, *args):
        calls.append(args)
        return evaluate(self, *args)
    monkeypatch.setattr(ImageQualityGate, "evaluate", spy)
    response = phase1_runtime.upload(content=png_bytes("valid"))
    assert response.status_code == 201 and len(calls) == 1
