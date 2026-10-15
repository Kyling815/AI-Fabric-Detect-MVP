import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from backend.app.core.acquisition import AcquisitionMetadata
from backend.app.core.fusion import FusionBox
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.physical_coordinates import PixelPoint
from backend.app.services.coordinate_mapper import CalibratedCoordinateMapper, map_anomaly_boxes
from backend.app.services.qc_decision import QCDecisionEngine
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.physical_coordinate_fixtures import calibration, candidate, frame
from tests.qc_fixtures import qc_input, qc_policy


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_offline_mapping_preserves_golden_http_sqlite_schema_and_annotation(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(CalibratedCoordinateMapper, "map_candidate", side_effect=AssertionError("upload must not map by default")) as mapper_call:
        response = phase1_runtime.upload()
    mapper_call.assert_not_called()
    snapshot = baseline()
    assert response.status_code == 201
    payload = response.json()
    assert normalize_response(payload) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
    before = json.dumps(payload, sort_keys=True)
    original_hash = hashlib.sha256(phase1_runtime.client.get(payload["annotated_image_url"]).content).digest()
    call_count = phase1_runtime.extractor.extract_path.call_count
    boxes = [FusionBox(**item) for item in payload["boxes"]]
    context = frame()
    mapped = map_anomaly_boxes(CalibratedCoordinateMapper(calibration()), context, boxes)
    assert all(item.status == "mapped" for item in mapped) and len(mapped) == len(boxes)
    assert json.dumps(payload, sort_keys=True) == before
    assert original_hash == hashlib.sha256(phase1_runtime.client.get(payload["annotated_image_url"]).content).digest()
    assert phase1_runtime.extractor.extract_path.call_count == call_count
    saved = phase1_runtime.client.get(f"/api/inspections/{payload['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}


def test_bridge_preserves_raw_anomaly_scores_and_frame_identity():
    box = FusionBox(x=32, y=0, w=16, h=32, score=2.5, label="dense_anomaly")
    context = frame(frame_id="x" * 128)
    result, = map_anomaly_boxes(CalibratedCoordinateMapper(calibration()), context, [box])
    assert result.status == "mapped" and result.candidate.source.score == 2.5
    assert (result.candidate.width_mm, result.candidate.length_mm) == (8, 8)
    assert result.frame.frame_id == context.frame_id


@pytest.mark.parametrize("failure", ["model_grid", "mask_descriptor", "raw_dict", "nonfinite_score"])
def test_bridge_does_not_relabel_model_grid_masks_as_source_pixels(failure):
    box = FusionBox(x=4, y=8, w=4, h=4, score=1, label="dense_anomaly")
    if failure == "model_grid":
        box = box.model_copy(update={"coordinate_frame": "model_grid"})
    elif failure == "mask_descriptor":
        box = qc_input().evidence.supervised_mask
    elif failure == "raw_dict":
        box = box.model_dump()
    else:
        box = box.model_copy(update={"score": float("nan")})
    result, = map_anomaly_boxes(CalibratedCoordinateMapper(calibration()), frame(), [box])
    assert result.status == "unavailable" and result.point is None and result.candidate is None


def test_invalid_candidate_does_not_hide_valid_candidate_or_merge_ids():
    boxes = [FusionBox(x=x, y=0, w=4, h=4, score=1, label="dense_anomaly") for x in (0, 47, 42)]
    results = map_anomaly_boxes(CalibratedCoordinateMapper(calibration()), frame(), boxes)
    assert [item.status for item in results] == ["mapped", "unavailable", "mapped"]
    assert results[0].candidate.source.candidate_id != results[2].candidate.source.candidate_id


def test_existing_optional_encoder_metadata_has_no_assumed_count_units(tmp_path):
    inspection = InspectionInput(image_path=tmp_path / "not_loaded.png", fabric_id="fixture_fabric",
                                 metadata=AcquisitionMetadata(camera_id="lane_left", encoder_position=1100.0))
    result = CalibratedCoordinateMapper(calibration()).map_point(inspection, PixelPoint(x=4, y=8))
    assert result.status == "unavailable" and result.point is None
    assert not list(tmp_path.iterdir())


def test_physical_size_does_not_silently_change_phase11_model_grid_qc():
    inputs = qc_input()
    engine = QCDecisionEngine(qc_policy())
    before = engine.evaluate(inputs).model_dump_json()
    physical = CalibratedCoordinateMapper(calibration()).map_candidate(frame(), candidate())
    assert physical.status == "mapped" and engine.evaluate(inputs).model_dump_json() == before
    assert inputs.severities[0].coordinate_frame == "model_grid"


def test_read_only_cli_no_data_or_model_writes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / "scripts/check_phase12_coordinates.py"), "--demo"],
                         cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(run.stdout)
    assert report["case_count"] == 7 and report["status"] == "HYPOTHETICAL_SOFTWARE_CHECK_ONLY"
    assert not any(report[key] for key in ("writes_performed", "models_trained_or_loaded", "production_runtime_integrated",
                                          "vendor_sdk_integrated", "real_calibration_verified", "stream_runtime_started"))
    assert list(tmp_path.iterdir()) == []
