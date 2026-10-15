import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from backend.app.core.physical_coordinates import PixelPoint
from backend.app.services.coordinate_mapper import CalibratedCoordinateMapper
from backend.app.services.line_scan_runtime import SimulatedStreamRuntime
from tests.line_scan_fixtures import run_to_end, scan_config
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.physical_coordinate_fixtures import calibration


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_separate_stream_runtime_preserves_golden_upload_response_database_and_images(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(SimulatedStreamRuntime, "step", side_effect=AssertionError("upload must not start simulated stream")) as step:
        response = phase1_runtime.upload()
    step.assert_not_called()
    assert response.status_code == 201
    expected = baseline()
    assert normalize_response(response.json()) == {**expected["common"], **expected["cases"][case]["post_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == expected["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == expected["response_schema"]
    image_hash = hashlib.sha256(phase1_runtime.client.get(response.json()["annotated_image_url"]).content).digest()
    calls = phase1_runtime.extractor.extract_path.call_count
    output = run_to_end(SimulatedStreamRuntime(scan_config()))
    assert len(output) == 3 and sum(bundle.line_count for bundle in output) == 19
    assert phase1_runtime.extractor.extract_path.call_count == calls
    assert image_hash == hashlib.sha256(phase1_runtime.client.get(response.json()["annotated_image_url"]).content).digest()
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**expected["common"], **expected["cases"][case]["get_overrides"]}


def test_chunk_first_row_anchors_reuse_phase12_mapper_only_with_explicit_calibration():
    config = scan_config(camera_ids=("lane_left", "lane_right", "lane_3", "lane_4"), chunk_height=32, total_lines=33)
    full, partial = run_to_end(SimulatedStreamRuntime(config))
    fields = calibration().model_dump(mode="json")
    fields["transport"]["mm_per_encoder_count"] = 0.125  # counts_per_line=2 → row pitch=.25 mm.
    for name, origin in (("lane_3", 40), ("lane_4", 60)):
        fields["cameras"].append({**fields["cameras"][0], "camera_id": name, "x_origin_mm": origin})
    mapper = CalibratedCoordinateMapper(fields)
    for lane, chunk in enumerate(full.chunks):
        frame = chunk.to_frame_coordinates()
        result = mapper.map_point(frame, PixelPoint(x=0, y=0))
        assert result.status == "mapped" and result.point.x_mm == lane * 20 and result.point.y_mm == 102
        assert frame.roll_id == config.roll_id and frame.lot_id == config.lot_id
        assert CalibratedCoordinateMapper(None).map_point(frame, PixelPoint(x=0, y=0)).point is None
    for chunk in partial.chunks:
        result = mapper.map_point(chunk.to_frame_coordinates(), PixelPoint(x=0, y=0))
        assert result.status == "unavailable" and result.reason == "CALIBRATED_GEOMETRY_MISMATCH"


def test_no_disk_or_model_activity_in_cli_and_report_is_deterministic(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/check_phase13_stream.py"), "--demo"]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["camera_count"] == 4 and report["chunk_count"] == 12 and report["bundle_count"] == 3
    assert report["runtime"]["delivered_line_ticks"] == 19 and report["runtime"]["backpressure_events"] > 0
    assert not any(report[key] for key in ("production_api_integrated", "models_trained_or_loaded", "vendor_sdk_integrated",
                                          "physical_calibration_inferred", "writes_performed", "seam_merging_implemented"))
    assert list(tmp_path.iterdir()) == []


def test_runtime_does_not_mutate_config_and_chunk_metadata_is_payload_free():
    config = scan_config()
    before = config.model_dump_json()
    chunks = [chunk for bundle in run_to_end(SimulatedStreamRuntime(config)) for chunk in bundle.chunks]
    assert config.model_dump_json() == before
    assert len({chunk.chunk_id for chunk in chunks}) == 12
    for chunk in chunks:
        assert "rgb_bytes" not in chunk.metadata()
        assert json.loads(json.dumps(chunk.metadata()))["rgb_sha256"] == hashlib.sha256(chunk.rgb_bytes).hexdigest()
        with pytest.raises(ValueError):
            chunk.encoder_start = 0
