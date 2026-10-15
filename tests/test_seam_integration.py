import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from backend.app.core.physical_coordinates import PixelCandidate
from backend.app.services.line_scan_runtime import SimulatedStreamRuntime
from backend.app.services.qc_decision import QCDecisionEngine
from backend.app.services.seam_merger import SeamMerger, seam_observations_from_chunks
from tests.line_scan_fixtures import run_to_end, scan_config
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.qc_fixtures import qc_input, qc_policy
from tests.seam_fixtures import calibration, duplicate_pair, seam_policy


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_offline_seam_merging_preserves_golden_api_database_annotations_and_schema(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(SeamMerger, "merge", side_effect=AssertionError("upload must not merge seams")) as seam_call:
        response = phase1_runtime.upload()
    seam_call.assert_not_called()
    assert response.status_code == 201
    expected = baseline()
    assert normalize_response(response.json()) == {**expected["common"], **expected["cases"][case]["post_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == expected["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == expected["response_schema"]
    before = hashlib.sha256(phase1_runtime.client.get(response.json()["annotated_image_url"]).content).digest()
    calls = phase1_runtime.extractor.extract_path.call_count
    assert SeamMerger(calibration(), seam_policy()).merge(duplicate_pair()).groups[0].merged
    assert before == hashlib.sha256(phase1_runtime.client.get(response.json()["annotated_image_url"]).content).digest()
    assert phase1_runtime.extractor.extract_path.call_count == calls
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**expected["common"], **expected["cases"][case]["get_overrides"]}


def stream_examples():
    config = scan_config(camera_ids=("lane_left", "lane_right", "lane_3", "lane_4"), chunk_height=32, total_lines=33)
    return run_to_end(SimulatedStreamRuntime(config))


def pixels():
    return {camera: [PixelCandidate(candidate_id="anomaly_0", x=x, y=0, w=4, h=1, label="dense_anomaly", score=2.5)]
            for camera, x in (("lane_left", 42), ("lane_right", 2))}


def bridge(bundle, candidates=None):
    return seam_observations_from_chunks(bundle, pixels() if candidates is None else candidates,
        source_kind="anomaly", model_version="hypothetical_model", reference_version="hypothetical_reference")


def test_phase13_bridge_preserves_camera_chunk_lineage_and_phase12_calibration_guards():
    full, partial = stream_examples()
    before = full.model_dump()
    items = bridge(full)
    fields = calibration().model_dump(mode="json")
    fields["transport"]["mm_per_encoder_count"] = 0.125  # Explicit coherent .25 mm scan-row pitch.
    engine = SeamMerger(fields, seam_policy())
    result = engine.merge(items)
    assert result.groups[0].merged and len(result.observations) == 2
    assert {item.frame.frame_id for item in result.observations} == {chunk.chunk_id for chunk in full.chunks[:2]}
    assert len({item.capture_group_id for item in items}) == 1
    assert full.model_dump() == before
    tail = engine.merge(bridge(partial))
    assert len(tail.groups) == 2 and all(not group.merged and group.minimum is None for group in tail.groups)
    assert all(mapping.reason == "CALIBRATED_GEOMETRY_MISMATCH" for mapping in tail.mappings)
    mixed = engine.merge([items[0], bridge(partial)[1]])
    assert len(mixed.groups) == 2 and not any(group.merged for group in mixed.groups)


@pytest.mark.parametrize("failure", ["unknown_camera", "raw_dict", "model_grid", "too_many", "generator_values", "bad_chunk_payload"])
def test_chunk_bridge_rejects_unsafe_input_instead_of_guessing_or_dropping(failure):
    full, _ = stream_examples()
    candidates = pixels()
    if failure == "unknown_camera":
        candidates["unknown"] = []
    elif failure == "raw_dict":
        candidates["lane_left"][0] = candidates["lane_left"][0].model_dump()
    elif failure == "model_grid":
        candidates["lane_left"][0] = candidates["lane_left"][0].model_copy(update={"coordinate_frame": "model_grid"})
    elif failure == "too_many":
        candidates["lane_left"] *= 257
    elif failure == "generator_values":
        candidates["lane_left"] = iter(candidates["lane_left"])
    else:
        full = full.model_copy(update={"chunks": (full.chunks[0].model_copy(update={"rgb_bytes": b"bad"}), *full.chunks[1:])})
    with pytest.raises(ValueError):
        bridge(full, candidates)


def test_seam_grouping_does_not_recalculate_qc_severity_or_labels():
    inputs, engine = qc_input(), QCDecisionEngine(qc_policy())
    before = engine.evaluate(inputs).model_dump_json()
    assert SeamMerger(calibration(), seam_policy()).merge(duplicate_pair()).affects_qc is False
    assert engine.evaluate(inputs).model_dump_json() == before


def test_read_only_synthetic_cli_has_six_required_scenarios_and_is_repeatable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/check_phase14_seam.py"), "--demo"]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout and list(tmp_path.iterdir()) == []
    report = json.loads(first.stdout)
    assert report["case_count"] == 6 and report["status"] == "HYPOTHETICAL_SOFTWARE_CHECK_ONLY"
    assert not any(report[key] for key in ("production_runtime_integrated", "models_trained_or_loaded", "real_calibration_verified",
                                          "mask_union_implemented", "qc_changed", "writes_performed"))
