"""Read-only synthetic calibrated seam scenarios. No production policy activation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.physical_coordinates import EncoderReading, FrameCoordinates, PhysicalCalibration, PixelCandidate
from backend.app.core.seam import SeamObservation, SeamPolicy
from backend.app.services.seam_merger import SeamMerger


def demo():
    calibration = PhysicalCalibration.model_validate_json((PROJECT_ROOT / "config/physical_calibration_example.json").read_text(encoding="utf-8"))
    policy = SeamPolicy.model_validate_json((PROJECT_ROOT / "config/seam_merging_example.json").read_text(encoding="utf-8"))
    def item(camera, x, *, label="defect", y=8, w=4, identifier="candidate"):
        frame = FrameCoordinates(fabric_id="hypothetical_fabric", lot_id="hypothetical_lot", roll_id="hypothetical_roll",
            camera_id=camera, frame_id="hypothetical_" + camera, frame_index=1, image_width=48, image_height=32,
            coordinate_frame="oriented_source_pixels", position=EncoderReading(encoder_id="simulated_encoder", count=1100, frame_index=1, simulated=True))
        return SeamObservation(capture_group_id="hypothetical_capture_1", timestamp=datetime(2026, 10, 15, 2, tzinfo=timezone.utc),
            source_kind="supervised", model_version="hypothetical_model_v1", frame=frame,
            candidate=PixelCandidate(candidate_id=identifier, x=x, y=y, w=w, h=4, label=label, score=0.94))
    left, right = item("lane_left", 42), item("lane_right", 2)
    cases = [("same_defect_adjacent", calibration, [left, right], 1, 1),
        ("nearby_separate", calibration, [left, item("lane_right", 2, y=12)], 2, 0),
        ("conflicting_classes", calibration, [left, item("lane_right", 2, label="scratch")], 2, 0),
        ("edge_of_fov", calibration, [item("lane_left", 46, w=2), item("lane_right", 6)], 1, 1),
        ("no_calibration", None, [left, right], 2, 0),
        ("ambiguous_one_to_many", calibration, [left, right, item("lane_right", 2, identifier="another")], 3, 0)]
    output = []
    for name, profile, observations, expected_groups, expected_merged in cases:
        result = SeamMerger(profile, policy).merge(observations)
        if len(result.groups) != expected_groups or sum(group.merged for group in result.groups) != expected_merged:
            raise AssertionError("Seam demo mismatch: " + name)
        output.append({"case": name, "expected_groups": expected_groups, "expected_merged_groups": expected_merged,
                       "result": result.model_dump(mode="json")})
    return {"phase": 14, "status": "HYPOTHETICAL_SOFTWARE_CHECK_ONLY", "case_count": len(output), "cases": output,
        "production_runtime_integrated": False, "models_trained_or_loaded": False, "real_calibration_verified": False,
        "mask_union_implemented": False, "qc_changed": False, "writes_performed": False,
        "note": "Synthetic bbox grouping with exact source lineage. Starting match tolerances are NOT calibrated factory acceptance rules; envelope is not a mask/defect-area union. Phase 15 has not started."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    if parser.parse_args().demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
