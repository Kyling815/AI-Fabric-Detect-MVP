"""Read-only hypothetical coordinate demonstration; no runtime flags, hardware or IO writes."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.physical_coordinates import PhysicalCalibration, PixelCandidate, PixelPoint, SimulatedLinePosition
from backend.app.services.coordinate_mapper import CalibratedCoordinateMapper, camera_overlaps
from backend.app.services.simulated_acquisition import (SimulatedCameraConfig, SimulatedCameraSource,
                                                       SimulatedEncoderConfig, SimulatedEncoderSource)


def demo():
    calibration = PhysicalCalibration.model_validate_json((PROJECT_ROOT / "config/physical_calibration_example.json").read_text(encoding="utf-8"))
    encoder = SimulatedEncoderSource(SimulatedEncoderConfig(encoder_id="simulated_encoder", start_count=1000, counts_per_frame=100))
    cameras = [SimulatedCameraSource(SimulatedCameraConfig(camera_id=item.camera_id, fabric_id="hypothetical_fabric",
        image_width=item.image_width, image_height=item.image_height, frame_prefix=item.camera_id, roll_id="hypothetical_roll",
        lot_id="hypothetical_lot")) for item in calibration.cameras]
    frames = [source.capture(1, encoder.read(1)).frame for source in cameras]
    mapper = CalibratedCoordinateMapper(calibration)
    left = mapper.map_candidate(frames[0], PixelCandidate(candidate_id="left_candidate", x=42, y=8, w=4, h=4, label="suspicious_defect", score=1.2))
    right = mapper.map_candidate(frames[1], PixelCandidate(candidate_id="right_candidate", x=2, y=8, w=4, h=4, label="suspicious_defect", score=1.2))
    if left.candidate is None or right.candidate is None or left.candidate.minimum != right.candidate.minimum:
        raise AssertionError("Overlapping camera coordinate mismatch")
    line_frame = cameras[0].capture(1, SimulatedLinePosition(frame_index=1, line_index=40)).frame
    cases = [("encoder_count_origin", mapper.map_point(frames[0], PixelPoint(x=0, y=0)), "mapped"),
        ("simulated_line_origin", mapper.map_point(line_frame, PixelPoint(x=0, y=0)), "mapped"),
        ("last_pixel_edge", mapper.map_point(frames[0], PixelPoint(x=48, y=32)), "mapped"),
        ("left_overlap_candidate", left, "mapped"), ("right_overlap_candidate", right, "mapped"),
        ("no_calibration", CalibratedCoordinateMapper(None).map_point(frames[0], PixelPoint(x=4, y=8)), "unavailable"),
        ("invalid_calibration", CalibratedCoordinateMapper({}).map_point(frames[0], PixelPoint(x=4, y=8)), "unavailable")]
    if cases[0][1].point.y_mm != 112 or cases[1][1].point != cases[0][1].point:
        raise AssertionError("Transport unit conversion mismatch")
    outputs = []
    for name, result, expected in cases:
        if result.status != expected:
            raise AssertionError("Coordinate demo mismatch: " + name)
        record = result.model_dump(mode="json")
        if result.point is not None:
            record["y_m"] = result.point.y_m
        outputs.append({"case": name, "expected_status": expected, "result": record})
    return {"phase": 12, "status": "HYPOTHETICAL_SOFTWARE_CHECK_ONLY", "case_count": len(outputs), "cases": outputs,
        "overlap_regions": [item.model_dump(mode="json") for item in camera_overlaps(calibration)],
        "real_calibration_verified": False, "production_runtime_integrated": False, "vendor_sdk_integrated": False,
        "stream_runtime_started": False, "writes_performed": False, "models_trained_or_loaded": False,
        "note": "Explicit synthetic linear calibration only. Overlapping candidates retain separate IDs; no merging or physical QC policy. Source edge coordinates, synchronized first-row transport anchor, no perspective/lens/slip compensation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    if parser.parse_args().demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
