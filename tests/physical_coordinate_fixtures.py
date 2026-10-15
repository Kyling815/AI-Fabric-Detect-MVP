"""Explicit synthetic calibration, independent of model/data acquisition claims."""

from pathlib import Path

from backend.app.core.physical_coordinates import EncoderReading, FrameCoordinates, PhysicalCalibration, PixelCandidate


def calibration():
    root = Path(__file__).resolve().parents[1]
    return PhysicalCalibration.model_validate_json((root / "config/physical_calibration_example.json").read_text(encoding="utf-8"))


def frame(camera="lane_left", count=1100, **updates):
    fields = dict(fabric_id="fixture_fabric", camera_id=camera, frame_id="fixture_frame_1", frame_index=1,
        image_width=48, image_height=32, coordinate_frame="oriented_source_pixels",
        position=EncoderReading(encoder_id="simulated_encoder", frame_index=1, count=count, simulated=True),
        roll_id="fixture_roll", lot_id="fixture_lot")
    fields.update(updates)
    return FrameCoordinates(**fields)


def candidate(**updates):
    fields = dict(candidate_id="fixture_candidate", x=4, y=8, w=4, h=4, label="suspicious_defect", score=1.2)
    fields.update(updates)
    return PixelCandidate(**fields)
