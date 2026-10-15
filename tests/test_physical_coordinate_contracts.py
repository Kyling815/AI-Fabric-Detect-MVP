import pytest

from backend.app.core.physical_coordinates import (CoordinateMappingResult, EncoderReading, FrameCoordinates,
    PhysicalCalibration, PhysicalPoint, PixelCandidate, PixelPoint, SimulatedLinePosition)
from tests.physical_coordinate_fixtures import calibration, candidate, frame


@pytest.mark.parametrize("field,value", [("x_mm_per_pixel", 0), ("y_mm_per_pixel", -1), ("x_mm_per_pixel", True),
    ("y_mm_per_pixel", "0.25"), ("x_origin_mm", float("nan")), ("y_offset_mm", float("inf")),
    ("x_direction", True), ("y_direction", 0), ("image_width", False), ("image_height", 0),
    ("coordinate_frame", "model_grid"), ("pixel_convention", "pixel_centres"), ("camera_id", " "),
    ("x_mm_per_pixel", 1e308), ("x_origin_mm", 1e308), ("y_offset_mm", 1e308)])
def test_invalid_camera_calibration_rejected(field, value):
    fields = calibration().model_dump(mode="json")
    fields["cameras"][0][field] = value
    with pytest.raises(ValueError):
        PhysicalCalibration.model_validate(fields)


@pytest.mark.parametrize("field,value", [("mm_per_encoder_count", 0), ("mm_per_line", -1), ("encoder_origin_count", 1.5),
    ("mm_per_encoder_count", "0.1"), ("mm_per_line", True), ("y_origin_mm", float("inf")),
    ("line_origin_index", -1), ("travel_direction", False), ("encoder_id", " encoder ")])
def test_invalid_transport_units_rejected(field, value):
    fields = calibration().model_dump(mode="json")
    fields["transport"][field] = value
    with pytest.raises(ValueError):
        PhysicalCalibration.model_validate(fields)


@pytest.mark.parametrize("failure", ["no_camera", "duplicate_lane", "no_version", "no_world_frame", "implicit_units"])
def test_calibration_must_be_explicit_traceable_and_unique(failure):
    fields = calibration().model_dump(mode="json")
    if failure == "no_camera":
        fields["cameras"] = []
    elif failure == "duplicate_lane":
        fields["cameras"].append(fields["cameras"][0])
    elif failure == "no_version":
        fields.pop("calibration_version")
    elif failure == "no_world_frame":
        fields.pop("world_frame_id")
    else:
        fields["transport"].pop("mm_per_encoder_count")
    with pytest.raises(ValueError):
        PhysicalCalibration.model_validate(fields)


@pytest.mark.parametrize("failure", ["wrong_frame", "untyped_position", "bad_units", "model_grid", "invalid_simulated_line"])
def test_frame_never_infers_units_or_synchronization(failure):
    fields = frame().model_dump(mode="json")
    if failure == "wrong_frame":
        fields["position"]["frame_index"] = 2
    elif failure == "untyped_position":
        fields["position"] = 1100
    elif failure == "bad_units":
        fields["position"]["kind"] = "millimetres"
    elif failure == "model_grid":
        fields["coordinate_frame"] = "model_grid"
    else:
        fields["position"] = dict(kind="simulated_line", line_index=40, frame_index=1, simulated=False)
    with pytest.raises(ValueError):
        FrameCoordinates.model_validate(fields)


@pytest.mark.parametrize("field,value", [("x", -1), ("y", float("nan")), ("x", True), ("w", 0), ("h", -1),
    ("coordinate_frame", "model_grid"), ("score", float("inf")), ("score", "0.5")])
def test_source_candidates_require_finite_explicit_pixels(field, value):
    fields = candidate().model_dump(mode="json")
    fields[field] = value
    with pytest.raises(ValueError):
        PixelCandidate.model_validate(fields)


@pytest.mark.parametrize("failure", ["unavailable_with_point", "mapped_without_calibration", "mapped_without_frame", "mapped_without_payload"])
def test_output_cannot_claim_uncalibrated_physical_coordinates(failure):
    fields = dict(status="mapped", reason="fixture", calibration=calibration(), frame=frame(), point=PhysicalPoint(x_mm=2, y_mm=114))
    if failure == "unavailable_with_point":
        fields["status"] = "unavailable"
    else:
        fields.pop({"mapped_without_calibration": "calibration", "mapped_without_frame": "frame", "mapped_without_payload": "point"}[failure])
    with pytest.raises(ValueError):
        CoordinateMappingResult(**fields)


@pytest.mark.parametrize("value", [True, "1000", 1.5])
def test_encoder_count_is_not_a_float_metadata_guess(value):
    with pytest.raises(ValueError):
        EncoderReading(encoder_id="simulated_encoder", frame_index=1, count=value, simulated=True)


def test_snapshot_contracts_are_frozen():
    with pytest.raises(ValueError):
        calibration().transport.mm_per_encoder_count = 10
    assert PixelPoint(x=0, y=0).x == 0
    assert SimulatedLinePosition(frame_index=0, line_index=0).simulated is True
