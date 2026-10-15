from copy import deepcopy
from itertools import product

import pytest

from backend.app.core.physical_coordinates import (EncoderReading, PhysicalCalibration, PixelPoint, SimulatedLinePosition)
from backend.app.services.coordinate_mapper import CalibratedCoordinateMapper, CoordinateMapper, camera_overlaps
from tests.physical_coordinate_fixtures import calibration, candidate, frame


def test_known_calibration_fixture_and_metre_conversion():
    mapper = CalibratedCoordinateMapper(calibration())
    assert isinstance(mapper, CoordinateMapper)
    result = mapper.map_point(frame(), PixelPoint(x=4, y=8))
    assert result.status == "mapped" and (result.point.x_mm, result.point.y_mm, result.point.y_m) == (2, 114, 0.114)
    assert result.calibration.calibration_version == "hypothetical_linear_fixture_v1"
    assert result.frame.roll_id == "fixture_roll" and result.frame.lot_id == "fixture_lot"


@pytest.mark.parametrize("count,expected", [(1000, 102), (1100, 112), (900, 92), (0, 2), (-100, -8)])
def test_encoder_conversion_relative_to_explicit_origin(count, expected):
    result = CalibratedCoordinateMapper(calibration()).map_point(frame(count=count), PixelPoint(x=0, y=0))
    assert result.status == "mapped" and result.point.y_mm == pytest.approx(expected)


@pytest.mark.parametrize("line,expected", [(0, 102), (1, 102.25), (40, 112), (4000, 1102)])
def test_simulated_line_conversion_is_separate_and_calibrated(line, expected):
    context = frame(position=SimulatedLinePosition(frame_index=1, line_index=line))
    result = CalibratedCoordinateMapper(calibration()).map_point(context, PixelPoint(x=0, y=0))
    assert result.status == "mapped" and result.point.y_mm == pytest.approx(expected)


@pytest.mark.parametrize("x_direction,y_direction,travel_direction", list(product((-1, 1), repeat=3)))
def test_camera_offsets_and_reversed_axes_are_explicit(x_direction, y_direction, travel_direction):
    fields = calibration().model_dump(mode="json")
    fields["cameras"][0].update(x_origin_mm=10, y_offset_mm=-4, x_direction=x_direction, y_direction=y_direction)
    fields["transport"]["travel_direction"] = travel_direction
    mapper = CalibratedCoordinateMapper(fields)
    point = mapper.map_point(frame(), PixelPoint(x=4, y=8)).point
    assert point.x_mm == 10 + x_direction * 2
    assert point.y_mm == 100 + travel_direction * 10 - 4 + y_direction * 2
    box = mapper.map_candidate(frame(), candidate()).candidate
    assert box.width_mm == 2 and box.length_mm == 1
    assert box.minimum.x_mm < box.maximum.x_mm and box.minimum.y_mm < box.maximum.y_mm


@pytest.mark.parametrize("x,y", list(product((0, 0.5, 47.5, 48), (0, 0.5, 31.5, 32))))
def test_boundary_edges_and_pixel_centres(x, y):
    result = CalibratedCoordinateMapper(calibration()).map_point(frame(), PixelPoint(x=x, y=y))
    assert result.status == "mapped" and result.point.x_mm == x * 0.5 and result.point.y_mm == 112 + y * 0.25


@pytest.mark.parametrize("x,y", [(48.001, 0), (0, 32.001), (1000, 1)])
def test_outside_pixels_are_not_clipped_into_a_physical_claim(x, y):
    result = CalibratedCoordinateMapper(calibration()).map_point(frame(), PixelPoint(x=x, y=y))
    assert result.status == "unavailable" and result.reason == "PIXEL_OUTSIDE_CALIBRATED_FRAME"
    assert result.point is None and result.candidate is None


@pytest.mark.parametrize("values,expected", [({}, "mapped"), ({"x": 0, "y": 0, "w": 48, "h": 32}, "mapped"),
    ({"x": 47, "w": 1}, "mapped"), ({"x": 48}, "unavailable"), ({"y": 31, "h": 2}, "unavailable")])
def test_half_open_candidate_edges_and_extents(values, expected):
    result = CalibratedCoordinateMapper(calibration()).map_candidate(frame(), candidate(**values))
    assert result.status == expected
    if expected == "mapped":
        assert result.candidate.width_mm == candidate(**values).w * 0.5
        assert result.candidate.length_mm == candidate(**values).h * 0.25


@pytest.mark.parametrize("failure", ["no_calibration", "invalid_calibration", "wrong_camera", "wrong_size", "wrong_encoder",
    "wrong_frame", "wrong_coordinate_frame", "forged_calibration", "nonfinite_point", "overflow_count", "precision_loss"])
def test_unavailable_mapping_never_contains_physical_values(failure):
    policy, context, point = calibration(), frame(), PixelPoint(x=4, y=8)
    if failure == "no_calibration":
        policy = None
    elif failure == "invalid_calibration":
        policy = {}
    elif failure == "wrong_camera":
        context = frame(camera="unknown")
    elif failure == "wrong_size":
        context = frame(image_width=24)
    elif failure == "wrong_encoder":
        context = frame(position=EncoderReading(encoder_id="other", frame_index=1, count=1100, simulated=True))
    elif failure == "wrong_frame":
        context = context.model_copy(update={"frame_index": 2})
    elif failure == "wrong_coordinate_frame":
        context = context.model_copy(update={"coordinate_frame": "model_grid"})
    elif failure == "forged_calibration":
        policy = policy.model_copy(update={"cameras": (policy.cameras[0].model_copy(update={"x_mm_per_pixel": -1}),)})
    elif failure == "nonfinite_point":
        point = point.model_copy(update={"x": float("nan")})
    elif failure == "overflow_count":
        context = frame(count=10 ** 1000)
    else:
        fields = policy.model_dump(mode="json")
        fields["transport"]["y_origin_mm"] = 1e308
        policy = fields
    result = CalibratedCoordinateMapper(policy).map_point(context, point)
    assert result.status == "unavailable" and result.point is None and result.candidate is None


def test_lane_overlap_candidates_agree_in_world_without_merging():
    policy = calibration()
    mapper = CalibratedCoordinateMapper(policy)
    overlaps = camera_overlaps(policy)
    assert len(overlaps) == 1 and (overlaps[0].x_min_mm, overlaps[0].x_max_mm) == (20, 24)
    left = mapper.map_candidate(frame(), candidate(candidate_id="left", x=42)).candidate
    right = mapper.map_candidate(frame(camera="lane_right"), candidate(candidate_id="right", x=2)).candidate
    assert left.minimum == right.minimum and left.maximum == right.maximum
    assert left.overlap_camera_ids == ("lane_right",) and right.overlap_camera_ids == ("lane_left",)
    assert left.source.candidate_id != right.source.candidate_id


@pytest.mark.parametrize("x,w,expected", [(0, 4, ()), (38, 2, ()), (39, 2, ("lane_right",)), (46, 2, ("lane_right",))])
def test_overlap_requires_positive_intersection_not_just_touching(x, w, expected):
    result = CalibratedCoordinateMapper(calibration()).map_candidate(frame(), candidate(x=x, w=w))
    assert result.candidate.overlap_camera_ids == expected


@pytest.mark.parametrize("origin,expected_count", [(24, 0), (25, 0), (23, 1), (0, 1)])
def test_touching_disjoint_and_coincident_camera_lanes(origin, expected_count):
    fields = calibration().model_dump(mode="json")
    fields["cameras"][1]["x_origin_mm"] = origin
    assert len(camera_overlaps(PhysicalCalibration.model_validate(fields))) == expected_count


def test_three_lanes_reversed_lane_and_independent_offsets():
    fields = calibration().model_dump(mode="json")
    third = deepcopy(fields["cameras"][1])
    third.update(camera_id="lane_mirrored", x_origin_mm=44, x_direction=-1, y_offset_mm=7)
    fields["cameras"].append(third)
    policy = PhysicalCalibration.model_validate(fields)
    assert len(camera_overlaps(policy)) == 3
    mapper = CalibratedCoordinateMapper(policy)
    normal = mapper.map_point(frame(camera="lane_right"), PixelPoint(x=2, y=8)).point
    mirror = mapper.map_point(frame(camera="lane_mirrored"), PixelPoint(x=46, y=8)).point
    assert mirror.x_mm == normal.x_mm and mirror.y_mm == normal.y_mm + 5


def test_mapping_does_not_mutate_calibration_or_source():
    policy, context, box = calibration(), frame(), candidate()
    snapshots = [item.model_dump_json() for item in (policy, context, box)]
    CalibratedCoordinateMapper(policy).map_candidate(context, box)
    assert snapshots == [item.model_dump_json() for item in (policy, context, box)]


def test_simulated_line_origin_is_not_assumed_zero():
    fields = calibration().model_dump(mode="json")
    fields["transport"]["line_origin_index"] = 1000
    context = frame(position=SimulatedLinePosition(frame_index=1, line_index=1040))
    result = CalibratedCoordinateMapper(fields).map_point(context, PixelPoint(x=0, y=0))
    assert result.point.y_mm == 112
