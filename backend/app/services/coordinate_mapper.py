"""Pure explicit linear pixel/count mapping. Never changes AI/QC or merges lanes."""

from itertools import combinations
import math
from typing import Protocol, runtime_checkable

from backend.app.core.physical_coordinates import (CameraOverlap, CoordinateMappingResult, EncoderReading,
    FrameCoordinates, PhysicalCalibration, PhysicalCandidate, PhysicalPoint, PixelCandidate, PixelPoint)


@runtime_checkable
class CoordinateMapper(Protocol):
    def map_point(self, frame: FrameCoordinates, point: PixelPoint) -> CoordinateMappingResult: ...
    def map_candidate(self, frame: FrameCoordinates, candidate: PixelCandidate) -> CoordinateMappingResult: ...


def _x_extent(camera):
    first = camera.x_origin_mm
    last = first + camera.x_direction * camera.image_width * camera.x_mm_per_pixel
    return min(first, last), max(first, last)


def camera_overlaps(calibration: PhysicalCalibration) -> tuple[CameraOverlap, ...]:
    calibration = PhysicalCalibration.model_validate(calibration.model_dump())
    result = []
    for left, right in combinations(calibration.cameras, 2):
        left_min, left_max = _x_extent(left)
        right_min, right_max = _x_extent(right)
        low, high = max(left_min, right_min), min(left_max, right_max)
        if low < high:
            result.append(CameraOverlap(camera_ids=(left.camera_id, right.camera_id), x_min_mm=low, x_max_mm=high))
    return tuple(result)


class CalibratedCoordinateMapper:
    def __init__(self, calibration: PhysicalCalibration | dict | None):
        self.calibration, self.error = None, "CALIBRATION_UNAVAILABLE"
        if calibration is not None:
            try:
                fields = calibration.model_dump() if isinstance(calibration, PhysicalCalibration) else calibration
                self.calibration = PhysicalCalibration.model_validate(fields)
                self.error = None
            except Exception:
                self.error = "INVALID_CALIBRATION"

    def _prepare(self, frame):
        if self.error:
            raise ValueError(self.error)
        calibration = PhysicalCalibration.model_validate(self.calibration.model_dump())
        frame = FrameCoordinates.model_validate(frame.model_dump())
        camera = next((item for item in calibration.cameras if item.camera_id == frame.camera_id), None)
        if camera is None:
            raise ValueError("CAMERA_NOT_CALIBRATED")
        if (frame.image_width, frame.image_height) != (camera.image_width, camera.image_height):
            raise ValueError("CALIBRATED_GEOMETRY_MISMATCH")
        position, transport = frame.position, calibration.transport
        if isinstance(position, EncoderReading):
            if position.encoder_id != transport.encoder_id:
                raise ValueError("ENCODER_ID_MISMATCH")
            displacement = (position.count - transport.encoder_origin_count) * transport.mm_per_encoder_count
        else:
            displacement = (position.line_index - transport.line_origin_index) * transport.mm_per_line
        anchor = transport.y_origin_mm + transport.travel_direction * displacement + camera.y_offset_mm
        y_end = anchor + camera.y_direction * camera.image_height * camera.y_mm_per_pixel
        if not math.isfinite(anchor) or not math.isfinite(y_end) or y_end == anchor:
            raise ValueError("INVALID_MAPPING_INPUT_OR_RANGE")
        return calibration, camera, frame, anchor

    @staticmethod
    def _point(camera, anchor, x, y):
        if not (0 <= x <= camera.image_width and 0 <= y <= camera.image_height):
            raise ValueError("PIXEL_OUTSIDE_CALIBRATED_FRAME")
        return PhysicalPoint(x_mm=camera.x_origin_mm + camera.x_direction * x * camera.x_mm_per_pixel,
                             y_mm=anchor + camera.y_direction * y * camera.y_mm_per_pixel)

    @staticmethod
    def _failure(exc):
        codes = {"CALIBRATION_UNAVAILABLE", "INVALID_CALIBRATION", "CAMERA_NOT_CALIBRATED", "CALIBRATED_GEOMETRY_MISMATCH",
                 "ENCODER_ID_MISMATCH", "PIXEL_OUTSIDE_CALIBRATED_FRAME"}
        reason = str(exc) if str(exc) in codes else "INVALID_MAPPING_INPUT_OR_RANGE"
        return CoordinateMappingResult(status="unavailable", reason=reason)

    def map_point(self, frame: FrameCoordinates, point: PixelPoint) -> CoordinateMappingResult:
        try:
            calibration, camera, frame, anchor = self._prepare(frame)
            point = PixelPoint.model_validate(point.model_dump())
            mapped = self._point(camera, anchor, point.x, point.y)
            return CoordinateMappingResult(status="mapped", reason="EXPLICIT_LINEAR_CALIBRATION", calibration=calibration,
                                           frame=frame, point=mapped)
        except Exception as exc:
            return self._failure(exc)

    def map_candidate(self, frame: FrameCoordinates, candidate: PixelCandidate) -> CoordinateMappingResult:
        try:
            calibration, camera, frame, anchor = self._prepare(frame)
            candidate = PixelCandidate.model_validate(candidate.model_dump())
            first = self._point(camera, anchor, candidate.x, candidate.y)
            last = self._point(camera, anchor, candidate.x + candidate.w, candidate.y + candidate.h)
            low = PhysicalPoint(x_mm=min(first.x_mm, last.x_mm), y_mm=min(first.y_mm, last.y_mm))
            high = PhysicalPoint(x_mm=max(first.x_mm, last.x_mm), y_mm=max(first.y_mm, last.y_mm))
            overlaps = tuple(pair.camera_ids[1] if pair.camera_ids[0] == camera.camera_id else pair.camera_ids[0]
                for pair in camera_overlaps(calibration) if camera.camera_id in pair.camera_ids
                and max(low.x_mm, pair.x_min_mm) < min(high.x_mm, pair.x_max_mm))
            mapped = PhysicalCandidate(source=candidate, minimum=low, maximum=high,
                width_mm=candidate.w * camera.x_mm_per_pixel, length_mm=candidate.h * camera.y_mm_per_pixel,
                overlap_camera_ids=overlaps)
            return CoordinateMappingResult(status="mapped", reason="EXPLICIT_LINEAR_CALIBRATION", calibration=calibration,
                                           frame=frame, candidate=mapped)
        except Exception as exc:
            return self._failure(exc)


def map_anomaly_boxes(mapper: CoordinateMapper, frame: FrameCoordinates, boxes) -> tuple[CoordinateMappingResult, ...]:
    """Explicit dev bridge for existing original-pixel FusionBox; no model-grid inference."""
    from backend.app.core.fusion import FusionBox
    results = []
    for index, box in enumerate(boxes):
        try:
            box = FusionBox.model_validate(box.model_dump())
            candidate = PixelCandidate(candidate_id=f"anomaly_{index}", x=box.x, y=box.y, w=box.w, h=box.h,
                                       label=box.label, score=box.score)
            results.append(mapper.map_candidate(frame, candidate))
        except Exception:
            results.append(CoordinateMappingResult(status="unavailable", reason="INVALID_SOURCE_CANDIDATE"))
    return tuple(results)
