"""Explicit linear calibration contracts; pixels/counts are never implicitly mm.

Coordinates are edges in the EXIF-oriented source image: x in [0,width],
y in [0,height]. A pixel centre is (column+.5,row+.5); boxes are half-open.
The transport position anchors the first row edge of one synchronized frame.
"""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CoordinateContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def clean_strings_and_axes(cls, values):
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, str) and (not value.strip() or value != value.strip()):
                    raise ValueError("Identifiers/units must be nonblank and clean")
                if name.endswith("direction") and type(value) is not int:
                    raise ValueError("Axis direction must be integer -1 or +1")
        return values


class CameraCalibration(CoordinateContract):
    camera_id: str = Field(min_length=1, max_length=128)
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)
    coordinate_frame: Literal["oriented_source_pixels"]
    pixel_convention: Literal["pixel_edges"]
    x_mm_per_pixel: float = Field(gt=0, strict=True)
    y_mm_per_pixel: float = Field(gt=0, strict=True)
    x_origin_mm: float = Field(strict=True)
    y_offset_mm: float = Field(strict=True)
    x_direction: Literal[-1, 1]
    y_direction: Literal[-1, 1]

    @model_validator(mode="after")
    def finite_extent(self):
        extents = (self.image_width * self.x_mm_per_pixel, self.image_height * self.y_mm_per_pixel)
        x_end = self.x_origin_mm + self.x_direction * extents[0]
        y_end = self.y_offset_mm + self.y_direction * extents[1]
        if (not all(math.isfinite(value) for value in (*extents, x_end, y_end))
                or x_end == self.x_origin_mm or y_end == self.y_offset_mm):
            raise ValueError("Calibrated lane extents exceed usable finite numeric precision")
        return self


class TransportCalibration(CoordinateContract):
    encoder_id: str = Field(min_length=1, max_length=128)
    mm_per_encoder_count: float = Field(gt=0, strict=True)
    encoder_origin_count: int = Field(strict=True)
    mm_per_line: float = Field(gt=0, strict=True)
    line_origin_index: int = Field(ge=0, strict=True)
    y_origin_mm: float = Field(strict=True)
    travel_direction: Literal[-1, 1]


class PhysicalCalibration(CoordinateContract):
    calibration_version: str = Field(min_length=1, max_length=128)
    world_frame_id: str = Field(min_length=1, max_length=128)
    transport: TransportCalibration
    cameras: tuple[CameraCalibration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_lanes(self):
        if len({camera.camera_id for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Duplicate calibrated camera lane")
        return self


class EncoderReading(CoordinateContract):
    kind: Literal["encoder_count"] = "encoder_count"
    encoder_id: str = Field(min_length=1, max_length=128)
    frame_index: int = Field(ge=0, strict=True)
    count: int = Field(strict=True)
    simulated: bool = Field(strict=True)


class SimulatedLinePosition(CoordinateContract):
    kind: Literal["simulated_line"] = "simulated_line"
    frame_index: int = Field(ge=0, strict=True)
    line_index: int = Field(ge=0, strict=True)
    simulated: Literal[True] = True

    @field_validator("simulated", mode="before")
    @classmethod
    def simulation_only(cls, value):
        if value is not True:
            raise ValueError("Line-position source is simulated, not a hardware encoder")
        return value


Position = Annotated[EncoderReading | SimulatedLinePosition, Field(discriminator="kind")]


class FrameCoordinates(CoordinateContract):
    fabric_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    frame_id: str = Field(min_length=1, max_length=128)
    frame_index: int = Field(ge=0, strict=True)
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)
    coordinate_frame: Literal["oriented_source_pixels"]
    position: Position
    roll_id: str | None = Field(default=None, min_length=1, max_length=128)
    lot_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def synchronized_position(self):
        if self.position.frame_index != self.frame_index:
            raise ValueError("Transport reading must refer to the same frame index")
        return self


class PixelPoint(CoordinateContract):
    x: float = Field(ge=0, strict=True)
    y: float = Field(ge=0, strict=True)


class PixelCandidate(PixelPoint):
    candidate_id: str = Field(min_length=1, max_length=128)
    w: float = Field(gt=0, strict=True)
    h: float = Field(gt=0, strict=True)
    label: str = Field(min_length=1, max_length=128)
    score: float = Field(ge=0, strict=True)  # Anomaly distance, NOT a probability.
    coordinate_frame: Literal["oriented_source_pixels"] = "oriented_source_pixels"


class PhysicalPoint(CoordinateContract):
    x_mm: float = Field(strict=True)
    y_mm: float = Field(strict=True)

    @property
    def y_m(self):
        return self.y_mm / 1000.0


class PhysicalCandidate(CoordinateContract):
    source: PixelCandidate
    minimum: PhysicalPoint
    maximum: PhysicalPoint
    width_mm: float = Field(gt=0, strict=True)
    length_mm: float = Field(gt=0, strict=True)
    overlap_camera_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def bounded_rectangle(self):
        if (self.maximum.x_mm <= self.minimum.x_mm or self.maximum.y_mm <= self.minimum.y_mm
                or abs((self.maximum.x_mm - self.minimum.x_mm) - self.width_mm) > 1e-9 * max(1, self.width_mm)
                or abs((self.maximum.y_mm - self.minimum.y_mm) - self.length_mm) > 1e-9 * max(1, self.length_mm)):
            raise ValueError("Physical rectangle must agree with its calibrated extents")
        return self


class CameraOverlap(CoordinateContract):
    camera_ids: tuple[str, str]
    x_min_mm: float = Field(strict=True)
    x_max_mm: float = Field(strict=True)

    @model_validator(mode="after")
    def positive_overlap(self):
        if self.camera_ids[0] == self.camera_ids[1] or self.x_min_mm >= self.x_max_mm:
            raise ValueError("Overlap needs two distinct cameras and positive width")
        return self


class CoordinateMappingResult(CoordinateContract):
    status: Literal["mapped", "unavailable"]
    reason: str
    calibration: PhysicalCalibration | None = None
    frame: FrameCoordinates | None = None
    point: PhysicalPoint | None = None
    candidate: PhysicalCandidate | None = None
    mapping_model: Literal["linear_axis_aligned_v1"] = "linear_axis_aligned_v1"

    @model_validator(mode="after")
    def no_uncalibrated_physical_claim(self):
        if self.status == "unavailable":
            if self.point is not None or self.candidate is not None:
                raise ValueError("Unavailable mapping cannot contain physical coordinates")
        elif self.calibration is None or self.frame is None or (self.point is None) == (self.candidate is None):
            raise ValueError("Mapped output requires calibration, frame and exactly one payload")
        return self
