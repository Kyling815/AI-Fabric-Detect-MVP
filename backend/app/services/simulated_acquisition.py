"""Finite, deterministic in-memory camera/encoder fixtures; no SDK or stream loop."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import Field

from backend.app.core.physical_coordinates import CoordinateContract, EncoderReading, FrameCoordinates, Position


@dataclass(frozen=True)
class CameraCapture:
    frame: FrameCoordinates
    rgb_bytes: bytes
    simulated: bool = True


@runtime_checkable
class CameraSource(Protocol):
    def capture(self, frame_index: int, position: Position) -> CameraCapture: ...


@runtime_checkable
class EncoderSource(Protocol):
    def read(self, frame_index: int) -> EncoderReading: ...


class SimulatedEncoderConfig(CoordinateContract):
    encoder_id: str = Field(min_length=1, max_length=128)
    start_count: int = Field(strict=True)
    counts_per_frame: int = Field(strict=True)


class SimulatedEncoderSource:
    def __init__(self, config: SimulatedEncoderConfig):
        self.config = SimulatedEncoderConfig.model_validate(config.model_dump())

    def read(self, frame_index: int) -> EncoderReading:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Explicit nonnegative integer frame index required")
        return EncoderReading(encoder_id=self.config.encoder_id, frame_index=frame_index,
            count=self.config.start_count + frame_index * self.config.counts_per_frame, simulated=True)


class SimulatedCameraConfig(CoordinateContract):
    camera_id: str = Field(min_length=1, max_length=128)
    fabric_id: str = Field(min_length=1, max_length=128)
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)
    frame_prefix: str = Field(min_length=1, max_length=100)
    roll_id: str | None = Field(default=None, min_length=1, max_length=128)
    lot_id: str | None = Field(default=None, min_length=1, max_length=128)


class SimulatedCameraSource:
    def __init__(self, config: SimulatedCameraConfig):
        self.config = SimulatedCameraConfig.model_validate(config.model_dump())
        if self.config.image_width * self.config.image_height > 4_000_000:
            raise ValueError("In-memory simulated capture exceeds four million pixels")

    def capture(self, frame_index: int, position: Position) -> CameraCapture:
        fields = self.config.model_dump(exclude={"frame_prefix"})
        frame = FrameCoordinates(**fields, frame_id=f"{self.config.frame_prefix}_{frame_index}", frame_index=frame_index,
                                 coordinate_frame="oriented_source_pixels", position=position)
        rgb = bytes([frame_index % 256]) * (frame.image_width * frame.image_height * 3)
        return CameraCapture(frame=frame, rgb_bytes=rgb)
