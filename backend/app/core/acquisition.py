"""Internal acquisition metadata and opt-in capture-quality configuration."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.ai.preprocessing.config import PreprocessConfig


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Optional acquisition context; encoder_position has no assumed physical units."""

    lot_id: str | None = None
    roll_id: str | None = None
    camera_id: str | None = None
    frame_id: str | None = None
    timestamp: datetime | None = None
    encoder_position: float | None = None
    # x, y, width, height in EXIF-oriented source pixels; validation only, no crop.
    roi: tuple[int, int, int, int] | None = None


MetadataField = Literal["lot_id", "roll_id", "camera_id", "frame_id", "timestamp", "encoder_position"]


class QualityGateConfig(BaseModel):
    """Starting diagnostic thresholds, not calibrated production acceptance rules."""

    min_width: int = Field(default=64, ge=1)
    min_height: int = Field(default=64, ge=1)
    blur_threshold: float = Field(default=70.0, ge=0)
    max_dark_ratio: float = Field(default=0.20, ge=0, le=1)
    max_bright_ratio: float = Field(default=0.20, ge=0, le=1)
    dark_pixel_threshold: int = Field(default=20, ge=0, le=255)
    bright_pixel_threshold: int = Field(default=235, ge=0, le=255)
    check_metadata: bool = Field(default=False, strict=True)
    required_metadata: tuple[MetadataField, ...] = ()
    expected_camera_id: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def coherent_policy(self) -> "QualityGateConfig":
        if self.dark_pixel_threshold >= self.bright_pixel_threshold:
            raise ValueError("dark_pixel_threshold must be below bright_pixel_threshold")
        if not self.check_metadata and (self.required_metadata or self.expected_camera_id is not None):
            raise ValueError("metadata requirements need check_metadata=true")
        if self.expected_camera_id is not None and not self.expected_camera_id.strip():
            raise ValueError("expected_camera_id must not be empty")
        return self

    def image_quality_config(self) -> PreprocessConfig:
        """Reuse v2 metrics/validation only; never invoke its preprocessing transforms."""
        return PreprocessConfig(**self.model_dump(exclude={"check_metadata", "required_metadata", "expected_camera_id"}))
