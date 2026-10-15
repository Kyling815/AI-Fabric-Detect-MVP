"""Offline Phase 8 baseline settings. These are starting values, not optimal values."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.segmentation_data import SegmentationPipelineConfig


class SegmentationBaselineConfig(BaseModel):
    pipeline: SegmentationPipelineConfig = Field(default_factory=SegmentationPipelineConfig)
    pretrained_encoder: bool = Field(default=True, strict=True)
    freeze_encoder_batchnorm: bool = Field(default=True, strict=True)
    decoder_channels: tuple[int, int, int, int, int] = (128, 64, 32, 16, 16)
    device: str = "auto"
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    cpu_threads: int = Field(default=2, ge=1, le=16, strict=True)
    ce_weight: float = Field(default=1.0, ge=0)
    dice_weight: float = Field(default=1.0, ge=0)
    focal_gamma: float = Field(default=2.0, ge=0, le=10)
    min_delta: float = Field(default=0.0, ge=0)
    gradient_clip_norm: float = Field(default=5.0, gt=0)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("decoder_channels", mode="before")
    @classmethod
    def strict_channels(cls, value):
        if not isinstance(value, (tuple, list)) or any(type(item) is not int or not 1 <= item <= 512 for item in value):
            raise ValueError("Five bounded integer decoder channel widths required")
        return value

    @field_validator("device")
    @classmethod
    def supported_device(cls, value):
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu or cuda")
        return value

    @model_validator(mode="after")
    def valid_baseline(self):
        if self.ce_weight + self.dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if min(self.pipeline.training.image_size) < 32:
            raise ValueError("ResNet18 baseline requires dimensions >=32")
        return self
