"""Configuration for diagnostic-only texture period estimation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PatternEstimatorConfig(BaseModel):
    mode: Literal["observe_only"] = "observe_only"
    min_period_px: int = Field(default=4, ge=2)
    max_period_px: int = Field(default=256, ge=2)
    min_cycles: int = Field(default=4, ge=3)
    min_profile_length: int = Field(default=64, ge=16)
    min_profile_std: float = Field(default=2.0, gt=0)
    min_peak_correlation: float = Field(default=0.45, gt=0, le=1)
    confidence_threshold: float = Field(default=0.65, gt=0, le=1)
    period_tolerance: float = Field(default=0.10, gt=0, le=0.25)
    band_count: int = Field(default=7, ge=1, le=32)
    band_width: int = Field(default=3, ge=1, le=31)
    max_analysis_size: int = Field(default=1024, ge=64, le=4096)

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_range(self) -> "PatternEstimatorConfig":
        if self.max_period_px < self.min_period_px:
            raise ValueError("max_period_px must be >= min_period_px")
        if self.band_width % 2 == 0:
            raise ValueError("band_width must be odd")
        return self
