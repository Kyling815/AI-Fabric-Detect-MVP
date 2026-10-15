"""Versioned experimental feature geometry; dimension alone is not compatibility."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.pattern import PatternEstimatorConfig


class PatternPoolingConfig(BaseModel):
    enabled: bool = Field(default=False, strict=True)
    mode: Literal["period_aware", "fixed_multiscale"] = "period_aware"
    confidence_threshold: float = Field(default=0.65, gt=0, le=1)
    motif_scales: tuple[Literal[1, 2, 4], ...] = (1, 2, 4)
    fallback_kernels: tuple[int, ...] = (3, 5, 7)
    max_kernel: int = Field(default=31, ge=1, le=63, strict=True)
    fusion: Literal["mean"] = "mean"
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("motif_scales", "fallback_kernels", mode="before")
    @classmethod
    def strict_scale_values(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
            raise ValueError("Scales/kernels must be integer lists, not booleans or floats")
        return value

    @model_validator(mode="after")
    def coherent_scales(self):
        if not self.motif_scales or tuple(sorted(set(self.motif_scales))) != self.motif_scales:
            raise ValueError("motif_scales must be nonempty, sorted and unique")
        if self.max_kernel % 2 != 1:
            raise ValueError("max_kernel must be odd")
        if (not self.fallback_kernels or len(self.fallback_kernels) > 4
                or tuple(sorted(set(self.fallback_kernels))) != self.fallback_kernels
                or any(type(value) is not int or value < 1 or value % 2 != 1 or value > self.max_kernel
                       for value in self.fallback_kernels)):
            raise ValueError("fallback kernels must be 1-4 sorted unique odd positive sizes <= max_kernel")
        return self


class FeatureGeometry(BaseModel):
    schema_version: Literal[1] = 1
    algorithm: Literal["dense_baseline_v1", "pattern_pooling_mean_v1"]
    backbone_version: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    model_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_weights_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_version: str | None
    use_pretrained_weights: bool
    feature_dim: int = Field(ge=1, strict=True)
    feature_stride: Literal[8] = 8
    layer3_stride: Literal[16] = 16
    image_max_size: int = Field(ge=1, strict=True)
    preprocessing: Literal["exif_rgb_bilinear_dense_v1"] = "exif_rgb_bilinear_dense_v1"
    pooling: PatternPoolingConfig | None = None
    period_estimator: PatternEstimatorConfig | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported feature geometry version")
        return value

    @field_validator("backbone_version", "architecture", "adapter_version")
    @classmethod
    def nonblank_identifier(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Geometry identifiers must be nonblank or explicitly null")
        return value

    @model_validator(mode="after")
    def coherent_geometry(self):
        if self.algorithm == "dense_baseline_v1":
            if self.pooling is not None or self.period_estimator is not None:
                raise ValueError("Baseline geometry must not include experimental pooling")
        elif self.pooling is None or not self.pooling.enabled:
            raise ValueError("Experimental geometry requires explicitly enabled pooling")
        elif (self.pooling.mode == "period_aware") != (self.period_estimator is not None):
            raise ValueError("Period-estimator provenance must match pooling mode")
        if (self.adapter_weights_sha256 is None) != (self.adapter_version is None):
            raise ValueError("Adapter version and weight fingerprint must agree")
        return self

    @property
    def signature(self) -> str:
        value = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def encoder_version(self) -> str:
        return f"{self.backbone_version}::geometry:{self.signature}"
