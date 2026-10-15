"""Shadow-only settings and bounded diagnostics; never QC or public API schemas."""

import hashlib
import struct
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.segmentation_data import SegmentationTaxonomy


class SupervisedShadowConfig(BaseModel):
    mode: Literal["shadow"] = "shadow"
    checkpoint_path: str = ""
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_model_version: str | None = Field(default=None, min_length=1, max_length=128)
    trusted_local: bool = Field(default=False, strict=True)
    allowed_fabric_ids: tuple[str, ...] = ()
    allow_synthetic: bool = Field(default=False, strict=True)
    timeout_seconds: float = Field(default=10.0, ge=0.05, le=30)
    cpu_threads: int = Field(default=1, ge=1, le=4, strict=True)
    max_checkpoint_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=512 * 1024 * 1024, strict=True)
    max_mask_runs: int = Field(default=4096, ge=0, le=8192, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def strict_timeout(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("Timeout must be numeric")
        return value

    @model_validator(mode="after")
    def explicit_artifact_policy(self):
        identifiers = (*self.allowed_fabric_ids, *((self.expected_model_version,) if self.expected_model_version else ()))
        if any(not value.strip() or value != value.strip() for value in identifiers):
            raise ValueError("Shadow identifiers must be nonblank without surrounding whitespace")
        if len(set(self.allowed_fabric_ids)) != len(self.allowed_fabric_ids):
            raise ValueError("Duplicate shadow fabric ids")
        if self.checkpoint_path.startswith(("\\\\", "//")):
            raise ValueError("Only trusted local checkpoints, not network shares, are supported")
        if self.checkpoint_path and (not self.checkpoint_sha256 or not self.expected_model_version or not self.allowed_fabric_ids):
            raise ValueError("Checkpoint requires pinned SHA-256, model version and explicit fabric allowlist")
        return self


class ShadowClassSummary(BaseModel):
    name: str
    predicted_pixels: int = Field(ge=0, strict=True)
    mean_class_probability: float = Field(ge=0, le=1)
    mean_winning_confidence: float | None = Field(default=None, ge=0, le=1)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SupervisedShadowObservation(BaseModel):
    status: Literal["observed", "unavailable", "timeout", "error", "busy", "configuration_error"]
    reason: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    mode: Literal["shadow"] = "shadow"
    requested_model_version: str | None = None
    model_version: str | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    synthetic_checkpoint: bool | None = Field(default=None, strict=True)
    elapsed_ms: float = Field(default=0, ge=0)
    original_shape: tuple[int, int] | None = None  # Oriented H,W pixels, NOT physical coordinates.
    mask_shape: tuple[int, int] | None = None  # Model-grid H,W.
    mask_coordinate_frame: Literal["model_grid"] = "model_grid"
    taxonomy: tuple[str, ...] = ()
    classes: tuple[ShadowClassSummary, ...] = ()
    mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mask_rle: tuple[tuple[int, int], ...] | None = None  # Flat row-major class-id runs, bounded; never in log.
    mask_rle_omitted: bool = Field(default=False, strict=True)
    confidence_calibrated: Literal[False] = False
    affects_qc: Literal[False] = False
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("affects_qc", "confidence_calibrated", mode="before")
    @classmethod
    def must_be_false(cls, value):
        if type(value) is not bool or value:
            raise ValueError("Shadow never changes QC; confidence is not calibrated")
        return value

    @model_validator(mode="after")
    def coherent_output(self):
        if self.status != "observed":
            if self.classes or self.taxonomy or self.mask_shape or self.original_shape or self.mask_sha256 or self.mask_rle or self.mask_rle_omitted:
                raise ValueError("Unavailable/failed shadow must not contain fabricated segmentation evidence")
            return self
        SegmentationTaxonomy(classes=self.taxonomy)
        if not self.model_version or not self.checkpoint_sha256 or self.synthetic_checkpoint is None or not self.mask_sha256:
            raise ValueError("Observed shadow requires verified version/hash/provenance and mask checksum")
        if self.model_version != self.requested_model_version:
            raise ValueError("Observed version does not match the pinned requested version")
        if not self.original_shape or not self.mask_shape or any(type(value) is not int or value <= 0 for value in (*self.original_shape, *self.mask_shape)):
            raise ValueError("Observed shadow requires explicit positive pixel geometry")
        if max(self.mask_shape) > 512 or tuple(item.name for item in self.classes) != self.taxonomy:
            raise ValueError("Invalid shadow taxonomy/geometry")
        area = self.mask_shape[0] * self.mask_shape[1]
        if sum(item.predicted_pixels for item in self.classes) != area:
            raise ValueError("Shadow class counts do not cover the mask")
        if any((item.predicted_pixels == 0) != (item.mean_winning_confidence is None) for item in self.classes):
            raise ValueError("Winning confidence must correspond to predicted class support")
        if abs(sum(item.mean_class_probability for item in self.classes) - 1) > 1e-5:
            raise ValueError("Class probabilities do not sum to one")
        if self.mask_rle is None:
            if not self.mask_rle_omitted:
                raise ValueError("Missing mask runs must be explicitly marked omitted")
        else:
            if self.mask_rle_omitted or len(self.mask_rle) > 8192:
                raise ValueError("Mask run policy mismatch")
            counts = [0] * len(self.taxonomy)
            for label, length in self.mask_rle:
                if type(label) is not int or type(length) is not int or not 0 <= label < len(counts) or length < 1:
                    raise ValueError("Invalid mask class-id run")
                counts[label] += length
            if counts != [item.predicted_pixels for item in self.classes]:
                raise ValueError("Mask runs disagree with class counts")
            digest = hashlib.sha256()
            for label, length in self.mask_rle:
                digest.update(struct.pack("<i", label) * length)
            if digest.hexdigest() != self.mask_sha256:
                raise ValueError("Mask checksum disagrees with class-id runs")
        return self

    def audit_payload(self) -> dict:
        """Keep logs bounded: no raw image, paths, arrays, full probabilities or RLE."""
        return self.model_dump(mode="json", exclude={"mask_rle"})
