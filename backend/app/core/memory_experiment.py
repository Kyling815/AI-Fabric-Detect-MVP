"""Strict provenance for opt-in, uncalibrated research memory (not a registry)."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryExperimentConfig(BaseModel):
    enabled: bool = Field(default=False, strict=True)
    distance_metric: Literal["current", "cosine", "l2"] = "current"
    coreset_ratio: float = Field(default=1.0, gt=0, le=1)
    selector: Literal["seeded_random"] = "seeded_random"
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("coreset_ratio", mode="before")
    @classmethod
    def ratio_is_numeric(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("coreset_ratio must be numeric")
        return value


class SourceGoodImage(BaseModel):
    image_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fabric_id: str = Field(min_length=1)
    lot_id: str | None
    label: Literal["normal"] = "normal"
    split: Literal["reference", "research_reference"] = "research_reference"
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryIdentity(BaseModel):
    reference_version: str = Field(min_length=1)
    encoder_version: str = Field(min_length=1)
    adapter_version: str | None
    fabric_id: str = Field(min_length=1)
    lot_id: str | None
    threshold_profile_version: str | None
    source_good_images: tuple[SourceGoodImage, ...] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("reference_version", "encoder_version", "adapter_version", "fabric_id", "lot_id", "threshold_profile_version")
    @classmethod
    def nonblank_identifier(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Identifiers must be nonblank or explicitly null")
        return value

    @model_validator(mode="after")
    def consistent_sources(self):
        if any(source.fabric_id != self.fabric_id for source in self.source_good_images):
            raise ValueError("Source Good images must belong to the declared fabric")
        if self.lot_id is not None and any(source.lot_id != self.lot_id for source in self.source_good_images):
            raise ValueError("Source lot identifiers disagree with provenance")
        ids = [source.image_id for source in self.source_good_images]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source image identifiers")
        return self


class MemoryExperimentMetadata(MemoryIdentity):
    schema_version: Literal[1] = 1
    artifact_type: Literal["memory_experiment"] = "memory_experiment"
    status: Literal["EXPERIMENT_ONLY"] = "EXPERIMENT_ONLY"
    threshold_calibration: Literal["NOT_CALIBRATED"] = "NOT_CALIBRATED"
    distance_metric: Literal["cosine", "l2"]
    requested_metric: Literal["current", "cosine", "l2"]
    score_definition: Literal["one_minus_cosine", "squared_l2"]
    vector_policy: Literal["preserve_unit", "normalize", "raw"]
    coreset_ratio: float = Field(gt=0, le=1)
    actual_coreset_ratio: float = Field(gt=0, le=1)
    selector: Literal["seeded_random"]
    random_seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    full_feature_count: int = Field(ge=1, strict=True)
    memory_bank_count: int = Field(ge=1, strict=True)
    feature_dim: int = Field(ge=1, strict=True)
    source_embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_bank_path: Literal["memory_bank.npy"] = "memory_bank.npy"
    faiss_index_path: Literal["memory_bank.faiss"] = "memory_bank.faiss"
    selected_indices_path: Literal["selected_indices.npy"] = "selected_indices.npy"

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported experimental reference schema version")
        return value

    @model_validator(mode="after")
    def consistent_experiment(self):
        metric = "cosine" if self.requested_metric == "current" else self.requested_metric
        definition = "one_minus_cosine" if metric == "cosine" else "squared_l2"
        policy = {"current": "preserve_unit", "cosine": "normalize", "l2": "raw"}[self.requested_metric]
        if (self.distance_metric, self.score_definition, self.vector_policy) != (metric, definition, policy):
            raise ValueError("Metric, vector policy and score definition disagree")
        count = max(1, math.ceil(self.full_feature_count * self.coreset_ratio))
        if self.memory_bank_count != count or abs(self.actual_coreset_ratio - count / self.full_feature_count) > 1e-12:
            raise ValueError("Coreset counts/ratios disagree")
        return self
