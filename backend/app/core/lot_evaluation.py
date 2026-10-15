"""Declared independent research queries; no locked-test relabeling or calibration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.lot_memory import validate_memory_identifier


class ResearchAggregationConfig(BaseModel):
    image_score_method: Literal["max", "top_k_mean", "percentile_995", "max_median_contrast",
                                "percentile_995_median_contrast", "top_k_median_contrast"] = "max"
    image_score_top_fraction: float = Field(default=0.01, gt=0, le=1)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class LotEvaluationSample(BaseModel):
    image_id: str = Field(min_length=1)
    fabric_id: str
    lot_id: str
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    embeddings_path: str = Field(min_length=1)
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: Literal["normal", "defect"]
    origin: Literal["real", "synthetic"]
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("fabric_id", "lot_id")
    @classmethod
    def valid_scope(cls, value):
        return validate_memory_identifier(value, "fabric/lot_id")


class LotEvaluationManifest(BaseModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    fabric_id: str
    lot_id: str
    encoder_version: str = Field(min_length=1)
    adapter_version: str | None
    split: Literal["research_evaluation"] = "research_evaluation"
    independence_basis: str = Field(min_length=1)
    aggregation: ResearchAggregationConfig = Field(default_factory=ResearchAggregationConfig)
    samples: tuple[LotEvaluationSample, ...] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported evaluation manifest version")
        return value

    @field_validator("fabric_id", "lot_id")
    @classmethod
    def valid_scope(cls, value):
        return validate_memory_identifier(value, "fabric/lot_id")

    @model_validator(mode="after")
    def coherent_dataset(self):
        if any(sample.fabric_id != self.fabric_id or sample.lot_id != self.lot_id for sample in self.samples):
            raise ValueError("Queries must belong to the requested fabric/current lot")
        ids = [sample.image_id for sample in self.samples]
        hashes = [sample.image_sha256 for sample in self.samples]
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValueError("Duplicate evaluation identities/image hashes")
        return self
