"""Geometry-bound research FAISS memory; old equal-dimension banks are not enough."""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, MemorySearchResult
from backend.app.ai.pattern_pooling import PatternFeatureResult
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity
from backend.app.core.pattern_pooling import FeatureGeometry


class PatternMemoryGeometryManifest(BaseModel):
    schema_version: Literal[1] = 1
    artifact_type: Literal["pattern_feature_memory_experiment"] = "pattern_feature_memory_experiment"
    status: Literal["EXPERIMENT_ONLY"] = "EXPERIMENT_ONLY"
    geometry: FeatureGeometry
    geometry_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported pattern-memory geometry version")
        return value

    @model_validator(mode="after")
    def consistent_signature(self):
        if self.geometry_signature != self.geometry.signature:
            raise ValueError("Feature geometry signature mismatch")
        return self


class PatternAwareAnomalyMemory:
    def __init__(self, memory: ExperimentalAnomalyMemory, geometry: FeatureGeometry):
        if memory.identity.encoder_version != geometry.encoder_version or memory.identity.adapter_version != geometry.adapter_version:
            raise ValueError("Memory identity must pin exact encoder/feature geometry and adapter")
        if memory.bank.shape[1] != geometry.feature_dim:
            raise ValueError("Memory dimension does not match feature geometry")
        self.memory, self.geometry = memory, geometry

    @classmethod
    def build(cls, reference_features: list[PatternFeatureResult], *, identity: MemoryIdentity,
              config: MemoryExperimentConfig) -> "PatternAwareAnomalyMemory":
        if not reference_features:
            raise ValueError("Approved reference features are required")
        geometry = reference_features[0].geometry
        if any(value.geometry.signature != geometry.signature for value in reference_features):
            raise ValueError("Cannot mix reference features from different representation geometries")
        if identity.encoder_version != geometry.encoder_version or identity.adapter_version != geometry.adapter_version:
            raise ValueError("Supply identity pinned to the exact feature geometry; never infer from dimension")
        values = np.concatenate([value.features.embeddings for value in reference_features], axis=0)
        memory = ExperimentalAnomalyMemory.build(values, identity=identity, config=config)
        return cls(memory, geometry)

    def search(self, query: PatternFeatureResult, *, k: int = 1) -> MemorySearchResult:
        if query.geometry.signature != self.geometry.signature:
            raise ValueError("Query/reference feature geometry mismatch; rebuild a matching experimental bank")
        return self.memory.search(query.features.embeddings, k=k)

    def save_new(self, directory: Path) -> PatternMemoryGeometryManifest:
        self.memory.save_new(directory)  # Phase 4 exclusive output allocation/production path guard.
        manifest = PatternMemoryGeometryManifest(geometry=self.geometry, geometry_signature=self.geometry.signature)
        with (directory / "pattern_geometry.json").open("x", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2) + "\n")
        return manifest

    @classmethod
    def load(cls, directory: Path, *, expected_geometry: FeatureGeometry, expected_identity: MemoryIdentity,
             config: MemoryExperimentConfig) -> "PatternAwareAnomalyMemory":
        manifest = PatternMemoryGeometryManifest.model_validate_json((directory / "pattern_geometry.json").read_text(encoding="utf-8"))
        if manifest.geometry.signature != expected_geometry.signature:
            raise ValueError("Saved geometry is incompatible with the requested representation")
        memory = ExperimentalAnomalyMemory.load(directory, expected_identity=expected_identity, config=config)
        return cls(memory, expected_geometry)
