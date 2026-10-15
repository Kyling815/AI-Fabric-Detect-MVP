"""Opt-in research hierarchy. No production reference registry or auto-adaptation."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity


def validate_memory_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError(f"Invalid {name}; use a nonblank identifier of 1-128 ASCII letters/digits/._-")
    return value


class LotMemoryConfig(BaseModel):
    enabled: bool = Field(default=False, strict=True)
    mode: Literal["global_only", "lot_only", "global_plus_lot"] = "global_only"
    missing_lot_policy: Literal["error", "global_only"] = "error"
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryArtifactSpec(BaseModel):
    path: str = Field(min_length=1)
    identity: MemoryIdentity
    config: MemoryExperimentConfig
    model_config = ConfigDict(extra="forbid", frozen=True)


class FabricLotMemoryManifest(BaseModel):
    schema_version: Literal[1] = 1
    artifact_type: Literal["fabric_lot_memory_experiment"] = "fabric_lot_memory_experiment"
    status: Literal["EXPERIMENT_ONLY"] = "EXPERIMENT_ONLY"
    fabric_id: str
    lot_id: str | None = None
    config: LotMemoryConfig = Field(default_factory=LotMemoryConfig)
    global_memory: MemoryArtifactSpec | None = None
    lot_memory: MemoryArtifactSpec | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported hierarchy artifact version")
        return value

    @field_validator("fabric_id")
    @classmethod
    def valid_fabric(cls, value):
        return validate_memory_identifier(value, "fabric_id")

    @field_validator("lot_id")
    @classmethod
    def valid_lot(cls, value):
        return None if value is None else validate_memory_identifier(value, "lot_id")
