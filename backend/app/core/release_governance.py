"""Phase18 fail-closed release contracts. No SLA/threshold defaults or activation."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=512)]
Count = Annotated[int, Field(ge=0)]
METRIC_DIRECTIONS = {
    "defect_recall": "min", "false_negatives": "max", "underkill": "max",
    "false_positive_rate": "max", "overkill": "max", "precision": "min", "f1": "min",
    "segmentation_iou": "min", "segmentation_dice": "min", "unknown_defect_recall": "min",
    "false_alarms_per_100m": "max", "missed_defects_per_100m": "max",
    "p50_latency_ms": "max", "p95_latency_ms": "max", "throughput_images_per_second": "min",
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def clean_values(cls, value):
        if isinstance(value, str) and (not value.strip() or value != value.strip()):
            raise ValueError("Nonblank strings without surrounding whitespace required")
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Timezone-aware timestamp required")
        return value


class ArtifactPin(Contract):
    path: Name
    sha256: Digest

    @field_validator("path")
    @classmethod
    def explicit_absolute_path(cls, value):
        if not Path(value).is_absolute():
            raise ValueError("Absolute artifact paths required; do not depend on caller working directory")
        return value


class Candidate(Contract):
    schema_version: Literal[1] = 1
    candidate_id: Name
    fabric_id: Name
    model_version: Name
    reference_id: Name
    reference_version: Annotated[int, Field(ge=1)]
    backbone_version: Name
    runtime_version: Name
    frozen_at: datetime
    origin: Literal["real", "synthetic"]
    trained: bool
    research_only: bool
    precision: Literal["FP32", "FP16", "FP16_MIXED"]
    defect_classes: tuple[Name, ...] = Field(min_length=1)
    segmentation_applicable: bool
    model: ArtifactPin
    training_report: ArtifactPin | None
    artifacts: dict[str, ArtifactPin]  # metadata, bank, index, thresholds, runtime recipe

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.defect_classes)) != len(self.defect_classes) or "background" in self.defect_classes:
            raise ValueError("Unique foreground defect classes required")
        expected = {"reference_metadata", "memory_bank", "faiss_index", "thresholds", "runtime_recipe", "deployment_engine"}
        if set(self.artifacts) != expected:
            raise ValueError("Complete model/reference/threshold/runtime provenance required")
        return self


class Criterion(Contract):
    direction: Literal["min", "max"]
    threshold: Annotated[float, Field(ge=0)] | None = None
    applicable: bool = True
    not_applicable_reason: Name | None = None


class AcceptancePolicy(Contract):
    schema_version: Literal[1] = 1
    policy_version: Name
    approved_by: Name | None = None
    approved_at: datetime | None = None
    valid_until: datetime | None = None
    criteria: dict[str, Criterion]
    per_class_recall: dict[str, Annotated[float, Field(ge=0, le=1)] | None] = {}

    @model_validator(mode="after")
    def policy_contract(self):
        if set(self.criteria) != set(METRIC_DIRECTIONS):
            raise ValueError("All acceptance metric entries must be explicit")
        for name, rule in self.criteria.items():
            if rule.direction != METRIC_DIRECTIONS[name]:
                raise ValueError("Wrong acceptance direction: " + name)
            if not rule.applicable and (name not in {"segmentation_iou", "segmentation_dice"}
                                        or rule.not_applicable_reason is None):
                raise ValueError("Only segmentation metrics may be explicitly inapplicable")
            if (rule.threshold is not None and name in {"defect_recall", "underkill", "false_positive_rate", "overkill",
                    "precision", "f1", "segmentation_iou", "segmentation_dice", "unknown_defect_recall"}
                    and rule.threshold > 1):
                raise ValueError("Rate thresholds must be <=1")
        if self.approved_at and self.valid_until and self.valid_until <= self.approved_at:
            raise ValueError("Policy validity must follow approval")
        return self


class SampleProvenance(Contract):
    sample_id: Name
    fabric_id: Name
    source_id: Name
    source_sha256: Digest
    ancestor_source_ids: tuple[Name, ...] = ()
    ancestor_source_sha256s: tuple[Digest, ...] = ()
    lot_id: Name
    roll_id: Name
    camera_id: Name
    origin: Literal["real", "synthetic"]
    annotation_source: Literal["human_verified", "synthetic", "unverified"] = "unverified"
    annotation_verified_by: Name | None = None
    partition: Literal["train", "validation", "calibration", "reference", "development", "historical_locked", "postmortem", "fresh_locked"]
    image: ArtifactPin
    mask: ArtifactPin | None = None
    start_m: Annotated[float, Field(ge=0)] | None = None
    end_m: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def interval(self):
        if (self.start_m is None) != (self.end_m is None) or (self.start_m is not None and self.end_m <= self.start_m):
            raise ValueError("Measured start/end must form a positive interval")
        return self


class DatasetInventory(Contract):
    dataset_version: Name
    purpose: Literal["development_and_historical", "fresh_locked"]
    provenance_verified_by: Name
    complete_exposure_inventory: bool
    samples: tuple[SampleProvenance, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_samples(self):
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate sample ids")
        return self


class LockedTest(Contract):
    lock_id: Name
    locked_at: datetime
    custodian: Name
    previously_opened: bool
    historical_or_postmortem: bool
    bound_candidate_sha256: Digest
    bound_policy_sha256: Digest
    inventory: DatasetInventory
    length_calibration: ArtifactPin | None = None


class ValidationEvidence(Contract):
    candidate_sha256: Digest
    used_inventory_sha256: Digest
    evaluated_at: datetime
    evaluator: Name
    synthetic: bool
    passed: bool
    functional_checks_passed: bool
    numerical_parity_passed: bool
    raw_result: ArtifactPin


class Observation(Contract):
    sample_id: Name
    predicted_defective: bool
    defect_class: Name | None
    unknown_defect: bool
    truth_events: Count
    matched_events: Count
    false_alarm_events: Count
    # Foreground pixel confusion for this sample, only when segmentation applies.
    pixel_tp: Count | None = None
    pixel_fp: Count | None = None
    pixel_fn: Count | None = None

    @model_validator(mode="after")
    def coherent_events(self):
        if self.matched_events > self.truth_events:
            raise ValueError("One-to-one matches cannot exceed truth events")
        if self.truth_events == 0 and (self.unknown_defect or self.defect_class is not None):
            raise ValueError("Normal observation cannot carry a defect class")
        if self.truth_events > 0 and self.defect_class is None:
            raise ValueError("Defect observations require an explicit class")
        if not self.predicted_defective and (self.matched_events or self.false_alarm_events):
            raise ValueError("GOOD prediction cannot contain matched/false-alarm events")
        if self.predicted_defective and self.matched_events + self.false_alarm_events == 0:
            raise ValueError("Positive prediction needs explicit matched or false-alarm events")
        pixels = (self.pixel_tp, self.pixel_fp, self.pixel_fn)
        if any(v is None for v in pixels) and not all(v is None for v in pixels):
            raise ValueError("Pixel counts must be complete or entirely unavailable")
        return self


class EvaluationEvidence(Contract):
    candidate_sha256: Digest
    policy_sha256: Digest
    locked_test_sha256: Digest
    evaluated_at: datetime
    evaluator: Name
    synthetic: bool
    scope: Literal["full_hybrid_pipeline", "segmentation_architecture_only"]
    raw_result: ArtifactPin
    event_matching_protocol: Name
    observations: tuple[Observation, ...] = Field(min_length=1)
    latency_ms: tuple[Annotated[float, Field(gt=0)], ...] = Field(min_length=1)
    timing_sample_ids: tuple[Name, ...] = Field(min_length=1)
    warmup_excluded: bool
    cuda_synchronized_if_used: bool
    throughput_images: Annotated[int, Field(gt=0)]
    throughput_wall_seconds: Annotated[float, Field(gt=0)]
    throughput_protocol: Name

    @model_validator(mode="after")
    def no_missing_or_duplicate_rows(self):
        ids = [o.sample_id for o in self.observations]
        if len(set(ids)) != len(ids) or len(self.latency_ms) != len(self.timing_sample_ids):
            raise ValueError("Duplicate observations or timing row mismatch")
        if set(self.timing_sample_ids) != set(ids) or self.throughput_images != len(self.timing_sample_ids):
            raise ValueError("Timing must cover every evaluated sample, with explicit throughput count")
        return self


class ReleaseBundle(Contract):
    candidate: ArtifactPin
    policy: ArtifactPin
    used_inventory: ArtifactPin
    locked_test: ArtifactPin
    validation: ArtifactPin
    evaluation: ArtifactPin


class ManualApproval(Contract):
    candidate_sha256: Digest
    report_sha256: Digest
    expected_registry_sha256: Digest
    reviewer: Name
    reviewed_at: datetime
    reason: Name
    confirm: Name
