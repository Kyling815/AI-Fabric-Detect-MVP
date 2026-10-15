"""Business acceptance policy/rule inputs, independent of CNN and anomaly thresholds."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.fusion import DefectEvidence
from backend.app.core.segmentation_data import DEFECT_CLASSES


@dataclass(frozen=True)
class LegacyDenseQCInput:
    image_score: float
    has_localization: bool
    review_threshold: float
    bad_threshold: float
    localized_score: float | None = None
    localized_review_threshold: float | None = None


@dataclass(frozen=True)
class LegacyPatchQCInput:
    defect_area_ratio: float
    defect_count: int
    max_anomaly_score: float | None
    good_area_threshold: float
    bad_area_threshold: float


class QCContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class KnownSeverityRule(QCContract):
    class_name: str
    unit: Literal["model_grid_area_fraction"] = "model_grid_area_fraction"
    allow_max_fraction: float = Field(ge=0, le=1)
    reject_min_fraction: float = Field(gt=0, le=1)

    @field_validator("class_name")
    @classmethod
    def foreground_class(cls, value):
        if value not in DEFECT_CLASSES[1:]:
            raise ValueError("QC rule must name a supported foreground class")
        return value

    @field_validator("allow_max_fraction", "reject_min_fraction", mode="before")
    @classmethod
    def numeric_thresholds(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("Severity fractions must be numeric")
        return value

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.allow_max_fraction >= self.reject_min_fraction:
            raise ValueError("Allow threshold must be below reject threshold; gap requires review")
        return self


class QCPolicy(QCContract):
    policy_version: str = Field(default="qc_unapproved_starting_policy_v1", min_length=1, max_length=128)
    allowed_fabric_ids: tuple[str, ...] = ()
    known_rules: tuple[KnownSeverityRule, ...] = ()
    unknown_high_action: Literal["BAD", "NEEDS_REVIEW"] = "NEEDS_REVIEW"
    allow_synthetic_evidence: bool = Field(default=False, strict=True)
    allow_uncalibrated_supervised: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def clean_policy(self):
        if any(not value.strip() or value != value.strip() for value in (self.policy_version, *self.allowed_fabric_ids)):
            raise ValueError("Policy/fabric identifiers must be nonblank without surrounding whitespace")
        if len(set(self.allowed_fabric_ids)) != len(self.allowed_fabric_ids):
            raise ValueError("Duplicate QC fabric scope")
        classes = [rule.class_name for rule in self.known_rules]
        if len(classes) != len(set(classes)):
            raise ValueError("Duplicate QC class rules")
        return self


class ClassSeverity(QCContract):
    class_name: str
    class_id: int = Field(ge=1, strict=True)
    area_px: int = Field(ge=1, strict=True)
    frame_pixels: int = Field(ge=1, strict=True)
    coordinate_frame: Literal["model_grid"] = "model_grid"

    @model_validator(mode="after")
    def meaningful_area(self):
        if self.area_px > self.frame_pixels:
            raise ValueError("Class area exceeds model grid")
        return self

    @property
    def fraction(self):
        return self.area_px / self.frame_pixels


class QCRuleInput(QCContract):
    fabric_id: str = Field(min_length=1, max_length=128)
    evidence: DefectEvidence | None = None
    severities: tuple[ClassSeverity, ...] = ()

    @model_validator(mode="after")
    def traceable_severity(self):
        if not self.fabric_id.strip() or self.fabric_id != self.fabric_id.strip():
            raise ValueError("Explicit clean QC fabric id required")
        keys = [(item.class_id, item.class_name) for item in self.severities]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate class severity")
        if self.severities:
            if self.evidence is None or self.evidence.supervised_mask is None:
                raise ValueError("Severity needs source mask evidence")
            candidates = {(item.class_id, item.class_name): item.predicted_pixels for item in self.evidence.known_defects}
            shape = self.evidence.supervised_mask.shape
            known = self.evidence.known_defects
            if (len({item.class_id for item in known}) != len(known) or len({item.class_name for item in known}) != len(known)
                    or sum(item.predicted_pixels for item in known) > shape[0] * shape[1]):
                raise ValueError("Known class counts/ids must describe a valid disjoint class mask")
            for item in self.severities:
                if candidates.get((item.class_id, item.class_name)) != item.area_px or item.frame_pixels != shape[0] * shape[1]:
                    raise ValueError("Severity must match exact class pixel count and model-grid geometry")
        return self


class QCClassAssessment(QCContract):
    class_name: str
    area_fraction: float = Field(ge=0, le=1)
    outcome: Literal["allowed", "review", "reject", "unmapped"]
    allow_max_fraction: float | None = None
    reject_min_fraction: float | None = None


class QCDecision(QCContract):
    label: Literal["GOOD", "NEEDS_REVIEW", "BAD"]
    reason: str = Field(pattern=r"^[A-Z0-9_]+$")
    policy: QCPolicy | None = None
    evidence_kind: str | None = None
    assessments: tuple[QCClassAssessment, ...] = ()
    issues: tuple[str, ...] = ()
    rule_trace: tuple[str, ...] = ()
    runtime_mode: Literal["observe_only", "enforce"] = "observe_only"
    applied: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def fail_safe_output(self):
        if self.label != "NEEDS_REVIEW" and self.policy is None:
            raise ValueError("Accepted/rejected decisions require a policy")
        if self.label == "GOOD" and (self.issues or self.policy is None or self.evidence_kind not in {"NORMAL", "KNOWN_DEFECT"}):
            raise ValueError("GOOD cannot waive missing/uncertain rule inputs")
        if self.label == "GOOD":
            if "evidence_completeness_and_policy_gates_passed" not in self.rule_trace:
                raise ValueError("GOOD requires complete auditable input checks")
            if self.evidence_kind == "KNOWN_DEFECT":
                rules = {item.class_name: item for item in self.policy.known_rules}
                if not self.assessments or any(item.outcome != "allowed" or item.class_name not in rules
                        or item.area_fraction > rules[item.class_name].allow_max_fraction
                        or item.allow_max_fraction != rules[item.class_name].allow_max_fraction
                        or item.reject_min_fraction != rules[item.class_name].reject_min_fraction for item in self.assessments):
                    raise ValueError("Known allowance must match explicit business rules")
        if self.applied and self.runtime_mode != "enforce":
            raise ValueError("Observation must not apply final QC")
        return self
