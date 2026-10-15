"""Evidence contracts only. No customer QC rules, learned fusion or GOOD/BAD output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.segmentation_data import DEFECT_CLASSES, SegmentationTaxonomy


BranchStatus = Literal["available", "missing", "disabled", "failed", "timeout", "busy", "invalid"]
ScoreMethod = Literal["top_k_mean", "max", "percentile_995", "max_median_contrast", "percentile_995_median_contrast", "top_k_median_contrast"]


class FusionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def no_numeric_bool(cls, value, info):
        # Explicit numeric/id checks are on each field's strict type; floats must not coerce booleans/strings.
        if info.field_name in {"confidence", "score", "localized_score", "known_threshold", "anomaly_high_threshold", "anomaly_review_threshold", "known_confidence", "anomaly_score"}:
            if isinstance(value, (bool, str)):
                raise ValueError("Evidence scores and thresholds must be numeric")
        return value


class FusionPolicy(FusionContract):
    mode: Literal["observe_only"] = "observe_only"  # Business/QC integration belongs to a separately approved phase.
    policy_version: str = Field(default="fusion_starting_rules_v1", min_length=1, max_length=128)
    known_threshold: float = Field(default=0.9, gt=0, le=1)
    anomaly_high_threshold: float = Field(default=0.7, gt=0)
    anomaly_review_threshold: float = Field(default=0.3, gt=0)
    anomaly_score_method: ScoreMethod = "top_k_mean"
    allow_synthetic_supervised: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def coherent_thresholds(self):
        if self.anomaly_high_threshold <= self.anomaly_review_threshold:
            raise ValueError("Anomaly high threshold must exceed review threshold")
        if not self.policy_version.strip() or self.policy_version != self.policy_version.strip():
            raise ValueError("Nonblank policy version required")
        return self


class EvidenceIdentity(FusionContract):
    fabric_id: str = Field(min_length=1, max_length=128)
    capture_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def clean_ids(self):
        if any(value != value.strip() or not value.strip() for value in (self.fabric_id, self.capture_id)):
            raise ValueError("Evidence identity must be nonblank without surrounding whitespace")
        return self


class CaptureEvidence(FusionContract):
    status: Literal["valid", "invalid", "unchecked", "error"] = "unchecked"
    source: Literal["quality_gate", "caller_declared", "not_checked"] = "not_checked"
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def meaningful_capture(self):
        if self.status in {"valid", "invalid"} and self.source == "not_checked":
            raise ValueError("Checked capture needs explicit producer")
        if self.status == "valid" and self.reasons:
            raise ValueError("Valid capture must not contain rejection reasons")
        if self.status in {"invalid", "error"} and not self.reasons:
            raise ValueError("Invalid/error capture needs reasons")
        return self


class KnownClassEvidence(FusionContract):
    class_name: str
    class_id: int = Field(ge=1, strict=True)
    confidence: float = Field(ge=0, le=1)  # Mean winning pixel confidence for this foreground class, NOT QC confidence.
    predicted_pixels: int = Field(ge=1, strict=True)

    @field_validator("class_name")
    @classmethod
    def supported_foreground(cls, value):
        if value not in DEFECT_CLASSES[1:]:
            raise ValueError("Supported foreground class required, never background")
        return value


class MaskDescriptor(FusionContract):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shape: tuple[int, int]
    original_shape: tuple[int, int]
    coordinate_frame: Literal["model_grid"] = "model_grid"
    payload_available: bool = Field(strict=True)

    @field_validator("shape", "original_shape", mode="before")
    @classmethod
    def strict_shapes(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(item) is not int or item < 1 for item in value):
            raise ValueError("Positive integer pixel shape required")
        return value

    @model_validator(mode="after")
    def bounded_grid(self):
        if max(self.shape) > 512:
            raise ValueError("Supervised model grid exceeds shadow bounds")
        return self


class SupervisedEvidence(FusionContract):
    status: BranchStatus
    reason: str | None = None
    identity: EvidenceIdentity | None = None
    model_version: str | None = Field(default=None, min_length=1)
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    synthetic: bool | None = Field(default=None, strict=True)
    taxonomy: tuple[str, ...] = ()
    candidates: tuple[KnownClassEvidence, ...] = ()
    mask: MaskDescriptor | None = None
    confidence_definition: Literal["mean_winning_pixel_probability_per_foreground_class"] = "mean_winning_pixel_probability_per_foreground_class"
    confidence_calibrated: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def meaningful_supervised(self):
        if self.status != "available":
            if self.candidates or self.mask or self.taxonomy or self.model_version or self.checkpoint_sha256 or self.synthetic is not None:
                raise ValueError("Unavailable supervised branch cannot fabricate verified evidence")
            return self
        if self.identity is None or not self.model_version or not self.checkpoint_sha256 or self.synthetic is None or self.mask is None:
            raise ValueError("Available supervised evidence needs identity, provenance and mask geometry")
        SegmentationTaxonomy(classes=self.taxonomy)
        labels = [item.class_id for item in self.candidates]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate supervised class ids")
        for item in self.candidates:
            if item.class_id >= len(self.taxonomy) or self.taxonomy[item.class_id] != item.class_name:
                raise ValueError("Foreground class name/id does not match taxonomy")
        if sum(item.predicted_pixels for item in self.candidates) > self.mask.shape[0] * self.mask.shape[1]:
            raise ValueError("Foreground counts exceed mask grid")
        return self


class FusionBox(FusionContract):
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    w: int = Field(gt=0, strict=True)
    h: int = Field(gt=0, strict=True)
    score: float = Field(ge=0)
    label: str = Field(min_length=1)
    coordinate_frame: Literal["oriented_source_pixels"] = "oriented_source_pixels"


class AnomalyEvidence(FusionContract):
    status: BranchStatus
    reason: str | None = None
    identity: EvidenceIdentity | None = None
    reference_version: str | None = Field(default=None, min_length=1)
    backbone_version: str | None = Field(default=None, min_length=1)
    score: float | None = Field(default=None, ge=0)
    localized_score: float | None = Field(default=None, ge=0)
    image_shape: tuple[int, int] | None = None
    boxes: tuple[FusionBox, ...] = ()
    score_method: ScoreMethod | None = None
    score_kind: Literal["dense_nn_distance_aggregate"] = "dense_nn_distance_aggregate"  # Not a probability, no [0,1] clipping.
    localized_review_trigger: bool = Field(default=False, strict=True)

    @field_validator("image_shape", mode="before")
    @classmethod
    def strict_shape(cls, value):
        if value is not None and (not isinstance(value, (list, tuple)) or any(type(item) is not int or item < 1 for item in value)):
            raise ValueError("Positive original pixel shape required")
        return value

    @model_validator(mode="after")
    def meaningful_anomaly(self):
        if self.status != "available":
            if self.score is not None or self.localized_score is not None or self.boxes or self.image_shape or self.localized_review_trigger or self.reference_version or self.backbone_version:
                raise ValueError("Unavailable anomaly branch cannot fabricate scored evidence")
            return self
        if self.identity is None or self.score is None or not self.reference_version or not self.backbone_version or not self.image_shape or self.score_method is None:
            raise ValueError("Available anomaly evidence needs score, identity, reference/backbone and geometry")
        if any(box.x + box.w > self.image_shape[1] or box.y + box.h > self.image_shape[0] for box in self.boxes):
            raise ValueError("Anomaly boxes exceed source image bounds")
        if self.localized_review_trigger and (not self.boxes or self.localized_score is None):
            raise ValueError("Localized review trigger needs localization evidence")
        return self


class RuleCheck(FusionContract):
    rule: str
    passed: bool = Field(strict=True)


class DefectEvidence(FusionContract):
    kind: Literal["KNOWN_DEFECT", "UNKNOWN_DEFECT", "NEEDS_REVIEW", "NORMAL", "INVALID_CAPTURE"]
    rule_id: str = Field(min_length=1)
    policy: FusionPolicy | None = None
    capture_status: Literal["valid", "invalid", "unchecked", "error"]
    capture_source: Literal["quality_gate", "caller_declared", "not_checked"] = "not_checked"
    capture_reasons: tuple[str, ...] = ()
    supervised_status: BranchStatus
    anomaly_status: BranchStatus
    identity: EvidenceIdentity | None = None
    supervised_model_version: str | None = None
    supervised_checkpoint_sha256: str | None = None
    supervised_synthetic: bool | None = Field(default=None, strict=True)
    supervised_confidence_calibrated: bool | None = Field(default=None, strict=True)
    known_confidence_definition: Literal["max_foreground_class_mean_winning_probability"] = "max_foreground_class_mean_winning_probability"
    anomaly_reference_version: str | None = None
    known_confidence: float | None = Field(default=None, ge=0, le=1)
    known_defects: tuple[KnownClassEvidence, ...] = ()
    supervised_mask: MaskDescriptor | None = None
    anomaly_score: float | None = Field(default=None, ge=0)
    anomaly_boxes: tuple[FusionBox, ...] = ()
    uncertainties: tuple[str, ...] = ()
    review_required: bool = Field(strict=True)
    trace: tuple[RuleCheck, ...] = ()
    affects_qc: Literal[False] = False

    @field_validator("affects_qc", mode="before")
    @classmethod
    def no_qc_activation(cls, value):
        if type(value) is not bool or value:
            raise ValueError("Phase 10 evidence must not activate final QC rules")
        return value

    @model_validator(mode="after")
    def fail_safe_result(self):
        if self.capture_status == "invalid" and self.kind != "INVALID_CAPTURE":
            raise ValueError("Invalid capture takes precedence over defect classification")
        if self.capture_status in {"unchecked", "error"} and self.kind != "NEEDS_REVIEW":
            raise ValueError("Unverified/error capture cannot claim classified or normal fabric")
        if (self.uncertainties or self.kind in {"NEEDS_REVIEW", "INVALID_CAPTURE"}) and not self.review_required:
            raise ValueError("Uncertain/invalid fusion cannot waive required review")
        if self.kind == "NORMAL":
            if (self.capture_status != "valid" or self.supervised_status != "available" or self.anomaly_status != "available"
                    or self.uncertainties or self.review_required or self.known_defects or self.policy is None or self.anomaly_score is None
                    or self.identity is None or not self.supervised_model_version or not self.supervised_checkpoint_sha256 or not self.anomaly_reference_version
                    or self.supervised_mask is None or self.supervised_synthetic is None
                    or self.capture_source == "not_checked"
                    or (self.supervised_synthetic and not self.policy.allow_synthetic_supervised)
                    or self.anomaly_score >= self.policy.anomaly_review_threshold
                    or (self.known_confidence is not None and self.known_confidence >= self.policy.known_threshold)):
                raise ValueError("NORMAL requires complete valid evidence below thresholds")
            checks = {item.rule: item.passed for item in self.trace}
            if (any(checks.get(name) is not True for name in ("capture_valid", "both_branches_available", "identity_compatible",
                    "geometry_compatible", "synthetic_supervised_allowed", "anomaly_score_method_matches"))
                    or any(checks.get(name) is not False for name in ("known_foreground_at_threshold", "high_anomaly_at_threshold",
                    "anomaly_review_at_threshold", "localized_anomaly_review"))):
                raise ValueError("NORMAL requires an auditable complete negative rule trace")
        if self.kind == "KNOWN_DEFECT" and (not self.known_defects or self.supervised_mask is None or self.policy is None
                or self.supervised_status != "available" or not self.supervised_model_version or not self.supervised_checkpoint_sha256
                or self.supervised_synthetic is None or (self.supervised_synthetic and not self.policy.allow_synthetic_supervised)
                or any(item.confidence < self.policy.known_threshold for item in self.known_defects)):
            raise ValueError("Known defect needs explicit foreground evidence and policy")
        if self.kind == "UNKNOWN_DEFECT" and (self.policy is None or self.anomaly_status != "available" or not self.anomaly_reference_version
                or self.anomaly_score is None or self.anomaly_score < self.policy.anomaly_high_threshold):
            raise ValueError("Unknown defect needs high anomaly evidence")
        if self.kind == "INVALID_CAPTURE" and self.capture_status != "invalid":
            raise ValueError("Invalid capture is not a fabric defect label")
        return self
