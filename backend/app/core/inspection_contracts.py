"""Internal contracts for the existing dense inspection path.

These are not HTTP/DB schemas. Existing evidence, box and decision types remain
the source of truth; no new scoring or validation policy is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from backend.app.ai.bbox import MergedBox
from backend.app.ai.dense_anomaly import DenseAnomalyResult
from backend.app.ai.dense_feature_extractor import DenseFeatureExtractorInfo, DenseFeatureResult
from backend.app.ai.fabric_decision import FabricDecision
from backend.app.services.fabric_registry import FabricReferenceEntry
from backend.app.core.acquisition import AcquisitionMetadata, QualityGateConfig
from backend.app.core.pattern import PatternEstimatorConfig
from backend.app.core.segmentation_shadow import SupervisedShadowConfig, SupervisedShadowObservation
from backend.app.core.fusion import DefectEvidence, FusionPolicy
from backend.app.core.qc import QCDecision, QCPolicy
from backend.app.core.runtime_profiling import RuntimeProfile

if TYPE_CHECKING:
    from backend.app.ai.quality_gate import QualityGateResult
    from backend.app.ai.pattern_period import PatternPeriodResult


# Reuse the existing candidate representation, including area and patch_count.
DefectCandidate = MergedBox


@dataclass(frozen=True)
class InspectionInput:
    image_path: Path
    fabric_id: str
    metadata: AcquisitionMetadata = field(default_factory=AcquisitionMetadata)


@dataclass(frozen=True)
class InspectionContext:
    reference: FabricReferenceEntry
    backbone: Mapping[str, Any]
    device: str
    default_image_max_size: int
    quality_gate_config: QualityGateConfig | None = None
    pattern_estimator_config: PatternEstimatorConfig | None = None
    pattern_observation_error: str | None = None
    supervised_shadow_config: SupervisedShadowConfig | None = None
    supervised_shadow_error: str | None = None
    fusion_policy: FusionPolicy | None = None
    fusion_error: str | None = None
    qc_policy: QCPolicy | None = None
    qc_policy_error: str | None = None
    hybrid_qc_mode: Literal["observe_only", "enforce"] = "observe_only"


@dataclass(frozen=True)
class DenseAnomalyEvidence:
    features: DenseFeatureResult
    extractor_info: DenseFeatureExtractorInfo
    anomaly: DenseAnomalyResult


@dataclass(frozen=True)
class InspectionResult:
    evidence: DenseAnomalyEvidence
    candidates: tuple[DefectCandidate, ...]
    decision: FabricDecision
    defect_area_ratio: float
    quality_gate_result: QualityGateResult | None = None
    pattern_period_result: PatternPeriodResult | None = None
    known_defects_shadow: SupervisedShadowObservation | None = None
    defect_evidence: DefectEvidence | None = None
    qc_evaluation: QCDecision | None = None
    legacy_decision: FabricDecision | None = None
    runtime_profile: RuntimeProfile | None = None
