"""Backward-compatible patch API; acceptance rules are owned by the QC service."""

from dataclasses import dataclass

from backend.app.core.qc import LegacyPatchQCInput
from backend.app.services.qc_decision import decide_legacy_patch, legacy_confidence


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    reason: str


def classify_inspection(
    defect_area_ratio: float,
    defect_count: int,
    max_anomaly_score: float | None,
    good_area_threshold: float,
    bad_area_threshold: float,
) -> ClassificationResult:
    return ClassificationResult(*decide_legacy_patch(LegacyPatchQCInput(defect_area_ratio, defect_count, max_anomaly_score,
                                                                       good_area_threshold, bad_area_threshold)))


def _confidence(base: float, max_score: float) -> float:
    return legacy_confidence(base, max_score)
