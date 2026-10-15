"""Backward-compatible dense API; acceptance rules are owned by the QC service."""

from dataclasses import dataclass

from backend.app.ai.bbox import MergedBox
from backend.app.core.qc import LegacyDenseQCInput
from backend.app.services.qc_decision import decide_legacy_dense


@dataclass(frozen=True)
class FabricDecision:
    label: str
    reason: str | None


def decide_fabric_inspection(
    *,
    image_score: float,
    boxes: list[MergedBox] | tuple[MergedBox, ...],
    thresholds: dict,
    localized_score: float | None = None,
) -> FabricDecision:
    inputs = LegacyDenseQCInput(image_score=image_score, has_localization=bool(boxes),
                review_threshold=float(thresholds["image_review_threshold"]), bad_threshold=float(thresholds["image_bad_threshold"]),
                localized_score=localized_score, localized_review_threshold=thresholds.get("localized_review_threshold"))
    return FabricDecision(*decide_legacy_dense(inputs))
