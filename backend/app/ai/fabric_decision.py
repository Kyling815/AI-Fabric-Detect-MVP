from dataclasses import dataclass

from backend.app.ai.bbox import MergedBox


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
    review = float(thresholds["image_review_threshold"])
    bad = float(thresholds["image_bad_threshold"])
    if bad < review:
        raise ValueError("image_bad_threshold must be >= image_review_threshold")

    localized_review = thresholds.get("localized_review_threshold")
    localized_trigger = (
        bool(boxes)
        and localized_score is not None
        and localized_review is not None
        and localized_score >= float(localized_review)
    )
    if image_score < review and localized_trigger:
        return FabricDecision(label="NEEDS_REVIEW", reason="STRONG_LOCALIZED_ANOMALY")
    if image_score < review:
        return FabricDecision(label="GOOD", reason=None)
    if not boxes:
        return FabricDecision(label="NEEDS_REVIEW", reason="ANOMALY_WITHOUT_LOCALIZATION")
    if image_score >= bad:
        return FabricDecision(label="BAD", reason=None)
    return FabricDecision(label="NEEDS_REVIEW", reason="ANOMALY_BELOW_BAD_THRESHOLD")
