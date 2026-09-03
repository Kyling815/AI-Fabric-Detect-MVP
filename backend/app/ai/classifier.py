from dataclasses import dataclass


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
    max_score = max_anomaly_score or 0.0
    if defect_count == 0:
        return ClassificationResult(
            label="Good",
            confidence=_confidence(0.9, max_score),
            reason="No localized suspicious defect was detected.",
        )

    if defect_area_ratio >= bad_area_threshold:
        return ClassificationResult(
            label="Bad",
            confidence=_confidence(0.75 + min(defect_area_ratio, 0.2), max_score),
            reason="Defect area exceeded the configured bad threshold.",
        )

    reason = (
        "A localized defect was detected below the configured minimum area threshold."
        if defect_area_ratio < good_area_threshold
        else "A small localized suspicious area was detected below the bad threshold."
    )
    return ClassificationResult(
        label="Acceptable",
        confidence=_confidence(0.7, max_score),
        reason=reason,
    )


def _confidence(base: float, max_score: float) -> float:
    return max(0.0, min(0.99, base + min(max_score, 0.2)))
