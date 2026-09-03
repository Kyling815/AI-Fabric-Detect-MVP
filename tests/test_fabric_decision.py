from backend.app.ai.bbox import MergedBox
from backend.app.ai.fabric_decision import decide_fabric_inspection


THRESHOLDS = {
    "image_review_threshold": 0.2,
    "image_bad_threshold": 0.4,
}


def test_good_below_review_threshold() -> None:
    decision = decide_fabric_inspection(image_score=0.1, boxes=[], thresholds=THRESHOLDS)
    assert decision.label == "GOOD"


def test_high_anomaly_without_bbox_is_needs_review() -> None:
    decision = decide_fabric_inspection(image_score=0.8, boxes=[], thresholds=THRESHOLDS)
    assert decision.label == "NEEDS_REVIEW"
    assert decision.reason == "ANOMALY_WITHOUT_LOCALIZATION"


def test_high_anomaly_with_bbox_is_bad() -> None:
    box = MergedBox(x=1, y=1, w=5, h=5, score=0.8)
    decision = decide_fabric_inspection(image_score=0.8, boxes=[box], thresholds=THRESHOLDS)
    assert decision.label == "BAD"


def test_strong_localized_evidence_prevents_good_false_negative() -> None:
    box = MergedBox(x=1, y=1, w=5, h=5, score=0.8)
    thresholds = {**THRESHOLDS, "localized_review_threshold": 0.5}

    decision = decide_fabric_inspection(
        image_score=0.1,
        boxes=[box],
        thresholds=thresholds,
        localized_score=0.7,
    )

    assert decision.label == "NEEDS_REVIEW"
    assert decision.reason == "STRONG_LOCALIZED_ANOMALY"


def test_weak_localized_evidence_remains_good() -> None:
    box = MergedBox(x=1, y=1, w=5, h=5, score=0.3)
    thresholds = {**THRESHOLDS, "localized_review_threshold": 0.5}

    decision = decide_fabric_inspection(
        image_score=0.1,
        boxes=[box],
        thresholds=thresholds,
        localized_score=0.3,
    )

    assert decision.label == "GOOD"
