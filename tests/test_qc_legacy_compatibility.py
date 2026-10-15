from itertools import product

import pytest

from backend.app.ai.bbox import MergedBox
from backend.app.ai.classifier import classify_inspection
from backend.app.ai.fabric_decision import decide_fabric_inspection
from backend.app.core.qc import LegacyDenseQCInput, LegacyPatchQCInput
from backend.app.services.qc_decision import decide_legacy_dense, decide_legacy_patch


@pytest.mark.parametrize("score,has_boxes,localized", list(product((0.0, 0.299999, 0.3, 0.699999, 0.7, 1.0), (False, True), (False, True))))
def test_dense_legacy_exact_labels_reasons_at_boundaries(score, has_boxes, localized):
    thresholds = dict(image_review_threshold=0.3, image_bad_threshold=0.7, localized_review_threshold=0.8)
    boxes = [MergedBox(0, 0, 2, 2, score)] if has_boxes else []
    local_score = 0.9 if localized else 0.0
    expected = (("NEEDS_REVIEW", "STRONG_LOCALIZED_ANOMALY") if score < 0.3 and has_boxes and localized else
                ("GOOD", None) if score < 0.3 else ("NEEDS_REVIEW", "ANOMALY_WITHOUT_LOCALIZATION") if not has_boxes else
                ("BAD", None) if score >= 0.7 else ("NEEDS_REVIEW", "ANOMALY_BELOW_BAD_THRESHOLD"))
    result = decide_fabric_inspection(image_score=score, boxes=boxes, thresholds=thresholds, localized_score=local_score)
    assert (result.label, result.reason) == expected
    assert decide_legacy_dense(LegacyDenseQCInput(score, has_boxes, 0.3, 0.7, local_score, 0.8)) == expected


@pytest.mark.parametrize("area,count,score", list(product((0.0, 0.001, 0.02, 0.1), (0, 1), (None, 0.0, 0.9))))
def test_patch_legacy_exact_labels_confidence_and_reasons(area, count, score):
    result = classify_inspection(area, count, score, 0.01, 0.1)
    expected_label = "Good" if count == 0 else "Bad" if area >= 0.1 else "Acceptable"
    base = 0.9 if count == 0 else 0.75 + min(area, 0.2) if area >= 0.1 else 0.7
    expected_confidence = max(0.0, min(0.99, base + min(score or 0.0, 0.2)))
    expected_reason = ("No localized suspicious defect was detected." if count == 0 else
        "Defect area exceeded the configured bad threshold." if area >= 0.1 else
        "A localized defect was detected below the configured minimum area threshold." if area < 0.01 else
        "A small localized suspicious area was detected below the bad threshold.")
    assert (result.label, result.confidence, result.reason) == (expected_label, expected_confidence, expected_reason)
    assert decide_legacy_patch(LegacyPatchQCInput(area, count, score, 0.01, 0.1)) == (expected_label, expected_confidence, expected_reason)


def test_legacy_threshold_error_is_unchanged():
    with pytest.raises(ValueError, match="image_bad_threshold must be >= image_review_threshold"):
        decide_fabric_inspection(image_score=0, boxes=[], thresholds=dict(image_review_threshold=1, image_bad_threshold=0))
