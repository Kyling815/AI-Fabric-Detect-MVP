from backend.app.ai.classifier import classify_inspection


def test_classifier_returns_good_when_no_defects() -> None:
    result = classify_inspection(
        defect_area_ratio=0.0,
        defect_count=0,
        max_anomaly_score=None,
        good_area_threshold=0.001,
        bad_area_threshold=0.01,
    )

    assert result.label == "Good"


def test_classifier_returns_bad_for_large_area() -> None:
    result = classify_inspection(
        defect_area_ratio=0.02,
        defect_count=1,
        max_anomaly_score=0.9,
        good_area_threshold=0.001,
        bad_area_threshold=0.01,
    )

    assert result.label == "Bad"


def test_classifier_returns_acceptable_for_small_local_defect() -> None:
    result = classify_inspection(
        defect_area_ratio=0.004,
        defect_count=1,
        max_anomaly_score=0.8,
        good_area_threshold=0.001,
        bad_area_threshold=0.01,
    )

    assert result.label == "Acceptable"


def test_classifier_keeps_many_small_regions_acceptable_until_bad_area_threshold() -> None:
    result = classify_inspection(
        defect_area_ratio=0.05,
        defect_count=12,
        max_anomaly_score=1.4,
        good_area_threshold=0.02,
        bad_area_threshold=0.10,
    )

    assert result.label == "Acceptable"
