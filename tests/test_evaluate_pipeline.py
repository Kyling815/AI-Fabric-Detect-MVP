from scripts.evaluate_pipeline import ImageScore, choose_best_threshold, compute_threshold_metrics, parse_thresholds


def test_parse_thresholds_sorts_unique_values() -> None:
    assert parse_thresholds("0.2, 0.1, 0.2") == [0.1, 0.2]


def test_compute_threshold_metrics_counts_binary_outcomes() -> None:
    scores = [
        ImageScore(
            path="normal.jpg",
            class_name="non defective",
            expected_defective=False,
            patch_count=2,
            max_anomaly_score=0.02,
            mean_anomaly_score=0.01,
            anomaly_scores=[0.01, 0.02],
            boxes=[
                {"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10},
                {"patch_index": 1, "x": 10, "y": 0, "w": 10, "h": 10},
            ],
        ),
        ImageScore(
            path="defect.jpg",
            class_name="holes",
            expected_defective=True,
            patch_count=2,
            max_anomaly_score=0.6,
            mean_anomaly_score=0.3,
            anomaly_scores=[0.01, 0.6],
            boxes=[
                {"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10},
                {"patch_index": 1, "x": 10, "y": 0, "w": 10, "h": 10},
            ],
        ),
    ]

    metrics = compute_threshold_metrics(
        scores,
        threshold=0.5,
        min_area_ratio=0.001,
        min_suspicious_patches=1,
        min_cluster_patches=1,
    )

    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.tn == 1
    assert metrics.fn == 0
    assert metrics.f1 == 1.0


def test_choose_best_threshold_prefers_highest_f1() -> None:
    low = compute_threshold_metrics(
        [
            ImageScore("normal.jpg", "non defective", False, 1, 0.8, 0.8, [0.8], [{"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10}]),
            ImageScore("defect.jpg", "holes", True, 1, 0.9, 0.9, [0.9], [{"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10}]),
        ],
        threshold=0.5,
        min_area_ratio=0.001,
        min_suspicious_patches=1,
        min_cluster_patches=1,
    )
    high = compute_threshold_metrics(
        [
            ImageScore("normal.jpg", "non defective", False, 1, 0.8, 0.8, [0.8], [{"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10}]),
            ImageScore("defect.jpg", "holes", True, 1, 0.9, 0.9, [0.9], [{"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10}]),
        ],
        threshold=0.85,
        min_area_ratio=0.001,
        min_suspicious_patches=1,
        min_cluster_patches=1,
    )

    assert choose_best_threshold([low, high]).threshold == 0.85


def test_compute_threshold_metrics_can_ignore_single_patch_noise() -> None:
    scores = [
        ImageScore(
            path="normal.jpg",
            class_name="non defective",
            expected_defective=False,
            patch_count=2,
            max_anomaly_score=0.8,
            mean_anomaly_score=0.4,
            anomaly_scores=[0.8, 0.01],
            boxes=[
                {"patch_index": 0, "x": 0, "y": 0, "w": 10, "h": 10},
                {"patch_index": 1, "x": 20, "y": 0, "w": 10, "h": 10},
            ],
        )
    ]

    metrics = compute_threshold_metrics(
        scores,
        threshold=0.5,
        min_area_ratio=0.001,
        min_suspicious_patches=1,
        min_cluster_patches=2,
    )

    assert metrics.tn == 1
    assert metrics.fp == 0
