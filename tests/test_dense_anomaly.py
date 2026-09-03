import numpy as np
import pytest

from backend.app.ai.dense_anomaly import (
    aggregate_image_score,
    localize_anomaly_map,
    resolve_pixel_threshold,
)


def test_top_k_mean_preserves_small_local_anomaly() -> None:
    scores = np.zeros(1000, dtype=np.float32)
    scores[-5:] = 0.8
    assert aggregate_image_score(scores, method="top_k_mean", top_fraction=0.005) == pytest.approx(0.8)


def test_median_contrast_removes_uniform_domain_shift() -> None:
    original = np.full(100, 0.1, dtype=np.float32)
    shifted = np.full(100, 0.5, dtype=np.float32)
    original[-5:] = 0.4
    shifted[-5:] = 0.8

    original_score = aggregate_image_score(
        original,
        method="top_k_median_contrast",
        top_fraction=0.05,
    )
    shifted_score = aggregate_image_score(
        shifted,
        method="top_k_median_contrast",
        top_fraction=0.05,
    )

    assert original_score == pytest.approx(0.3)
    assert shifted_score == pytest.approx(original_score)


def test_median_mad_pixel_threshold_tracks_image_baseline() -> None:
    original = np.full((10, 10), 0.1, dtype=np.float32)
    shifted = np.full((10, 10), 0.5, dtype=np.float32)
    original[5, 5] = 0.8
    shifted[5, 5] = 1.2
    thresholds = {
        "pixel_threshold_mode": "median_mad",
        "pixel_threshold": 0.0,
        "pixel_mad_multiplier": 6.0,
    }

    original_threshold = resolve_pixel_threshold(original, thresholds)
    shifted_threshold = resolve_pixel_threshold(shifted, thresholds)

    assert shifted_threshold - original_threshold == pytest.approx(0.4)
    assert original[5, 5] >= original_threshold
    assert shifted[5, 5] >= shifted_threshold


def test_missing_legacy_pixel_threshold_disables_localization() -> None:
    anomaly_map = np.ones((3, 3), dtype=np.float32)

    assert np.isinf(resolve_pixel_threshold(anomaly_map, {}))


def test_dense_localization_keeps_one_cell_wide_scratch() -> None:
    anomaly_map = np.zeros((20, 20), dtype=np.float32)
    anomaly_map[5:16, 10] = 0.9
    boxes = localize_anomaly_map(
        anomaly_map,
        image_width=200,
        image_height=200,
        pixel_threshold=0.5,
        minimum_defect_area=20,
    )
    assert len(boxes) == 1
    assert boxes[0].h > boxes[0].w


def test_dense_localization_rejects_subminimum_component() -> None:
    anomaly_map = np.zeros((20, 20), dtype=np.float32)
    anomaly_map[10, 10] = 0.9
    assert not localize_anomaly_map(
        anomaly_map,
        image_width=20,
        image_height=20,
        pixel_threshold=0.5,
        minimum_defect_area=2,
    )
