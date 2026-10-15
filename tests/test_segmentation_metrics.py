import numpy as np
import pytest

from backend.app.ai.segmentation_metrics import SegmentationMetrics


def test_known_confusion_iou_dice_and_pixel_accuracy():
    metric = SegmentationMetrics(("background", "defect"))
    target = np.array([[0, 0], [1, 1]], dtype=np.int64)
    prediction = np.array([[0, 1], [0, 1]], dtype=np.int64)
    metric.update(prediction, target)
    value = metric.compute()
    assert value["confusion_matrix"] == [[1, 1], [1, 1]]
    assert value["pixel_accuracy"] == 0.5
    assert value["per_class"]["defect"]["iou"] == pytest.approx(1 / 3)
    assert value["per_class"]["defect"]["dice"] == 0.5


def test_empty_or_absent_foreground_does_not_invent_perfect_score():
    metric = SegmentationMetrics(("background", "scratch", "stain"))
    assert metric.compute()["pixel_accuracy"] is None
    metric.update(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64))
    value = metric.compute()
    assert value["foreground_mean_iou_present"] is None
    assert value["per_class"]["scratch"]["iou"] is None


def test_false_positive_absent_class_is_zero_not_excluded():
    metric = SegmentationMetrics(("background", "defect"))
    metric.update(np.ones((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64))
    assert metric.compute()["per_class"]["defect"]["iou"] == 0


def test_ignore_pixels_and_streaming_accumulation():
    metric = SegmentationMetrics(("background", "defect"), ignore_index=255)
    prediction, target = np.array([[0, 1, 0]]), np.array([[0, 1, 255]])
    metric.update(prediction, target); metric.update(prediction, target)
    assert metric.compute()["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metric.compute()["valid_pixels"] == 4
    metric.update(np.array([[99]]), np.array([[255]]))
    assert metric.compute()["valid_pixels"] == 4


@pytest.mark.parametrize("prediction,target", [(np.zeros((2, 2)), np.zeros((2, 2))), (np.zeros((2, 2), dtype=int), np.zeros((3, 2), dtype=int)),
                                             (np.array([[2]]), np.array([[0]])), (np.array([[0]]), np.array([[-1]])), (np.array([], dtype=int), np.array([], dtype=int))])
def test_bad_metric_masks_rejected(prediction, target):
    with pytest.raises(ValueError):
        SegmentationMetrics(("background", "defect")).update(prediction, target)
