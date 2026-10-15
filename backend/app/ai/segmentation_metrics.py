"""Pixel confusion matrix/IoU/Dice only; no training or fabricated model accuracy."""

import numpy as np


class SegmentationMetrics:
    def __init__(self, classes: tuple[str, ...], *, ignore_index: int | None = None):
        if len(classes) < 2 or classes[0] != "background" or len(set(classes)) != len(classes):
            raise ValueError("Unique taxonomy starting with background required")
        if ignore_index is not None and (type(ignore_index) is not int or ignore_index < len(classes)):
            raise ValueError("Ignore id must not collide with class ids")
        self.classes, self.ignore_index = classes, ignore_index
        self.confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray):
        predicted, actual = np.asarray(prediction), np.asarray(target)
        if predicted.shape != actual.shape or predicted.size == 0 or predicted.dtype.kind not in "iu" or actual.dtype.kind not in "iu":
            raise ValueError("Nonempty matching integer prediction/target masks required")
        keep = actual != self.ignore_index if self.ignore_index is not None else np.ones(actual.shape, dtype=bool)
        actual, predicted = actual[keep], predicted[keep]
        count = len(self.classes)
        if np.any(actual < 0) or np.any(actual >= count) or np.any(predicted < 0) or np.any(predicted >= count):
            raise ValueError("Invalid prediction/target class ids")
        self.confusion += np.bincount(count * actual.astype(np.int64) + predicted.astype(np.int64), minlength=count**2).reshape(count, count)

    def compute(self) -> dict:
        per_class = {}
        for index, name in enumerate(self.classes):
            tp = int(self.confusion[index, index])
            fp = int(self.confusion[:, index].sum()) - tp
            fn = int(self.confusion[index].sum()) - tp
            union, dice_denom = tp + fp + fn, 2 * tp + fp + fn
            per_class[name] = {"iou": tp / union if union else None, "dice": 2 * tp / dice_denom if dice_denom else None,
                               "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                               "target_pixels": tp + fn, "predicted_pixels": tp + fp}
        def average(metric, foreground=False):
            values = [value[metric] for name, value in per_class.items() if (not foreground or name != "background") and value[metric] is not None]
            return float(np.mean(values)) if values else None
        total = int(self.confusion.sum())
        return {"confusion_matrix": self.confusion.tolist(), "valid_pixels": total,
                "pixel_accuracy": float(np.trace(self.confusion) / total) if total else None,
                "mean_iou_present": average("iou"), "mean_dice_present": average("dice"),
                "foreground_mean_iou_present": average("iou", True), "foreground_mean_dice_present": average("dice", True),
                "per_class": per_class, "absent_class_policy": "undefined metrics are None and excluded from macro means"}
