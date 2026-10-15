"""Cross entropy (optionally focal-modulated) + differentiable multiclass Dice."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, *, ignore_index: int | None = None, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 focal_gamma: float | None = None):
        super().__init__()
        if (not all(math.isfinite(value) and value >= 0 for value in (ce_weight, dice_weight))
                or ce_weight + dice_weight <= 0
                or (focal_gamma is not None and (not math.isfinite(focal_gamma) or focal_gamma < 0))
                or (ignore_index is not None and (type(ignore_index) is not int or ignore_index < 0))):
            raise ValueError("Invalid segmentation loss configuration")
        self.ignore_index, self.ce_weight, self.dice_weight, self.focal_gamma = ignore_index, ce_weight, dice_weight, focal_gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (logits.ndim != 4 or logits.shape[1] < 2 or logits.shape[0] < 1 or logits.numel() == 0
                or target.shape != (logits.shape[0], *logits.shape[-2:]) or target.dtype != torch.long
                or not logits.is_floating_point() or not torch.isfinite(logits).all()):
            raise ValueError("Finite BCHW logits and matching BHW int64 targets required")
        classes = logits.shape[1]
        if self.ignore_index is not None and self.ignore_index < classes:
            raise ValueError("Ignore id collides with active classes")
        valid = target != self.ignore_index if self.ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
        if not valid.any() or (target[valid] < 0).any() or (target[valid] >= classes).any():
            raise ValueError("No valid target pixels or invalid class ids")
        safe_target = torch.where(valid, target, 0)
        scores = logits.float()  # Dice and probability math stays FP32 even under CUDA autocast.
        pixel_ce = F.cross_entropy(scores, safe_target, reduction="none")
        if self.focal_gamma is not None:
            pixel_ce = (1 - (-pixel_ce).exp()).pow(self.focal_gamma) * pixel_ce
        ce = pixel_ce[valid].mean()
        probabilities = scores.softmax(dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_target, num_classes=classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        axes = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(axes)
        denominator = probabilities.sum(axes) + one_hot.sum(axes)
        # All active classes participate, including background; false positives are penalized.
        dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        return self.ce_weight * ce + self.dice_weight * dice
