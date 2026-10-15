"""Tiny Phase 8 configs, temporary fixtures and isolated deterministic CPU settings."""

import pytest
import torch

from backend.app.ai.segmentation_trainer import training_reproducibility
from backend.app.core.segmentation_training import SegmentationBaselineConfig


def small_config(*, training_updates=None, **updates):
    values = SegmentationBaselineConfig().model_dump(mode="json")
    values["pipeline"]["training"].update(enabled=True, allow_synthetic=True, image_size=[32, 32], batch_size=3, epochs=2)
    values["pipeline"]["training"].update(training_updates or {})
    values.update(pretrained_encoder=False, device="cpu", decoder_channels=[16, 8, 8, 4, 4])
    values.update(updates)
    return SegmentationBaselineConfig.model_validate(values)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with training_reproducibility(42, 2):
        yield


def overfit_batch():
    mask = torch.zeros((2, 32, 32), dtype=torch.long)
    mask[1, 8:24, 12:20] = 1
    image = torch.full((2, 3, 32, 32), 0.15)
    image[1, :, 8:24, 12:20] = 0.9
    return image, mask
