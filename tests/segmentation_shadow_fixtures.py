"""Synthetic fixtures only: artificially strong foreground output, not a trained model claim."""

import hashlib

from PIL import Image
import pytest
import torch

from backend.app.ai.segmentation_dataset import sha256_file
from backend.app.ai.segmentation_model import ResNet18UNet
from backend.app.ai.segmentation_trainer import save_segmentation_checkpoint, training_reproducibility
from backend.app.core.segmentation_shadow import ShadowClassSummary, SupervisedShadowConfig, SupervisedShadowObservation
from tests.segmentation_training_fixtures import small_config


def shadow_config(**updates):
    values = dict(checkpoint_path="unused_fixture.pt", checkpoint_sha256="1" * 64, expected_model_version="fixture_shadow_v1",
                  trusted_local=True, allowed_fabric_ids=("fixture_fabric",), allow_synthetic=True)
    values.update(updates)
    return SupervisedShadowConfig(**values)


def observed_fixture(config=None):
    config = config or shadow_config()
    return SupervisedShadowObservation(status="observed", reason="shadow_only", requested_model_version=config.expected_model_version,
                model_version=config.expected_model_version, checkpoint_sha256=config.checkpoint_sha256, synthetic_checkpoint=True,
                original_shape=(32, 48), mask_shape=(32, 32), taxonomy=("background", "defect"),
                mask_sha256=hashlib.sha256(b"\x01\x00\x00\x00" * 1024).hexdigest(),
                classes=(ShadowClassSummary(name="background", predicted_pixels=0, mean_class_probability=0.01),
                         ShadowClassSummary(name="defect", predicted_pixels=1024, mean_class_probability=0.99, mean_winning_confidence=0.99)),
                mask_rle=((1, 1024),))


@pytest.fixture
def shadow_checkpoint(tmp_path):
    config = small_config(training_updates={"epochs": 1})
    version = tmp_path / "fixture_shadow_v1"
    version.mkdir()
    path = version / "fixture.pt"
    with training_reproducibility(42, 2):
        model = ResNet18UNet(config.pipeline.taxonomy, pretrained=False, decoder_channels=config.decoder_channels)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.copy_(torch.tensor((-4.0, 4.0)))
        save_segmentation_checkpoint(path, model, config=config, epoch=1, validation_loss=1,
                                     dataset_fingerprint="a" * 64, synthetic=True)
    image = tmp_path / "image.png"
    Image.new("RGB", (48, 32), (180, 180, 180)).save(image)
    return image, shadow_config(checkpoint_path=str(path), checkpoint_sha256=sha256_file(path))


def enable_shadow(runtime, config=None, **updates):
    values = (config or shadow_config()).model_dump(mode="json")
    values.update(updates)
    runtime.registry["fabrics"]["fixture_fabric"]["supervised"] = {"enabled": True, **values}
    runtime.save_registry()
