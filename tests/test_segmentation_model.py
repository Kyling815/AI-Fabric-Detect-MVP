from unittest.mock import patch

import pytest
import torch

from backend.app.ai.dense_feature_extractor import SharedDenseResNet18
from backend.app.ai.segmentation_loss import CrossEntropyDiceLoss
from backend.app.ai.segmentation_metrics import SegmentationMetrics
from backend.app.ai.segmentation_model import ResNet18UNet, cached_resnet18_weights
from backend.app.core.segmentation_data import SegmentationTaxonomy
from tests.segmentation_training_fixtures import deterministic_cpu, overfit_batch


@pytest.mark.parametrize("size,classes", [((32, 32), ("background", "defect")), ((35, 49), ("background", "scratch", "stain"))])
def test_forward_shape_and_prediction_contract(size, classes):
    model = ResNet18UNet(SegmentationTaxonomy(classes=classes), pretrained=False, decoder_channels=(16, 8, 8, 4, 4))
    image = torch.rand(1, 3, *size)
    assert model(image).shape == (1, len(classes), *size)
    result = model.predict(image)
    assert result.mask.shape == result.confidence.shape == (1, *size)
    assert result.mask.dtype == torch.long
    assert torch.allclose(result.probabilities.sum(1), torch.ones_like(result.confidence), atol=1e-6)
    assert torch.equal(result.mask, result.probabilities.argmax(1))
    assert torch.equal(result.confidence, result.probabilities.max(1).values)
    assert model.training  # Prediction restored mode.


def test_backward_updates_both_independent_encoder_and_decoder():
    model = ResNet18UNet(SegmentationTaxonomy(), pretrained=False, decoder_channels=(16, 8, 8, 4, 4))
    image, mask = overfit_batch()
    loss = CrossEntropyDiceLoss()(model(image), mask)
    assert torch.isfinite(loss)
    loss.backward()
    for weight in (model.encoder.conv1.weight, model.decoder[0].layers[0].weight, model.head.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0


def test_one_batch_overfit_is_learning_not_oracle():
    model = ResNet18UNet(SegmentationTaxonomy(), pretrained=False, decoder_channels=(16, 8, 8, 4, 4))
    image, mask = overfit_batch()
    criterion = CrossEntropyDiceLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": list(model.decoder.parameters()) + list(model.head.parameters()), "lr": 0.01},
    ], weight_decay=0)  # Explicit aggressive fixture-only rates, not production tuning.
    initial = float(criterion(model(image), mask).detach())
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), mask)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    result = model.predict(image)
    final = float(criterion(model(image), mask).detach())
    metrics = SegmentationMetrics(model.taxonomy.classes)
    metrics.update(result.mask.numpy(), mask.numpy())
    assert final < initial * 0.1
    assert metrics.compute()["per_class"]["defect"]["iou"] > 0.95


def test_prediction_deterministic_and_batchnorm_statistics_frozen():
    model = ResNet18UNet(SegmentationTaxonomy(), pretrained=False, decoder_channels=(16, 8, 8, 4, 4))
    before = model.encoder.bn1.running_mean.clone()
    images, _ = overfit_batch()
    model(images)
    assert torch.equal(before, model.encoder.bn1.running_mean)
    assert model.encoder.bn1.weight.requires_grad and model.head.training
    first, second = model.predict(images), model.predict(images)
    assert torch.equal(first.probabilities, second.probabilities)


def test_pretrained_encoder_is_new_instance_not_shared_anomaly_weights():
    anomaly = SharedDenseResNet18(device="cpu", use_pretrained_weights=True)
    before = anomaly.model.conv1.weight.detach().clone()
    supervised = ResNet18UNet(SegmentationTaxonomy(), pretrained=True, decoder_channels=(16, 8, 8, 4, 4))
    assert supervised.encoder_weights_sha256.startswith("f37072fd")
    assert torch.equal(supervised.encoder.conv1.weight, before)
    assert supervised.encoder.conv1.weight.data_ptr() != anomaly.model.conv1.weight.data_ptr()
    with torch.no_grad():
        supervised.encoder.conv1.weight.add_(1)
    assert torch.equal(anomaly.model.conv1.weight, before)


def test_missing_pretrained_cache_fails_without_download_or_random_fallback(tmp_path):
    with patch("torch.hub.get_dir", return_value=str(tmp_path)), patch("torch.hub.download_url_to_file", side_effect=AssertionError("no download")):
        with pytest.raises(FileNotFoundError, match="not cached"):
            cached_resnet18_weights()


def test_pretrained_cache_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "checkpoints/resnet18-f37072fd.pth"
    path.parent.mkdir()
    path.write_bytes(b"not official weights")
    with patch("torch.hub.get_dir", return_value=str(tmp_path)):
        with pytest.raises(ValueError, match="hash"):
            cached_resnet18_weights()


@pytest.mark.parametrize("failure", ["small", "channels", "integer", "nan", "range"])
def test_invalid_model_input_rejected(failure):
    model = ResNet18UNet(SegmentationTaxonomy(), pretrained=False, decoder_channels=(16, 8, 8, 4, 4))
    image = {"small": torch.rand(1, 3, 16, 32), "channels": torch.rand(1, 1, 32, 32),
             "integer": torch.zeros(1, 3, 32, 32, dtype=torch.long), "nan": torch.full((1, 3, 32, 32), float("nan")),
             "range": torch.ones(1, 3, 32, 32) * 2}[failure]
    with pytest.raises(ValueError):
        model(image)
