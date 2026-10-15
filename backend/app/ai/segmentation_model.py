"""Independent ResNet18 + U-Net-like decoder, never the cached anomaly encoder."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from backend.app.core.segmentation_data import SegmentationTaxonomy


ARCHITECTURE_VERSION = "independent_resnet18_unet_v1"
PREPROCESSING = "rgb_float_0_1_imagenet_mean_std_in_model_v1"


def cached_resnet18_weights() -> tuple[dict, str]:
    """Trusted local cache only; do not download or silently fall back to random."""
    name = Path(urlparse(models.ResNet18_Weights.IMAGENET1K_V1.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError("Pretrained ResNet18 is not cached; provision trusted weights explicitly before training")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if not digest.startswith(name.rsplit("-", 1)[1].split(".", 1)[0]):
        raise ValueError("Cached ResNet18 hash does not match the official filename prefix")
    return torch.load(path, map_location="cpu", weights_only=True), digest


class DecoderBlock(nn.Module):
    def __init__(self, incoming: int, skip: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming + skip, outgoing, 3, padding=1, bias=False), nn.GroupNorm(1, outgoing), nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(1, outgoing), nn.ReLU(),
        )

    def forward(self, value, skip=None, *, size=None):
        value = F.interpolate(value, size=skip.shape[-2:] if skip is not None else size, mode="bilinear", align_corners=False)
        if skip is not None:
            value = torch.cat((value, skip), dim=1)
        return self.layers(value)


@dataclass(frozen=True)
class SegmentationPrediction:
    probabilities: torch.Tensor  # BCHW, class probabilities, not calibrated.
    mask: torch.Tensor  # BHW, taxonomy ids, includes background.
    confidence: torch.Tensor  # BHW, probability of each winning class, NOT QC confidence.


class ResNet18UNet(nn.Module):
    def __init__(self, taxonomy: SegmentationTaxonomy, *, pretrained: bool = True,
                 decoder_channels: tuple[int, ...] = (128, 64, 32, 16, 16), freeze_encoder_batchnorm: bool = True):
        super().__init__()
        if type(pretrained) is not bool or type(freeze_encoder_batchnorm) is not bool:
            raise ValueError("Explicit model booleans required")
        if len(decoder_channels) != 5 or any(type(value) is not int or not 1 <= value <= 512 for value in decoder_channels):
            raise ValueError("Five bounded decoder widths required")
        self.taxonomy, self.decoder_channels = taxonomy, tuple(decoder_channels)
        self.freeze_encoder_batchnorm = freeze_encoder_batchnorm
        self.encoder = models.resnet18(weights=None)
        self.encoder_weights_sha256 = None
        if pretrained:
            state, self.encoder_weights_sha256 = cached_resnet18_weights()
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.fc = nn.Identity()
        widths = (512, *decoder_channels[:-1])
        skips = (256, 128, 64, 64, 0)
        self.decoder = nn.ModuleList(DecoderBlock(i, s, o) for i, s, o in zip(widths, skips, decoder_channels))
        self.head = nn.Conv2d(decoder_channels[-1], len(taxonomy.classes), 1)
        self.register_buffer("rgb_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder_batchnorm:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()  # Affine parameters still train; running stats do not drift on tiny batches.
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3 or min(images.shape[-2:]) < 32
                or not images.is_floating_point() or not torch.isfinite(images).all()
                or images.min() < 0 or images.max() > 1):
            raise ValueError("Finite BCHW RGB floats in [0,1], dimensions >=32, required")
        value = (images - self.rgb_mean) / self.rgb_std
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        first = self.encoder.layer1(self.encoder.maxpool(stem))
        second = self.encoder.layer2(first)
        third = self.encoder.layer3(second)
        value = self.encoder.layer4(third)
        for block, skip in zip(self.decoder[:4], (third, second, first, stem)):
            value = block(value, skip)
        return self.head(self.decoder[4](value, size=images.shape[-2:]))

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> SegmentationPrediction:
        previous = self.training
        try:
            self.eval()
            logits = self(images).float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite segmentation logits")
            probabilities = logits.softmax(dim=1)
            confidence, mask = probabilities.max(dim=1)
            return SegmentationPrediction(probabilities, mask, confidence)
        finally:
            self.train(previous)
