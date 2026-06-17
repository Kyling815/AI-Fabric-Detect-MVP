from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelSpec:
    name: str
    embedding_dim: int
    build: Callable[[bool], nn.Module]
    weights_url: str


def get_model_spec(model_name: str) -> ModelSpec:
    normalized = model_name.lower().replace("-", "_")
    registry = {
        "resnet18": ModelSpec("resnet18", 512, _build_resnet18, models.ResNet18_Weights.DEFAULT.url),
        "mobilenet_v3_small": ModelSpec(
            "mobilenet_v3_small",
            576,
            _build_mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.DEFAULT.url,
        ),
        "efficientnet_b0": ModelSpec(
            "efficientnet_b0",
            1280,
            _build_efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT.url,
        ),
    }
    if normalized not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")
    return registry[normalized]


def pretrained_weight_filename(model_name: str) -> str:
    spec = get_model_spec(model_name)
    return PathLikeUrl(spec.weights_url).filename


def _build_resnet18(use_pretrained_weights: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if use_pretrained_weights else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def _build_mobilenet_v3_small(use_pretrained_weights: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained_weights else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = nn.Identity()
    return model


@dataclass(frozen=True)
class PathLikeUrl:
    url: str

    @property
    def filename(self) -> str:
        parsed = urlparse(self.url)
        return parsed.path.rsplit("/", 1)[-1]


def _build_efficientnet_b0(use_pretrained_weights: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained_weights else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    return model
