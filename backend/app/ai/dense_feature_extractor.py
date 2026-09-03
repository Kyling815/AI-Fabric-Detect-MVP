from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

from backend.app.ai.patching import PatchBox
from backend.app.ai.projection_head import load_projection_head


BASE_BACKBONE_VERSION = "shared_resnet18_imagenet1k_v1"
ADAPTED_BACKBONE_VERSION = "shared_resnet18_imagenet1k_dense_adapter_company_v1"
ADAPTED_BACKBONE_PREFIX = "shared_resnet18_imagenet1k_dense_adapter_company_v"
SUPPORTED_BACKBONE_VERSION = BASE_BACKBONE_VERSION
SUPPORTED_BACKBONE_VERSIONS = {BASE_BACKBONE_VERSION, ADAPTED_BACKBONE_VERSION}


def is_adapted_backbone_version(backbone_version: str) -> bool:
    if not backbone_version.startswith(ADAPTED_BACKBONE_PREFIX):
        return False
    suffix = backbone_version.removeprefix(ADAPTED_BACKBONE_PREFIX)
    return suffix.isdigit() and int(suffix) >= 1


def is_supported_backbone_version(backbone_version: str) -> bool:
    return backbone_version == BASE_BACKBONE_VERSION or is_adapted_backbone_version(
        backbone_version
    )


@dataclass(frozen=True)
class DenseFeatureExtractorInfo:
    backbone_version: str
    architecture: str
    device: str
    use_pretrained_weights: bool
    feature_dim: int
    feature_stride: int


@dataclass(frozen=True)
class DenseFeatureResult:
    embeddings: np.ndarray
    boxes: tuple[PatchBox, ...]
    grid_rows: int
    grid_cols: int
    processed_width: int
    processed_height: int
    original_width: int
    original_height: int
    scale: float


class SharedDenseResNet18:
    """One cached ImageNet ResNet18 producing PatchCore-style local features."""

    def __init__(
        self,
        *,
        backbone_version: str = SUPPORTED_BACKBONE_VERSION,
        device: str = "auto",
        use_pretrained_weights: bool = True,
        adapter_path: str = "",
    ) -> None:
        if not is_supported_backbone_version(backbone_version):
            raise ValueError(f"Unsupported backbone version: {backbone_version}")
        self.backbone_version = backbone_version
        self.device = _resolve_device(device)
        self.use_pretrained_weights = use_pretrained_weights
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained_weights else None
        self.model = models.resnet18(weights=weights).to(self.device).eval()
        self.adapter = None
        self.adapter_info = None
        if is_adapted_backbone_version(backbone_version):
            if not adapter_path:
                raise ValueError(f"{backbone_version} requires adapter_path")
            self.adapter, self.adapter_info = load_projection_head(Path(adapter_path), device=self.device)
            if self.adapter_info.input_dim != 384:
                raise ValueError(
                    f"Dense adapter expects {self.adapter_info.input_dim} features; runtime produces 384"
                )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @property
    def info(self) -> DenseFeatureExtractorInfo:
        return DenseFeatureExtractorInfo(
            backbone_version=self.backbone_version,
            architecture=(
                "resnet18_layer2_layer3_dense+shared_projection_adapter"
                if self.adapter is not None
                else "resnet18_layer2_layer3_dense"
            ),
            device=str(self.device),
            use_pretrained_weights=self.use_pretrained_weights,
            feature_dim=self.adapter_info.projection_dim if self.adapter_info is not None else 384,
            feature_stride=8,
        )

    def extract_path(self, path: Path, *, image_max_size: int) -> DenseFeatureResult:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.extract_image(image, image_max_size=image_max_size)

    def extract_image(self, image: Image.Image, *, image_max_size: int) -> DenseFeatureResult:
        if image_max_size <= 0:
            raise ValueError("image_max_size must be positive")
        original_width, original_height = image.size
        scale = min(1.0, image_max_size / max(original_width, original_height))
        processed_width = max(1, round(original_width * scale))
        processed_height = max(1, round(original_height * scale))
        resized = image.convert("RGB").resize(
            (processed_width, processed_height),
            Image.Resampling.BILINEAR,
        )
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self._forward_features(tensor)
        _, _, grid_rows, grid_cols = features.shape
        embeddings = (
            features.squeeze(0)
            .permute(1, 2, 0)
            .reshape(grid_rows * grid_cols, -1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        boxes = _feature_cells(processed_width, processed_height, grid_cols, grid_rows)
        return DenseFeatureResult(
            embeddings=embeddings,
            boxes=boxes,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            processed_width=processed_width,
            processed_height=processed_height,
            original_width=original_width,
            original_height=original_height,
            scale=scale,
        )

    def _forward_features(self, tensor: torch.Tensor) -> torch.Tensor:
        model = self.model
        value = model.conv1(tensor)
        value = model.bn1(value)
        value = model.relu(value)
        value = model.maxpool(value)
        value = model.layer1(value)
        layer2 = model.layer2(value)
        layer3 = model.layer3(layer2)

        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        layer2 = F.normalize(layer2, p=2, dim=1)
        layer3 = F.normalize(layer3, p=2, dim=1)
        dense = F.normalize(torch.cat([layer2, layer3], dim=1), p=2, dim=1)
        if self.adapter is None:
            return dense
        batch, channels, rows, cols = dense.shape
        flat = dense.permute(0, 2, 3, 1).reshape(batch * rows * cols, channels)
        projected = self.adapter.forward_features(flat)
        return projected.reshape(batch, rows, cols, -1).permute(0, 3, 1, 2)


@lru_cache(maxsize=2)
def get_shared_dense_extractor(
    backbone_version: str,
    device: str,
    use_pretrained_weights: bool = True,
    adapter_path: str = "",
) -> SharedDenseResNet18:
    return SharedDenseResNet18(
        backbone_version=backbone_version,
        device=device,
        use_pretrained_weights=use_pretrained_weights,
        adapter_path=adapter_path,
    )


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _feature_cells(width: int, height: int, cols: int, rows: int) -> tuple[PatchBox, ...]:
    boxes = []
    for row in range(rows):
        y1 = round(row * height / rows)
        y2 = round((row + 1) * height / rows)
        for col in range(cols):
            x1 = round(col * width / cols)
            x2 = round((col + 1) * width / cols)
            boxes.append(PatchBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)))
    return tuple(boxes)
