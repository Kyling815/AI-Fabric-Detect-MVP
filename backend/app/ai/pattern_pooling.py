"""Offline, gated pooling on native layer2/3 maps, not exhaustive image patches."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from backend.app.ai.dense_feature_extractor import DenseFeatureResult, SharedDenseResNet18, _feature_cells
from backend.app.ai.pattern_period import PatternPeriodEstimator, PatternPeriodResult
from backend.app.core.pattern import PatternEstimatorConfig
from backend.app.core.pattern_pooling import FeatureGeometry, PatternPoolingConfig


@dataclass(frozen=True)
class PoolingPlan:
    representation: str
    layer2_kernels: tuple[tuple[int, int], ...]  # height/y, width/x on native grid
    layer3_kernels: tuple[tuple[int, int], ...]
    fallback_reason: str | None


@dataclass(frozen=True)
class PatternFeatureResult:
    features: DenseFeatureResult
    geometry: FeatureGeometry
    pooling_plan: PoolingPlan
    observed_period: PatternPeriodResult | None


def _odd_kernel(cells: float, limit: int) -> int:
    maximum = limit if limit % 2 else limit - 1
    value = max(1, math.ceil(cells))
    value = value if value % 2 else value + 1
    return min(value, max(1, maximum))


def plan_pattern_pooling(config: PatternPoolingConfig, period: PatternPeriodResult | None, *,
                         resize_x: float, resize_y: float, layer2_shape: tuple[int, int],
                         layer3_shape: tuple[int, int]) -> PoolingPlan:
    if not config.enabled:
        return PoolingPlan("baseline", ((3, 3),), ((3, 3),), None)
    if (not math.isfinite(resize_x) or not math.isfinite(resize_y) or not 0 < resize_x <= 1
            or not 0 < resize_y <= 1 or any(type(v) is not int or v < 1 for v in (*layer2_shape, *layer3_shape))):
        raise ValueError("Positive native map shapes and actual resize factors in (0,1] required")
    reason = "fixed_multiscale_mode"
    accepted = False
    if config.mode == "period_aware":
        reason = "low_or_incomplete_period_confidence"
        if period is not None and period.is_periodic:
            numbers = (period.period_x_px, period.period_y_px, period.confidence, period.confidence_x, period.confidence_y)
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in numbers):
                accepted = (period.period_x_px > 0 and period.period_y_px > 0
                            and period.period_x_px <= layer2_shape[1] * 8 / resize_x
                            and period.period_y_px <= layer2_shape[0] * 8 / resize_y
                            and all(config.confidence_threshold <= value <= 1 for value in numbers[2:]))
    kernels = []
    for stride, shape in ((8, layer2_shape), (16, layer3_shape)):
        bounds = (min(config.max_kernel, shape[0]), min(config.max_kernel, shape[1]))
        if accepted:
            values = tuple((_odd_kernel(period.period_y_px * resize_y * scale / stride, bounds[0]),
                            _odd_kernel(period.period_x_px * resize_x * scale / stride, bounds[1]))
                           for scale in config.motif_scales)
        else:
            values = tuple((_odd_kernel(size, bounds[0]), _odd_kernel(size, bounds[1])) for size in config.fallback_kernels)
        kernels.append(values)
    return PoolingPlan("period_aware" if accepted else "fixed_multiscale", kernels[0], kernels[1], None if accepted else reason)


def pool_native_features(layer2: torch.Tensor, layer3: torch.Tensor, plan: PoolingPlan) -> torch.Tensor:
    if (layer2.ndim != 4 or layer3.ndim != 4 or layer2.shape[0] != layer3.shape[0]
            or min(*layer2.shape, *layer3.shape) < 1 or not torch.isfinite(layer2).all()
            or not torch.isfinite(layer3).all() or layer2.device != layer3.device or layer2.dtype != layer3.dtype
            or not layer2.is_floating_point() or not layer3.is_floating_point()):
        raise ValueError("Finite, compatible NCHW floating layer2/layer3 maps required")
    if not plan.layer2_kernels or len(plan.layer2_kernels) != len(plan.layer3_kernels):
        raise ValueError("Both native maps must have matching nonempty scale lists")
    merged = []
    for k2, k3 in zip(plan.layer2_kernels, plan.layer3_kernels):
        for kernel in (k2, k3):
            if len(kernel) != 2 or any(type(size) is not int or size < 1 or size % 2 != 1 for size in kernel):
                raise ValueError("Pooling kernels must be positive odd (height,width) pairs")
        two = F.avg_pool2d(layer2, k2, stride=1, padding=(k2[0] // 2, k2[1] // 2))
        three = F.avg_pool2d(layer3, k3, stride=1, padding=(k3[0] // 2, k3[1] // 2))
        three = F.interpolate(three, size=two.shape[-2:], mode="bilinear", align_corners=False)
        dense = F.normalize(torch.cat([F.normalize(two, p=2, dim=1), F.normalize(three, p=2, dim=1)], dim=1), p=2, dim=1)
        merged.append(dense)
    return F.normalize(torch.stack(merged).mean(dim=0), p=2, dim=1)


def _weights_digest(module) -> str | None:
    if module is None:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PatternAwareDenseExtractor:
    def __init__(self, baseline: SharedDenseResNet18, *, config: PatternPoolingConfig | None = None,
                 adapter_version: str | None = None, estimator: PatternPeriodEstimator | None = None,
                 estimator_config: PatternEstimatorConfig | None = None):
        self.baseline = baseline
        self.config = config or PatternPoolingConfig()
        self.adapter_version = adapter_version
        if (baseline.adapter is None) != (adapter_version is None):
            raise ValueError("Declare adapter version exactly when an adapter is present")
        self.estimator = estimator if estimator is not None else PatternPeriodEstimator()
        self.estimator_config = estimator_config or PatternEstimatorConfig()

    def _geometry(self, image_max_size: int) -> FeatureGeometry:
        info = self.baseline.info
        return FeatureGeometry(
            algorithm="pattern_pooling_mean_v1" if self.config.enabled else "dense_baseline_v1",
            backbone_version=info.backbone_version, architecture=info.architecture,
            model_weights_sha256=_weights_digest(self.baseline.model), adapter_weights_sha256=_weights_digest(self.baseline.adapter),
            adapter_version=self.adapter_version, use_pretrained_weights=info.use_pretrained_weights,
            feature_dim=info.feature_dim, image_max_size=image_max_size,
            pooling=self.config if self.config.enabled else None,
            period_estimator=self.estimator_config if self.config.enabled and self.config.mode == "period_aware" else None,
        )

    def extract_path(self, path: Path, *, image_max_size: int) -> PatternFeatureResult:
        if not self.config.enabled:
            features = self.baseline.extract_path(path, image_max_size=image_max_size)
            return PatternFeatureResult(features, self._geometry(image_max_size), PoolingPlan("baseline", ((3, 3),), ((3, 3),), None), None)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.extract_image(image, image_max_size=image_max_size)

    def extract_image(self, image: Image.Image, *, image_max_size: int) -> PatternFeatureResult:
        if not self.config.enabled:
            features = self.baseline.extract_image(image, image_max_size=image_max_size)
            return PatternFeatureResult(features, self._geometry(image_max_size), PoolingPlan("baseline", ((3, 3),), ((3, 3),), None), None)
        if type(image_max_size) is not int or image_max_size <= 0:
            raise ValueError("image_max_size must be a positive integer")
        if any(module.training for module in self.baseline.model.modules()) or (
                self.baseline.adapter is not None and any(module.training for module in self.baseline.adapter.modules())):
            raise ValueError("Research pooling requires immutable eval-mode model/adapter")
        width, height = image.size
        scale = min(1.0, image_max_size / max(width, height))
        processed_width, processed_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = image.convert("RGB").resize((processed_width, processed_height), Image.Resampling.BILINEAR)
        period = None
        if self.config.mode == "period_aware":
            try:
                period = self.estimator.estimate(np.asarray(image.convert("RGB")), self.estimator_config)
            except Exception:
                period = PatternPeriodResult(reason="estimator_error")
        tensor = self.baseline.transform(resized).unsqueeze(0).to(self.baseline.device)
        with torch.inference_mode():
            model = self.baseline.model
            value = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
            layer2 = model.layer2(model.layer1(value))
            layer3 = model.layer3(layer2)
            plan = plan_pattern_pooling(self.config, period, resize_x=processed_width / width, resize_y=processed_height / height,
                                        layer2_shape=tuple(layer2.shape[-2:]), layer3_shape=tuple(layer3.shape[-2:]))
            dense = pool_native_features(layer2, layer3, plan)
            batch, channels, rows, cols = dense.shape
            if self.baseline.adapter is not None:
                if channels != self.baseline.adapter_info.input_dim:
                    raise ValueError("Pooled channels do not match existing projection adapter")
                flat = dense.permute(0, 2, 3, 1).reshape(batch * rows * cols, channels)
                dense = self.baseline.adapter.forward_features(flat).reshape(batch, rows, cols, -1).permute(0, 3, 1, 2)
            embeddings = dense.squeeze(0).permute(1, 2, 0).reshape(rows * cols, -1).cpu().numpy().astype(np.float32)
        if embeddings.shape[1] != self.baseline.info.feature_dim or not np.isfinite(embeddings).all():
            raise ValueError("Experimental embedding shape/values do not match declared geometry")
        features = DenseFeatureResult(embeddings, _feature_cells(processed_width, processed_height, cols, rows), rows, cols,
                                      processed_width, processed_height, width, height, scale)
        return PatternFeatureResult(features, self._geometry(image_max_size), plan, period)
