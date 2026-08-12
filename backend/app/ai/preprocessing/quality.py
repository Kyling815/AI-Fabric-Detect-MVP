"""Image decoding and quality validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.app.ai.preprocessing.config import PreprocessConfig


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class QualityMetrics:
    width: int = 0
    height: int = 0
    laplacian_variance: float = 0.0
    dark_ratio: float = 0.0
    bright_ratio: float = 0.0
    decode_success: bool = False
    channel_count: int = 0

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class QualityValidation:
    valid: bool
    reasons: tuple[str, ...]


def load_image(path: Path) -> tuple[np.ndarray | None, str | None]:
    """Decode an image as uint8 BGR without failing the surrounding batch."""

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None, "unsupported_format"
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None, "cannot_decode"
    if decoded is None:
        return None, "cannot_decode"
    if decoded.ndim == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
    elif decoded.ndim != 3 or decoded.shape[2] != 3:
        return None, "unsupported_channel_count"
    if decoded.dtype != np.uint8:
        decoded = cv2.normalize(decoded, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return decoded, None


def calculate_quality_metrics(
    image: np.ndarray | None,
    config: PreprocessConfig | None = None,
) -> QualityMetrics:
    if image is None:
        return QualityMetrics()
    cfg = config or PreprocessConfig()
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    if image.ndim == 2:
        gray = image
    elif channels in {3, 4}:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        return QualityMetrics(width=width, height=height, decode_success=True, channel_count=channels)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return QualityMetrics(
        width=width,
        height=height,
        laplacian_variance=laplacian_variance,
        dark_ratio=float(np.mean(gray <= cfg.dark_pixel_threshold)),
        bright_ratio=float(np.mean(gray >= cfg.bright_pixel_threshold)),
        decode_success=True,
        channel_count=channels,
    )


def validate_quality(metrics: QualityMetrics, config: PreprocessConfig) -> QualityValidation:
    reasons: list[str] = []
    if not metrics.decode_success:
        reasons.append("cannot_decode")
    else:
        if metrics.channel_count not in {1, 3, 4}:
            reasons.append("unsupported_channel_count")
        if metrics.width < config.min_width:
            reasons.append("image_width_too_small")
        if metrics.height < config.min_height:
            reasons.append("image_height_too_small")
        if metrics.laplacian_variance < config.blur_threshold:
            reasons.append("image_too_blurry")
        if metrics.dark_ratio > config.max_dark_ratio:
            reasons.append("image_too_dark")
        if metrics.bright_ratio > config.max_bright_ratio:
            reasons.append("image_too_bright")
    return QualityValidation(valid=not reasons, reasons=tuple(reasons))
