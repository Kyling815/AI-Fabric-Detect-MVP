"""Conservative fabric region-of-interest extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import cv2
import numpy as np

from backend.app.ai.preprocessing.config import PreprocessConfig


@dataclass(frozen=True)
class RoiMetadata:
    mode: str
    applied: bool
    x1: int
    y1: int
    x2: int
    y2: int
    retained_area_ratio: float
    confidence: float
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        return asdict(self)


def extract_roi(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, RoiMetadata]:
    height, width = image.shape[:2]
    if config.roi_mode == "none":
        return image.copy(), _full_metadata(width, height, "none")
    if config.roi_mode == "fixed":
        box = resolve_fixed_roi(config.fixed_roi or {}, width, height)
        if box is None:
            return image.copy(), _full_metadata(width, height, "fixed", "invalid_or_empty_fixed_roi")
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2].copy(), RoiMetadata(
            mode="fixed",
            applied=True,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            retained_area_ratio=((x2 - x1) * (y2 - y1)) / (width * height),
            confidence=1.0,
        )
    return _automatic_roi(image, config)


def resolve_fixed_roi(
    values: Mapping[str, float | int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    ratio_keys = {"x1_ratio", "y1_ratio", "x2_ratio", "y2_ratio"}
    pixel_keys = {"x1", "y1", "x2", "y2"}
    if ratio_keys.issubset(values):
        raw = (
            round(float(values["x1_ratio"]) * width),
            round(float(values["y1_ratio"]) * height),
            round(float(values["x2_ratio"]) * width),
            round(float(values["y2_ratio"]) * height),
        )
    elif pixel_keys.issubset(values):
        raw = tuple(round(float(values[key])) for key in ("x1", "y1", "x2", "y2"))
    else:
        return None
    x1, y1, x2, y2 = raw
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _automatic_roi(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, RoiMetadata]:
    height, width = image.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    # The border is a conservative background estimate. Retain pixels that
    # differ from it in Lab; this works for controlled tabletop captures while
    # falling back safely when fabric cannot be isolated.
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
    center = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(lab.astype(np.float32) - center, axis=2)
    threshold = max(8.0, float(np.percentile(distance, 65)))
    mask = (distance >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return image.copy(), _full_metadata(width, height, "automatic", "no_dominant_region")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[component]
    retained_ratio = float(area / (small.shape[0] * small.shape[1]))
    confidence = min(1.0, retained_ratio / max(config.automatic_roi_min_area_ratio, 1e-6))
    if retained_ratio < config.automatic_roi_min_area_ratio:
        return image.copy(), _full_metadata(
            width, height, "automatic", "retained_area_below_threshold", confidence
        )
    margin_x = round(config.automatic_roi_margin_ratio * small.shape[1])
    margin_y = round(config.automatic_roi_margin_ratio * small.shape[0])
    sx1, sy1 = max(0, x - margin_x), max(0, y - margin_y)
    sx2 = min(small.shape[1], x + w + margin_x)
    sy2 = min(small.shape[0], y + h + margin_y)
    x1, y1 = round(sx1 / scale), round(sy1 / scale)
    x2, y2 = min(width, round(sx2 / scale)), min(height, round(sy2 / scale))
    if x2 <= x1 or y2 <= y1:
        return image.copy(), _full_metadata(width, height, "automatic", "empty_automatic_roi")
    crop = image[y1:y2, x1:x2]
    return crop.copy(), RoiMetadata(
        mode="automatic",
        applied=True,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        retained_area_ratio=((x2 - x1) * (y2 - y1)) / (width * height),
        confidence=confidence,
    )


def _full_metadata(
    width: int,
    height: int,
    mode: str,
    fallback_reason: str | None = None,
    confidence: float = 1.0,
) -> RoiMetadata:
    return RoiMetadata(
        mode=mode,
        applied=False,
        x1=0,
        y1=0,
        x2=width,
        y2=height,
        retained_area_ratio=1.0,
        confidence=confidence,
        fallback_reason=fallback_reason,
    )
