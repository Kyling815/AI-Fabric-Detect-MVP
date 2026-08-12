"""Aspect-ratio resize, safe padding, and coordinate mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeMetadata:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    output_width: int
    output_height: int
    scale: float
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def resize_preserving_aspect_ratio(
    image: np.ndarray,
    target_max_size: int,
    allow_upscale: bool = False,
) -> tuple[np.ndarray, ResizeMetadata]:
    if target_max_size <= 0:
        raise ValueError("target_max_size must be positive.")
    height, width = image.shape[:2]
    scale = target_max_size / max(width, height)
    if not allow_upscale:
        scale = min(1.0, scale)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = (
        image.copy()
        if (resized_width, resized_height) == (width, height)
        else cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    )
    return resized, ResizeMetadata(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        output_width=resized_width,
        output_height=resized_height,
        scale=scale,
    )


def reflection_pad(
    image: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> np.ndarray:
    if min(top, bottom, left, right) < 0:
        raise ValueError("Padding values cannot be negative.")
    if not any((top, bottom, left, right)):
        return image.copy()
    # BORDER_REFLECT_101 cannot pad beyond a dimension in one call on every
    # OpenCV build. Repeated safe chunks retain deterministic reflection.
    result = image
    remaining = [top, bottom, left, right]
    while any(remaining):
        height, width = result.shape[:2]
        chunk = [
            min(remaining[0], max(1, height - 1)),
            min(remaining[1], max(1, height - 1)),
            min(remaining[2], max(1, width - 1)),
            min(remaining[3], max(1, width - 1)),
        ]
        border_type = cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
        result = cv2.copyMakeBorder(result, *chunk, borderType=border_type)
        remaining = [value - used for value, used in zip(remaining, chunk)]
    return result


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    reflection: bool = True,
) -> tuple[np.ndarray, ResizeMetadata]:
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Letterbox dimensions must be positive.")
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    horizontal = target_width - resized_width
    vertical = target_height - resized_height
    left, right = horizontal // 2, horizontal - horizontal // 2
    top, bottom = vertical // 2, vertical - vertical // 2
    if reflection:
        output = reflection_pad(resized, top, bottom, left, right)
    else:
        output = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return output, ResizeMetadata(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        output_width=target_width,
        output_height=target_height,
        scale=scale,
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
    )


def map_box_to_source(
    box: tuple[int, int, int, int],
    metadata: ResizeMetadata,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    source_x1 = math.floor((x1 - metadata.pad_left) / metadata.scale)
    source_y1 = math.floor((y1 - metadata.pad_top) / metadata.scale)
    source_x2 = math.ceil((x2 - metadata.pad_left) / metadata.scale)
    source_y2 = math.ceil((y2 - metadata.pad_top) / metadata.scale)
    return (
        max(0, min(metadata.source_width, source_x1)),
        max(0, min(metadata.source_height, source_y1)),
        max(0, min(metadata.source_width, source_x2)),
        max(0, min(metadata.source_height, source_y2)),
    )


def map_box_from_source(
    box: tuple[int, int, int, int],
    metadata: ResizeMetadata,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        round(x1 * metadata.scale) + metadata.pad_left,
        round(y1 * metadata.scale) + metadata.pad_top,
        round(x2 * metadata.scale) + metadata.pad_left,
        round(y2 * metadata.scale) + metadata.pad_top,
    )
