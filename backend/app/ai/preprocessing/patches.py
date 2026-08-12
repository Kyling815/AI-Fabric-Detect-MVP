"""Deterministic multi-scale overlapping patch extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from backend.app.ai.preprocessing.resize import reflection_pad


@dataclass(frozen=True)
class PatchMetadata:
    patch_index: int
    source_image: str
    x1: int
    y1: int
    x2: int
    y2: int
    patch_size: int
    scale: float
    fabric_type: str | None
    split: str
    padding_left: int = 0
    padding_top: int = 0
    padding_right: int = 0
    padding_bottom: int = 0
    fabric_coverage: float = 1.0
    patch_path: str | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedPatch:
    image: np.ndarray
    metadata: PatchMetadata


def deterministic_positions(length: int, patch_size: int, overlap_ratio: float) -> tuple[int, ...]:
    if length <= 0 or patch_size <= 0:
        raise ValueError("Length and patch size must be positive.")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("Patch overlap ratio must be in [0, 1).")
    if length <= patch_size:
        return (0,)
    stride = max(1, round(patch_size * (1.0 - overlap_ratio)))
    positions = list(range(0, length - patch_size + 1, stride))
    edge = length - patch_size
    if positions[-1] != edge:
        positions.append(edge)
    return tuple(positions)


def extract_overlapping_patches(
    image: np.ndarray,
    patch_sizes: tuple[int, ...] = (224,),
    overlap_ratio: float = 0.50,
    source_image: str | Path = "",
    fabric_type: str | None = None,
    split: str = "unspecified",
    source_scale: float = 1.0,
    fabric_mask: np.ndarray | None = None,
    minimum_fabric_coverage: float = 0.0,
) -> list[ExtractedPatch]:
    if not patch_sizes or any(size <= 0 for size in patch_sizes):
        raise ValueError("patch_sizes must contain positive integers.")
    if not 0.0 <= minimum_fabric_coverage <= 1.0:
        raise ValueError("minimum_fabric_coverage must be in [0, 1].")
    height, width = image.shape[:2]
    patches: list[ExtractedPatch] = []
    index = 0
    for patch_size in patch_sizes:
        pad_right = max(0, patch_size - width)
        pad_bottom = max(0, patch_size - height)
        padded = reflection_pad(image, 0, pad_bottom, 0, pad_right)
        padded_mask = (
            reflection_pad(fabric_mask, 0, pad_bottom, 0, pad_right)
            if fabric_mask is not None
            else None
        )
        for y in deterministic_positions(padded.shape[0], patch_size, overlap_ratio):
            for x in deterministic_positions(padded.shape[1], patch_size, overlap_ratio):
                patch = padded[y : y + patch_size, x : x + patch_size].copy()
                valid_x2, valid_y2 = min(width, x + patch_size), min(height, y + patch_size)
                coverage = (
                    float(np.mean(padded_mask[y : y + patch_size, x : x + patch_size] > 0))
                    if padded_mask is not None
                    else 1.0
                )
                if coverage < minimum_fabric_coverage:
                    continue
                patches.append(
                    ExtractedPatch(
                        image=patch,
                        metadata=PatchMetadata(
                            patch_index=index,
                            source_image=str(source_image),
                            x1=x,
                            y1=y,
                            x2=valid_x2,
                            y2=valid_y2,
                            patch_size=patch_size,
                            scale=source_scale,
                            fabric_type=fabric_type,
                            split=split,
                            padding_right=max(0, x + patch_size - width),
                            padding_bottom=max(0, y + patch_size - height),
                            fabric_coverage=coverage,
                        ),
                    )
                )
                index += 1
    return patches
