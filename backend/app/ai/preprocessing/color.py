"""Fabric-family-specific, conservative Lab color normalization."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_fabric_color(
    image: np.ndarray,
    reference_mean: tuple[float, float, float],
    reference_std: tuple[float, float, float],
    luminance_strength: float = 1.0,
    chromatic_strength: float = 0.25,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_mean = lab.reshape(-1, 3).mean(axis=0)
    current_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1.0)
    normalized = (lab - current_mean) / current_std * np.asarray(reference_std) + np.asarray(reference_mean)
    strengths = np.asarray((luminance_strength, chromatic_strength, chromatic_strength), dtype=np.float32)
    blended = lab * (1.0 - strengths) + normalized * strengths
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
