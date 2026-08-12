"""Lab luminance illumination-field correction."""

from __future__ import annotations

import cv2
import numpy as np


def correct_illumination(image: np.ndarray, sigma_ratio: float = 0.08) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Illumination correction expects a uint8 BGR image.")
    height, width = image.shape[:2]
    sigma = max(1.0, float(min(height, width)) * sigma_ratio)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    background = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    mean_background = max(float(background.mean()), 1.0)
    corrected = luminance / np.maximum(background, 1.0) * mean_background
    result_lab = lab.copy()
    target_luminance = np.clip(corrected, 0, 255).astype(np.uint8)
    result_lab[:, :, 0] = target_luminance
    output = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    # Some corrected Lab colors can fall outside the BGR gamut. In those
    # pixels only, conservatively reduce the luminance correction until the
    # original chromatic channels survive the round trip.
    original_luminance = lab[:, :, 0]
    for _ in range(7):
        roundtrip = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        chroma_error = np.max(
            np.abs(roundtrip[:, :, 1:].astype(np.int16) - lab[:, :, 1:].astype(np.int16)),
            axis=2,
        )
        unsafe = chroma_error > 2
        if not bool(np.any(unsafe)):
            break
        target_luminance[unsafe] = (
            target_luminance[unsafe].astype(np.uint16) + original_luminance[unsafe].astype(np.uint16)
        ) // 2
        result_lab[:, :, 0] = target_luminance
        output = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return output
