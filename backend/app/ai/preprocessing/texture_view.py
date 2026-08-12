"""Optional three-channel texture representation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def generate_texture_view(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    source = gray.astype(np.float32) / 255.0
    responses: list[np.ndarray] = []
    for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.5, 0.0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(source, cv2.CV_32F, kernel)))
    gabor = _to_uint8(np.maximum.reduce(responses))
    blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=1.0, sigmaY=1.0)
    log_response = _to_uint8(np.abs(cv2.Laplacian(blurred, cv2.CV_32F)))
    return cv2.merge((normalized.astype(np.uint8), gabor, log_response))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    minimum, maximum = float(array.min()), float(array.max())
    if maximum - minimum <= 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)
