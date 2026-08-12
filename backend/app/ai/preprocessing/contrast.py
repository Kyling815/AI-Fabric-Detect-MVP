"""Mild CLAHE applied only to Lab luminance."""

import cv2
import numpy as np


def apply_luminance_clahe(
    image: np.ndarray,
    clip_limit: float = 1.5,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return cv2.cvtColor(cv2.merge((clahe.apply(luminance), channel_a, channel_b)), cv2.COLOR_LAB2BGR)
