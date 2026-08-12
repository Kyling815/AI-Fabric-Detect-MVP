"""Mild edge-preserving denoising."""

import cv2
import numpy as np


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 20.0,
    sigma_space: float = 5.0,
) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=cv2.BORDER_REFLECT_101,
    )
