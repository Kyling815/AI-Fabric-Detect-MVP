"""Optional mild unsharp masking."""

import cv2
import numpy as np


def unsharp_mask(image: np.ndarray, alpha: float = 0.25, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
