"""Optional dominant-weave orientation normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from backend.app.ai.preprocessing.config import PreprocessConfig


@dataclass(frozen=True)
class OrientationMetadata:
    estimated_angle_degrees: float
    correction_degrees: float
    confidence: float
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def estimate_dominant_angle(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    if float(magnitude.sum()) <= 1e-6:
        return 0.0, 0.0
    # Gradient direction is perpendicular to the weave edge. Fold the angle
    # into [-45, 45) because woven structure is orthogonal.
    angles = np.rad2deg(np.arctan2(gy, gx)) + 90.0
    folded = ((angles + 45.0) % 90.0) - 45.0
    bins = np.linspace(-45.0, 45.0, 181)
    histogram, edges = np.histogram(folded, bins=bins, weights=magnitude)
    peak = int(np.argmax(histogram))
    angle = float((edges[peak] + edges[peak + 1]) / 2.0)
    confidence = float(histogram[peak] / max(float(histogram.sum()), 1e-6))
    # A single half-degree bin understates reliable clustered orientations.
    neighborhood = histogram[max(0, peak - 6) : min(len(histogram), peak + 7)]
    confidence = min(1.0, float(neighborhood.sum() / max(float(histogram.sum()), 1e-6)) * 3.0)
    return angle, confidence


def normalize_orientation(
    image: np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray, OrientationMetadata]:
    angle, confidence = estimate_dominant_angle(image)
    correction = config.expected_orientation_degrees - angle
    if not config.apply_orientation_normalization:
        return image.copy(), OrientationMetadata(angle, correction, confidence, False, "disabled")
    if confidence < config.orientation_confidence_threshold:
        return image.copy(), OrientationMetadata(angle, correction, confidence, False, "low_confidence")
    if abs(correction) > config.maximum_rotation_degrees:
        return image.copy(), OrientationMetadata(angle, correction, confidence, False, "unsafe_rotation")
    if math.isclose(correction, 0.0, abs_tol=0.1):
        return image.copy(), OrientationMetadata(angle, correction, confidence, False, "already_aligned")
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), correction, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated, OrientationMetadata(angle, correction, confidence, True, "applied")
