"""Auxiliary frequency-domain patch features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FFTFeatures:
    dominant_horizontal_frequency: float
    dominant_vertical_frequency: float
    peak_amplitude: float
    spectral_energy_ratio: float
    spectral_distance_from_normal: float | None
    horizontal_energy: float
    vertical_energy: float
    diagonal_energy: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def calculate_fft_features(
    patch: np.ndarray,
    normal_reference_spectrum: np.ndarray | None = None,
) -> FFTFeatures:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= float(gray.mean())
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray * window))).astype(np.float32)
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    spectrum[max(0, center_y - 1) : center_y + 2, max(0, center_x - 1) : center_x + 2] = 0
    horizontal_profile = spectrum.sum(axis=0)
    vertical_profile = spectrum.sum(axis=1)
    horizontal_index = int(np.argmax(horizontal_profile))
    vertical_index = int(np.argmax(vertical_profile))
    total_energy = float(np.square(spectrum).sum())
    peak_amplitude = float(spectrum.max(initial=0.0))
    peak_mask = spectrum >= np.percentile(spectrum, 95)
    spectral_energy_ratio = (
        float(np.square(spectrum[peak_mask]).sum() / total_energy) if total_energy > 0 else 0.0
    )
    yy, xx = np.ogrid[:height, :width]
    horizontal_band = np.broadcast_to(
        np.abs(yy - center_y) <= max(1, height // 16),
        spectrum.shape,
    )
    vertical_band = np.broadcast_to(
        np.abs(xx - center_x) <= max(1, width // 16),
        spectrum.shape,
    )
    horizontal_energy = _energy_ratio(spectrum, horizontal_band, total_energy)
    vertical_energy = _energy_ratio(spectrum, vertical_band, total_energy)
    diagonal_mask = ~horizontal_band & ~vertical_band
    diagonal_energy = _energy_ratio(spectrum, diagonal_mask, total_energy)
    distance = None
    if normal_reference_spectrum is not None:
        reference = np.asarray(normal_reference_spectrum, dtype=np.float32)
        if reference.shape != spectrum.shape:
            raise ValueError("Normal reference spectrum shape must match the patch spectrum.")
        a = spectrum / max(float(np.linalg.norm(spectrum)), 1e-12)
        b = reference / max(float(np.linalg.norm(reference)), 1e-12)
        distance = float(np.linalg.norm(a - b))
    return FFTFeatures(
        dominant_horizontal_frequency=abs(horizontal_index - center_x) / max(width, 1),
        dominant_vertical_frequency=abs(vertical_index - center_y) / max(height, 1),
        peak_amplitude=peak_amplitude,
        spectral_energy_ratio=spectral_energy_ratio,
        spectral_distance_from_normal=distance,
        horizontal_energy=horizontal_energy,
        vertical_energy=vertical_energy,
        diagonal_energy=diagonal_energy,
    )


def _energy_ratio(spectrum: np.ndarray, mask: np.ndarray, total_energy: float) -> float:
    return float(np.square(spectrum[mask]).sum() / total_energy) if total_energy > 0 else 0.0
