"""Row/column ACF period observations. Never changes CNN input or QC decisions."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.app.ai.preprocess import inspect_image_file
from backend.app.core.pattern import PatternEstimatorConfig


@dataclass(frozen=True)
class PatternPeriodResult:
    period_x_px: float | None = None
    period_y_px: float | None = None
    confidence: float = 0.0
    is_periodic: bool = False
    confidence_x: float = 0.0
    confidence_y: float = 0.0
    reason: str | None = None


class PatternPeriodEstimator:
    def estimate_path(self, path: Path, config: PatternEstimatorConfig | None = None) -> PatternPeriodResult:
        try:
            inspect_image_file(path)
            with Image.open(path) as source:
                gray = np.asarray(ImageOps.exif_transpose(source).convert("L"))
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            return PatternPeriodResult(reason="image_read_error")
        return self.estimate(gray, config)

    def estimate(self, image: np.ndarray, config: PatternEstimatorConfig | None = None) -> PatternPeriodResult:
        cfg = config or PatternEstimatorConfig()
        values = np.asarray(image)
        if (values.ndim not in {2, 3} or values.size == 0
                or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)):
            return PatternPeriodResult(reason="unsupported_input")
        if values.ndim == 3 and values.shape[2] not in {3, 4}:
            return PatternPeriodResult(reason="unsupported_input")
        if not np.isfinite(values).all():
            return PatternPeriodResult(reason="unsupported_input")
        gray = values.astype(np.float64)
        if gray.ndim == 3:
            gray = gray[:, :, :3] @ np.array([0.299, 0.587, 0.114])
        # Float images in [0, 1] use the same contrast scale as uint8 captures.
        if np.issubdtype(values.dtype, np.floating) and gray.min() >= 0 and gray.max() <= 1:
            gray = gray * 255.0
        source_height, source_width = gray.shape
        scale = min(1.0, cfg.max_analysis_size / max(source_height, source_width))
        if scale < 1:
            gray = cv2.resize(gray, (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
                              interpolation=cv2.INTER_AREA)
        x_scale = source_width / gray.shape[1]
        y_scale = source_height / gray.shape[0]
        period_x, confidence_x = _axis_period(gray, axis=0, source_scale=x_scale, config=cfg)
        period_y, confidence_y = _axis_period(gray, axis=1, source_scale=y_scale, config=cfg)
        confidence = max(confidence_x, confidence_y)
        periodic = period_x is not None or period_y is not None
        reason = None if periodic else "low_periodicity_confidence"
        if periodic and (period_x is None or period_y is None):
            reason = "one_axis_only"
        return PatternPeriodResult(period_x, period_y, confidence, periodic, confidence_x, confidence_y, reason)


def _axis_period(gray: np.ndarray, *, axis: int, source_scale: float,
                 config: PatternEstimatorConfig) -> tuple[float | None, float]:
    length = gray.shape[1 - axis]
    if length < config.min_profile_length:
        return None, 0.0
    # The full average can cancel a balanced checker. Narrow bands keep that
    # phase information; agreement across bands avoids trusting one noisy row.
    profiles = [gray.mean(axis=axis)]
    perpendicular = gray.shape[axis]
    half_band = config.band_width // 2
    for center in np.unique(np.linspace(0, perpendicular - 1, min(config.band_count, perpendicular)).astype(int)):
        start, end = max(0, center - half_band), min(perpendicular, center + half_band + 1)
        band = gray[start:end, :] if axis == 0 else gray[:, start:end]
        profiles.append(band.mean(axis=axis))
    usable = 0
    candidates = []
    for profile in profiles:
        result = _profile_period(profile, source_scale, config)
        if result is None:
            continue
        usable += 1
        period, score = result
        if period is not None:
            candidates.append((period, score))
    if not candidates or not usable:
        return None, 0.0
    groups = []
    for period, _ in candidates:
        members = [(p, score) for p, score in candidates if abs(p - period) <= config.period_tolerance * period]
        support = len(members) / usable
        confidence = float(np.clip(np.median([score for _, score in members]) * support, 0, 1))
        groups.append((confidence, float(np.median([p for p, _ in members]))))
    confidence, period = max(groups, key=lambda value: (value[0], -value[1]))
    return (period if confidence >= config.confidence_threshold else None), confidence


def _profile_period(profile: np.ndarray, source_scale: float,
                    config: PatternEstimatorConfig) -> tuple[float | None, float] | None:
    count = profile.size
    coordinates = np.arange(count, dtype=np.float64)
    centered_coordinates = coordinates - coordinates.mean()
    signal = profile - profile.mean()
    # Remove a linear lighting slope before ACF; this does not transform input.
    signal = signal - centered_coordinates * np.dot(signal, centered_coordinates) / np.dot(centered_coordinates, centered_coordinates)
    if float(signal.std()) < config.min_profile_std:
        return None
    fft_size = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(signal, n=fft_size)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)[:count]
    acf = covariance / (count - np.arange(count))
    acf = acf / acf[0]
    min_lag = max(2, int(np.ceil(config.min_period_px / source_scale)))
    max_lag = min(count // config.min_cycles, int(np.floor(config.max_period_px / source_scale)), count - 2)
    if max_lag < min_lag:
        return None, 0.0
    peaks = np.flatnonzero((acf[1:-1] >= acf[:-2]) & (acf[1:-1] > acf[2:])) + 1
    peaks = peaks[(peaks >= min_lag) & (peaks <= max_lag) & (acf[peaks] >= config.min_peak_correlation)]
    for peak in peaks:
        tolerance = max(1, int(round(peak * config.period_tolerance)))
        expected = 2 * peak
        start, end = max(1, expected - tolerance), min(count - 1, expected + tolerance + 1)
        second = start + int(np.argmax(acf[start:end]))
        if not (acf[second] >= acf[second - 1] and acf[second] > acf[second + 1]):
            continue
        correlation = min(float(acf[peak]), float(acf[second]))
        if correlation < config.min_peak_correlation:
            continue
        spacing_score = max(0.0, 1.0 - abs(second - expected) / max(1, tolerance + 1))
        score = float(np.clip(correlation * spacing_score, 0, 1))
        return float(peak * source_scale), score
    return None, 0.0
