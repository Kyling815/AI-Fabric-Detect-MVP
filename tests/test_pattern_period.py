from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from backend.app.ai.pattern_period import PatternPeriodEstimator
from backend.app.core.pattern import PatternEstimatorConfig
from tests.pattern_period_fixtures import pattern, png_bytes


@pytest.mark.parametrize("kind, expected_x, expected_y", [
    ("checker", 16, 24), ("stripes", 16, None), ("horizontal_stripes", None, 24),
    ("dots", 20, 24), ("stretch", 24, 30), ("lighting", 16, 24),
])
def test_synthetic_periods_in_original_pixel_units(kind, expected_x, expected_y):
    result = PatternPeriodEstimator().estimate(pattern(kind))
    assert result.is_periodic
    assert result.confidence >= 0.65
    for actual, expected in ((result.period_x_px, expected_x), (result.period_y_px, expected_y)):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, abs=1)


@pytest.mark.parametrize("kind", ["uniform", "noise", "rotate"])
def test_unreliable_synthetic_inputs_fall_back_without_forced_period(kind):
    result = PatternPeriodEstimator().estimate(pattern(kind))
    assert not result.is_periodic
    assert result.period_x_px is None and result.period_y_px is None
    assert 0 <= result.confidence < 0.65
    assert result.reason == "low_periodicity_confidence"


def test_balanced_checker_has_constant_global_profiles_but_bands_recover_period():
    image = pattern("checker")
    assert image.mean(axis=0).std() == 0
    assert image.mean(axis=1).std() == 0
    result = PatternPeriodEstimator().estimate(image)
    assert (result.period_x_px, result.period_y_px) == (16, 24)


@pytest.mark.parametrize("representation", ["rgb", "rgba", "float_unit", "float_byte", "noncontiguous"])
def test_representations_are_consistent_and_input_is_not_mutated(representation):
    image = pattern("checker")
    if representation in {"rgb", "rgba"}:
        image = np.repeat(image[:, :, None], 3 if representation == "rgb" else 4, axis=2)
    elif representation == "float_unit":
        image = image.astype(np.float32) / 255
    elif representation == "float_byte":
        image = image.astype(np.float32)
    else:
        image = image[:, ::-1]
    before = image.copy()
    image.flags.writeable = False
    estimator = PatternPeriodEstimator()
    result = estimator.estimate(image)
    assert result == estimator.estimate(image)
    assert (result.period_x_px, result.period_y_px) == (16, 24)
    np.testing.assert_array_equal(image, before)
    with pytest.raises(FrozenInstanceError):
        result.confidence = 1


@pytest.mark.parametrize("image", [np.empty((0, 3)), np.zeros(5), np.zeros((5, 5, 2)),
                                  np.array([["a"]]), np.full((64, 64), np.nan),
                                  np.full((64, 64), np.inf), np.zeros((64, 64), dtype=complex)])
def test_invalid_arrays_return_diagnostic_fallback(image):
    result = PatternPeriodEstimator().estimate(image)
    assert not result.is_periodic and result.reason == "unsupported_input"


def test_small_and_insufficient_cycles_do_not_force_period():
    estimator = PatternPeriodEstimator()
    assert not estimator.estimate(pattern("checker")[:32, :32]).is_periodic
    x = np.arange(128)
    image = np.tile(120 + 90 * np.sin(2 * np.pi * x / 64), (128, 1))
    assert not estimator.estimate(image).is_periodic


def test_linear_illumination_only_is_not_a_repeat():
    y, x = np.indices((288, 320))
    assert not PatternPeriodEstimator().estimate(50 + x * 0.2 + y * 0.1).is_periodic


def test_downsampling_is_diagnostic_and_periods_map_back_to_source_pixels():
    image = cv2.resize(pattern("checker"), (1280, 1152), interpolation=cv2.INTER_NEAREST)
    result = PatternPeriodEstimator().estimate(image, PatternEstimatorConfig(max_analysis_size=320))
    assert (result.period_x_px, result.period_y_px) == (64, 96)
    assert image.shape == (1152, 1280)


def test_path_decode_and_exif_orientation(tmp_path):
    path = tmp_path / "capture.png"
    path.write_bytes(png_bytes())
    result = PatternPeriodEstimator().estimate_path(path)
    assert (result.period_x_px, result.period_y_px) == (16, 24)
    exif = Image.Exif()
    exif[274] = 6
    rotated = tmp_path / "oriented.png"
    Image.fromarray(pattern("checker")).save(rotated, exif=exif)
    result = PatternPeriodEstimator().estimate_path(rotated)
    assert (result.period_x_px, result.period_y_px) == (24, 16)


@pytest.mark.parametrize("kind", ["missing", "corrupt", "bad_checksum"])
def test_path_failures_are_nonfatal(tmp_path, kind):
    path = tmp_path / "capture.png"
    if kind == "corrupt":
        path.write_bytes(b"invalid")
    elif kind == "bad_checksum":
        content = bytearray(png_bytes())
        content[29] ^= 1  # IHDR CRC, decoding must verify integrity.
        path.write_bytes(content)
    result = PatternPeriodEstimator().estimate_path(path)
    assert result.reason == "image_read_error" and not result.is_periodic


@pytest.mark.parametrize("values", [
    {"mode": "adaptive"}, {"min_period_px": 1}, {"max_period_px": 3, "min_period_px": 4},
    {"min_cycles": 2}, {"min_profile_length": 0}, {"min_profile_std": 0},
    {"min_peak_correlation": 1.1}, {"confidence_threshold": float("nan")},
    {"period_tolerance": 0}, {"band_count": 0}, {"band_width": 2},
    {"max_analysis_size": 32}, {"patch_size": 48},
])
def test_invalid_or_adaptive_configuration_is_rejected(values):
    with pytest.raises(ValidationError):
        PatternEstimatorConfig.model_validate(values)


def test_example_config_is_observe_only_and_frozen():
    path = Path(__file__).resolve().parents[1] / "config" / "pattern_observation_example.json"
    config = PatternEstimatorConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert config.mode == "observe_only"
    with pytest.raises(ValidationError):
        config.mode = "adaptive"


@pytest.mark.parametrize("seed", range(10))
def test_noise_seeds_do_not_report_periods(seed):
    image = np.random.default_rng(seed).integers(0, 256, size=(288, 320), dtype=np.uint8)
    assert not PatternPeriodEstimator().estimate(image).is_periodic


@pytest.mark.parametrize("present", [True, False])
def test_read_only_cli_reports_period_or_decode_fallback(tmp_path, present):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "capture.png"
    if present:
        path.write_bytes(png_bytes())
    command = [sys.executable, str(root / "scripts" / "observe_pattern_period.py"), str(path),
               "--config", str(root / "config" / "pattern_observation_example.json")]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    value = json.loads(completed.stdout)
    assert value["mode"] == "observe_only" and value["is_periodic"] == present
    assert value["period_x_px"] == (16 if present else None)
    assert set(tmp_path.iterdir()) == ({path} if present else set())
