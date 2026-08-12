import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.ai.preprocessing.config import (
    PreprocessConfig,
    config_for_ablation,
    load_registry_overrides,
)
from backend.app.ai.preprocessing.adapter import prepare_image
from backend.app.ai.preprocessing.contrast import apply_luminance_clahe
from backend.app.ai.preprocessing.denoise import bilateral_denoise
from backend.app.ai.preprocessing.fft_features import calculate_fft_features
from backend.app.ai.preprocessing.illumination import correct_illumination
from backend.app.ai.preprocessing.orientation import normalize_orientation
from backend.app.ai.preprocessing.patches import deterministic_positions, extract_overlapping_patches
from backend.app.ai.preprocessing.pipeline import preprocess_fabric_image, process_directory
from backend.app.ai.preprocessing.quality import calculate_quality_metrics, load_image, validate_quality
from backend.app.ai.preprocessing.resize import letterbox, map_box_to_source, reflection_pad, resize_preserving_aspect_ratio
from backend.app.ai.preprocessing.roi import extract_roi
from backend.app.ai.preprocessing.texture_view import generate_texture_view


def small_config(**overrides) -> PreprocessConfig:
    values = {
        "min_width": 32,
        "min_height": 32,
        "blur_threshold": 0.0,
        "max_dark_ratio": 1.0,
        "max_bright_ratio": 1.0,
    }
    values.update(overrides)
    return PreprocessConfig(**values)


def textured_image(width: int = 96, height: int = 64) -> np.ndarray:
    yy, xx = np.indices((height, width))
    gray = ((xx * 7 + yy * 3) % 180 + 35).astype(np.uint8)
    return cv2.merge((gray, np.roll(gray, 1, axis=0), np.roll(gray, 1, axis=1)))


def test_quality_metrics_and_rejection_reasons() -> None:
    image = np.zeros((24, 20, 3), dtype=np.uint8)
    config = PreprocessConfig(min_width=32, min_height=32, blur_threshold=1.0)
    metrics = calculate_quality_metrics(image, config)
    validation = validate_quality(metrics, config)
    assert metrics.decode_success
    assert metrics.channel_count == 3
    assert "image_width_too_small" in validation.reasons
    assert "image_height_too_small" in validation.reasons
    assert "image_too_blurry" in validation.reasons
    assert "image_too_dark" in validation.reasons


@pytest.mark.parametrize(
    "image",
    [
        np.full((64, 64, 3), 128, np.uint8),
        np.zeros((64, 64, 3), np.uint8),
        np.dstack([np.tile(np.arange(64, dtype=np.uint8), (64, 1))] * 3),
        textured_image(64, 64),
    ],
)
def test_illumination_correction_preserves_shape_type_and_chroma(image: np.ndarray) -> None:
    before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    corrected = correct_illumination(image)
    after = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8
    assert np.max(np.abs(before[:, :, 1:].astype(int) - after[:, :, 1:].astype(int))) <= 2


def test_fixed_roi_and_invalid_roi_fallback() -> None:
    image = textured_image(100, 80)
    cropped, metadata = extract_roi(
        image,
        small_config(
            roi_mode="fixed",
            fixed_roi={"x1_ratio": 0.1, "y1_ratio": 0.25, "x2_ratio": 0.9, "y2_ratio": 0.75},
        ),
    )
    assert cropped.shape[:2] == (40, 80)
    assert metadata.applied

    fallback, fallback_metadata = extract_roi(
        image,
        small_config(roi_mode="fixed", fixed_roi={"x1": 50, "y1": 20, "x2": 50, "y2": 30}),
    )
    assert np.array_equal(fallback, image)
    assert fallback_metadata.fallback_reason == "invalid_or_empty_fixed_roi"


def test_orientation_disabled_and_safety_limit_keep_input() -> None:
    image = textured_image()
    disabled, disabled_metadata = normalize_orientation(image, small_config())
    assert np.array_equal(disabled, image)
    assert not disabled_metadata.applied
    limited, limited_metadata = normalize_orientation(
        image,
        small_config(
            apply_orientation_normalization=True,
            orientation_confidence_threshold=0.0,
            maximum_rotation_degrees=0.0,
        ),
    )
    assert np.array_equal(limited, image)
    assert not limited_metadata.applied


def test_denoising_and_clahe_are_deterministic() -> None:
    image = textured_image()
    assert np.array_equal(bilateral_denoise(image), bilateral_denoise(image))
    assert np.array_equal(apply_luminance_clahe(image), apply_luminance_clahe(image))
    assert apply_luminance_clahe(image).shape == image.shape


def test_resize_letterbox_padding_and_coordinate_mapping() -> None:
    image = textured_image(100, 50)
    resized, metadata = resize_preserving_aspect_ratio(image, 60)
    assert resized.shape[:2] == (30, 60)
    assert metadata.scale == pytest.approx(0.6)
    boxed, boxed_metadata = letterbox(image, (120, 120))
    assert boxed.shape[:2] == (120, 120)
    assert map_box_to_source((0, boxed_metadata.pad_top, 120, 120 - boxed_metadata.pad_bottom), boxed_metadata) == (
        0,
        0,
        100,
        50,
    )
    padded = reflection_pad(image, 3, 4, 5, 6)
    assert padded.shape[:2] == (57, 111)


def test_patch_coordinates_overlap_and_reflection_padding() -> None:
    assert deterministic_positions(500, 224, 0.5) == (0, 112, 224, 276)
    patches = extract_overlapping_patches(
        textured_image(100, 80),
        patch_sizes=(64, 128),
        overlap_ratio=0.5,
        source_image="source.png",
        fabric_type="woven",
        split="train",
    )
    assert patches
    assert all(item.image.shape[:2] == (item.metadata.patch_size, item.metadata.patch_size) for item in patches)
    assert patches[-1].metadata.padding_right > 0
    assert all(item.metadata.split == "train" for item in patches)


def test_texture_and_fft_feature_shapes() -> None:
    image = textured_image()
    texture = generate_texture_view(image)
    features = calculate_fft_features(image)
    assert texture.shape == image.shape
    assert texture.dtype == np.uint8
    assert len(features.to_dict()) == 8
    assert 0 <= features.spectral_energy_ratio <= 1


def test_registry_fallback_and_validation(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "references": [],
                "fabric_preprocessing": {
                    "woven": {
                        "preprocessing_version": "v2",
                        "clahe_clip_limit": 1.2,
                        "orientation_normalization": True,
                        "color_normalization": {"enabled": False},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    overrides = load_registry_overrides(registry, "woven")
    assert overrides["clahe_clip_limit"] == 1.2
    assert overrides["apply_orientation_normalization"] is True
    assert load_registry_overrides(registry, "unknown") == {}
    assert config_for_ablation("P7").calculate_fft_features


def test_corrupted_input_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    image, error = load_image(path)
    assert image is None
    assert error == "cannot_decode"


def test_pipeline_is_deterministic() -> None:
    image = textured_image(96, 96)
    config = small_config(generate_texture_view=True, target_max_size=80)
    first = preprocess_fabric_image(image, config)
    second = preprocess_fabric_image(image, config)
    assert first.accepted and second.accepted
    assert np.array_equal(first.processed_image, second.processed_image)
    assert np.array_equal(first.texture_view, second.texture_view)
    assert first.configuration_hash == second.configuration_hash


def test_shared_adapter_prepares_v2_from_serialized_config(tmp_path: Path) -> None:
    path = tmp_path / "fabric.png"
    assert cv2.imwrite(str(path), textured_image(96, 96))
    config = small_config(target_max_size=80, patch_sizes=(32,), patch_overlap_ratio=0.5)
    prepared = prepare_image(
        path,
        preprocessing_version="v2",
        image_max_size=80,
        patch_size=32,
        patch_stride=16,
        preprocessing_config=config.to_dict(),
        split="validation",
    )
    assert prepared.preprocessing_version == "v2"
    assert prepared.configuration_hash == config.configuration_hash
    assert prepared.patches
    assert all(patch.image.size == (32, 32) for patch in prepared.patches)


def test_shared_adapter_maps_v2_boxes_back_through_roi_and_resize(tmp_path: Path) -> None:
    path = tmp_path / "fabric.png"
    assert cv2.imwrite(str(path), textured_image(200, 100))
    config = small_config(
        target_max_size=80,
        patch_sizes=(32,),
        roi_mode="fixed",
        fixed_roi={"x1": 20, "y1": 10, "x2": 180, "y2": 90},
    )
    prepared = prepare_image(
        path,
        preprocessing_version="v2",
        image_max_size=80,
        patch_size=32,
        patch_stride=16,
        preprocessing_config=config.to_dict(),
    )

    assert (prepared.width, prepared.height) == (80, 40)
    assert prepared.coordinate_transform.map_box_to_original((0, 0, 80, 40)) == (20, 10, 180, 90)


def test_batch_integration_records_rejections_patches_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    valid = source / "valid.png"
    rejected = source / "small.png"
    assert cv2.imwrite(str(valid), textured_image(96, 96))
    assert cv2.imwrite(str(rejected), textured_image(16, 16))
    output = tmp_path / "preprocessed_v2"
    config = small_config(min_width=32, min_height=32, patch_sizes=(64,))
    first = process_directory(
        source,
        output,
        config,
        split="train",
        extract_patches=True,
        save_texture_view=True,
    )
    assert first.processed == 1
    assert first.rejected == 1
    assert first.patch_count > 0
    assert (output / "rgb" / "train" / "valid.png").exists()
    assert (output / "rejected_quality" / "train" / "small.png").exists()
    with first.metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    second = process_directory(source, output, config, split="train")
    assert second.skipped_existing == 1
