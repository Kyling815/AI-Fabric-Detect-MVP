from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from backend.app.ai.segmentation_dataset import SegmentationPair, load_segmentation_sample, paired_segmentation_transform, sha256_file
from backend.app.core.segmentation_data import (SegmentationAugmentationConfig, SegmentationManifest, SegmentationPipelineConfig,
                                                SegmentationSample, SegmentationTaxonomy, SegmentationTrainingConfig)
from scripts.check_segmentation_pipeline import create_demo_manifest


@pytest.fixture
def manifest(tmp_path):
    return create_demo_manifest(tmp_path)


def change_mask(sample, values, *, mode=None):
    path = Path(sample.mask_path)
    image = Image.fromarray(values) if mode is None else Image.fromarray(values, mode=mode)
    image.save(path)
    return sample.model_copy(update={"mask_sha256": sha256_file(path)})


def test_good_empty_mask_and_missing_mask_are_accepted(manifest):
    good = manifest.samples[0]
    first = load_segmentation_sample(good, manifest.taxonomy, allow_synthetic=True)
    assert first.image.shape == (24, 32, 3) and first.mask.shape == (24, 32)
    assert first.mask.dtype == np.int64 and not first.mask.any()
    no_mask = good.model_copy(update={"mask_path": None, "mask_sha256": None})
    np.testing.assert_array_equal(load_segmentation_sample(no_mask, manifest.taxonomy, allow_synthetic=True).mask, first.mask)


def test_synthetic_requires_explicit_opt_in(manifest):
    with pytest.raises(ValueError, match="Synthetic"):
        load_segmentation_sample(manifest.samples[1], manifest.taxonomy)


def test_defect_mask_geometry_mismatch_rejected_without_resizing(manifest):
    sample = change_mask(manifest.samples[1], np.ones((12, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True)


@pytest.mark.parametrize("failure", ["inactive_label", "empty_defect", "foreground_on_good", "rgb_mask", "float_mask", "wrong_image_hash", "wrong_mask_hash"])
def test_invalid_mask_or_integrity_rejected(manifest, failure):
    sample = manifest.samples[1]
    if failure == "inactive_label":
        sample = change_mask(sample, np.full((24, 32), 2, dtype=np.uint8))
    elif failure == "empty_defect":
        sample = change_mask(sample, np.zeros((24, 32), dtype=np.uint8))
    elif failure == "foreground_on_good":
        sample = sample.model_copy(update={"label": "normal", "defect_classes": ()})
    elif failure == "rgb_mask":
        sample = change_mask(sample, np.ones((24, 32, 3), dtype=np.uint8))
    elif failure == "float_mask":
        path = Path(sample.mask_path).with_suffix(".tiff")
        Image.fromarray(np.ones((24, 32), dtype=np.float32)).save(path)
        sample = sample.model_copy(update={"mask_path": str(path), "mask_sha256": sha256_file(path)})
    else:
        sample = sample.model_copy(update={"image_sha256" if failure == "wrong_image_hash" else "mask_sha256": "0" * 64})
    with pytest.raises(ValueError):
        load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True)


def test_binary_255_requires_explicit_mapping_and_exact_pixels(manifest):
    sample = manifest.samples[1]
    mask = np.asarray(Image.open(sample.mask_path)) * 255
    sample = change_mask(sample, mask)
    sample = sample.model_copy(update={"mask_encoding": "binary_255", "binary_class_name": "defect"})
    assert set(np.unique(load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True).mask)) == {0, 1}
    mask[0, 0] = 128
    sample = change_mask(sample, mask)
    with pytest.raises(ValueError, match="exactly 0/255"):
        load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True)


def test_configurable_multiclass_subset_matches_exact_declared_mask_labels(manifest):
    sample = manifest.samples[1]
    mask = np.asarray(Image.open(sample.mask_path)).copy()
    mask[0:3, 0:3] = 2
    sample = change_mask(sample, mask).model_copy(update={"defect_classes": ("scratch", "stain")})
    taxonomy = SegmentationTaxonomy(classes=("background", "scratch", "stain"))
    assert set(np.unique(load_segmentation_sample(sample, taxonomy, allow_synthetic=True).mask)) == {0, 1, 2}


def test_palette_masks_preserve_ids_not_luminance(manifest):
    sample = manifest.samples[1]
    path = Path(sample.mask_path)
    values = np.asarray(Image.open(path)).copy()
    palette = Image.fromarray(values, mode="P")
    palette.putpalette([255, 255, 255, 0, 0, 0] + [0] * (768 - 6))
    palette.save(path)
    sample = sample.model_copy(update={"mask_sha256": sha256_file(path)})
    np.testing.assert_array_equal(load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True).mask, values)


@pytest.mark.parametrize("coordinates", ["raw_image", "oriented_image"])
def test_exif_orientation_applies_to_the_declared_mask_coordinate_frame(tmp_path, coordinates):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    image[1, 2] = 255
    mask[1, 2] = 1
    image_path, mask_path = tmp_path / "capture.png", tmp_path / "mask.png"
    exif = Image.Exif(); exif[274] = 6
    Image.fromarray(image).save(image_path, exif=exif)
    if coordinates == "oriented_image":
        mask = np.rot90(mask, -1)
    Image.fromarray(mask).save(mask_path)
    sample = SegmentationSample(sample_id="oriented", fabric_id="f", image_path=str(image_path), image_sha256=sha256_file(image_path),
                                 mask_path=str(mask_path), mask_sha256=sha256_file(mask_path), mask_coordinates=coordinates,
                                 label="defect", defect_classes=("defect",), origin="synthetic", annotation_source="synthetic_fixture", source_id="s")
    pair = load_segmentation_sample(sample, SegmentationTaxonomy(), allow_synthetic=True)
    assert pair.image.shape == (6, 4, 3)
    np.testing.assert_array_equal(pair.image[:, :, 0] > 0, pair.mask > 0)


@pytest.mark.parametrize("seed", range(8))
def test_paired_flip_rotation_preserve_alignment_and_do_not_mutate_input(seed):
    mask = np.zeros((6, 9), dtype=np.int64); mask[1:4, 2:5] = 1
    image = np.repeat((mask * 100).astype(np.uint8)[..., None], 3, axis=2)
    pair = SegmentationPair(image, mask, "alignment")
    config = SegmentationAugmentationConfig(enabled=True, horizontal_flip_probability=0.5, vertical_flip_probability=0.5,
                                            rotate_quarter_turns=True, brightness_delta=0, contrast_range=(1, 1))
    first = paired_segmentation_transform(pair, config, seed=seed)
    second = paired_segmentation_transform(pair, config, seed=seed)
    np.testing.assert_array_equal(first.image[:, :, 0], first.mask * 100)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(pair.image, image); np.testing.assert_array_equal(pair.mask, mask)


def test_resize_mask_nearest_and_photometric_augmentation_do_not_create_labels(manifest):
    pair = load_segmentation_sample(manifest.samples[1], manifest.taxonomy, allow_synthetic=True)
    transformed = paired_segmentation_transform(pair, SegmentationAugmentationConfig(enabled=True), seed=7, output_size=(13, 19))
    assert transformed.image.shape == (13, 19, 3) and transformed.mask.shape == (13, 19)
    assert set(np.unique(transformed.mask)) == {0, 1}


def test_ignore_id_is_explicit_and_absent_or_all_ignore_is_invalid(manifest):
    sample = manifest.samples[1]
    mask = np.asarray(Image.open(sample.mask_path)).copy(); mask[0, 0] = 255
    sample = change_mask(sample, mask)
    with pytest.raises(ValueError, match="invalid"):
        load_segmentation_sample(sample, manifest.taxonomy, allow_synthetic=True)
    taxonomy = SegmentationTaxonomy(ignore_index=255)
    assert load_segmentation_sample(sample, taxonomy, allow_synthetic=True).mask[0, 0] == 255
    sample = change_mask(sample, np.full((24, 32), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="no valid"):
        load_segmentation_sample(sample, taxonomy, allow_synthetic=True)


@pytest.mark.parametrize("classes", [("defect", "background"), ("background",), ("background", "fake_defect"), ("background", "scratch", "scratch")])
def test_invalid_taxonomy_rejected(classes):
    with pytest.raises(ValidationError):
        SegmentationTaxonomy(classes=classes)


@pytest.mark.parametrize("failure", ["unsupported_label", "duplicate_id", "wrong_version", "normal_defect_classes", "missing_mask"])
def test_invalid_manifest_records_rejected(manifest, failure):
    values = manifest.model_dump(mode="json")
    if failure == "unsupported_label":
        values["samples"][1]["defect_classes"] = ["wrong_color"]
    elif failure == "duplicate_id":
        values["samples"][1]["sample_id"] = values["samples"][0]["sample_id"]
    elif failure == "wrong_version":
        values["schema_version"] = 99
    elif failure == "normal_defect_classes":
        values["samples"][0]["defect_classes"] = ["defect"]
    else:
        values["samples"][1]["mask_path"] = None
        values["samples"][1]["mask_sha256"] = None
    with pytest.raises(ValidationError):
        SegmentationManifest.model_validate(values)


def test_example_training_config_is_disabled_and_does_not_share_encoder():
    root = Path(__file__).resolve().parents[1]
    config = SegmentationPipelineConfig.model_validate_json((root / "config/segmentation_pipeline_example.json").read_text())
    assert not config.training.enabled and not config.training.share_anomaly_encoder and not config.training.allow_synthetic
    with pytest.raises(ValidationError):
        SegmentationTrainingConfig(share_anomaly_encoder=True)


@pytest.mark.parametrize("override", [{"origin": "real"}, {"mask_path": ""}])
def test_synthetic_fixture_cannot_be_misrepresented_or_have_empty_mask_path(manifest, override):
    values = manifest.samples[1].model_dump()
    values.update(override)
    with pytest.raises(ValidationError):
        SegmentationSample.model_validate(values)


def test_augmentation_rejects_fractional_mask_labels():
    pair = SegmentationPair(np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2), 0.5), "fractional")
    with pytest.raises(ValueError):
        paired_segmentation_transform(pair, SegmentationAugmentationConfig(), seed=0, output_size=(4, 4))


@pytest.mark.parametrize("config", [{"image_size": [True, 256]}, {"image_size": [0, 256]}, {"share_anomaly_encoder": 0}, {"decoder_lr": 0}])
def test_invalid_training_specification_is_rejected(config):
    with pytest.raises(ValidationError):
        SegmentationTrainingConfig.model_validate(config)
