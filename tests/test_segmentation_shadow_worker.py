import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from backend.app.ai.segmentation_dataset import sha256_file
from backend.app.ai.segmentation_shadow_worker import observe_checkpoint
from tests.segmentation_shadow_fixtures import shadow_checkpoint
from tests.segmentation_training_fixtures import deterministic_cpu


def test_real_checkpoint_prediction_version_geometry_probabilities_mask_and_confidence(shadow_checkpoint):
    image, config = shadow_checkpoint
    value = observe_checkpoint(image, "fixture_fabric", config)
    assert value.status == "observed", value
    assert value.model_version == config.expected_model_version and value.checkpoint_sha256 == config.checkpoint_sha256
    assert value.synthetic_checkpoint and value.original_shape == (32, 48) and value.mask_shape == (32, 32)
    assert value.classes[1].predicted_pixels == 1024 and value.classes[1].mean_winning_confidence > 0.99
    assert value.mask_rle == ((1, 1024),)
    mask = np.ones((32, 32), dtype="<i4")
    assert value.mask_sha256 == hashlib.sha256(mask.tobytes()).hexdigest()
    repeated = observe_checkpoint(image, "fixture_fabric", config)
    assert value == repeated


@pytest.mark.parametrize("failure,reason", [("missing", "checkpoint_missing"), ("untrusted", "checkpoint_untrusted"),
                  ("wrong_fabric", "fabric_not_allowed"), ("hash", "checkpoint_hash_mismatch"), ("size", "checkpoint_too_large"),
                  ("version", "version_mismatch"), ("synthetic", "synthetic_not_allowed"), ("corrupt", "checkpoint_invalid")])
def test_unsafe_unavailable_checkpoint_returns_no_fabricated_evidence(shadow_checkpoint, tmp_path, failure, reason):
    image, config = shadow_checkpoint
    changes, fabric = {}, "fixture_fabric"
    if failure == "missing":
        changes["checkpoint_path"] = str(tmp_path / "absent.pt")
    elif failure == "untrusted":
        changes["trusted_local"] = False
    elif failure == "wrong_fabric":
        fabric = "other_fabric"
    elif failure == "hash":
        changes["checkpoint_sha256"] = "b" * 64
    elif failure == "size":
        changes["max_checkpoint_bytes"] = 1
    elif failure == "version":
        changes["expected_model_version"] = "other_v1"
    elif failure == "synthetic":
        changes["allow_synthetic"] = False
    else:
        broken = tmp_path / "bad.pt"
        broken.write_bytes(b"invalid checkpoint fixture")
        changes.update(checkpoint_path=str(broken), checkpoint_sha256=sha256_file(broken))
    value = observe_checkpoint(image, fabric, config.model_copy(update=changes))
    assert value.status == "unavailable" and value.reason == reason
    assert not value.classes and value.mask_shape is None and value.model_version is None


def test_hash_mismatch_is_rejected_before_deserialization(shadow_checkpoint):
    image, config = shadow_checkpoint
    with patch("backend.app.ai.segmentation_trainer.load_segmentation_checkpoint", side_effect=AssertionError("must not deserialize")):
        value = observe_checkpoint(image, "fixture_fabric", config.model_copy(update={"checkpoint_sha256": "b" * 64}))
    assert value.reason == "checkpoint_hash_mismatch"


def test_mask_runs_are_bounded_without_faking_or_dropping_class_counts(shadow_checkpoint):
    image, config = shadow_checkpoint
    value = observe_checkpoint(image, "fixture_fabric", config.model_copy(update={"max_mask_runs": 0}))
    assert value.status == "observed" and value.mask_rle is None and value.mask_rle_omitted
    assert sum(item.predicted_pixels for item in value.classes) == 1024 and value.mask_sha256


def test_exif_orientation_is_explicit_in_original_geometry(shadow_checkpoint, tmp_path):
    _, config = shadow_checkpoint
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (48, 32), (150, 150, 150)).save(path, exif=exif)
    value = observe_checkpoint(path, "fixture_fabric", config)
    assert value.status == "observed" and value.original_shape == (48, 32)
    assert value.mask_coordinate_frame == "model_grid"


def test_corrupt_capture_does_not_fabricate_segmentation(shadow_checkpoint, tmp_path):
    _, config = shadow_checkpoint
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"invalid capture")
    value = observe_checkpoint(path, "fixture_fabric", config)
    assert value.status == "error" and value.reason == "image_decode_failed" and not value.classes


def test_invalid_exif_orientation_is_not_silently_reinterpreted(shadow_checkpoint, tmp_path):
    _, config = shadow_checkpoint
    path = tmp_path / "invalid_exif.jpg"
    exif = Image.Exif()
    exif[274] = 9
    Image.new("RGB", (48, 32)).save(path, exif=exif)
    value = observe_checkpoint(path, "fixture_fabric", config)
    assert value.status == "error" and value.reason == "image_decode_failed"


def test_inference_exception_is_sanitized(shadow_checkpoint):
    image, config = shadow_checkpoint
    with patch("backend.app.ai.segmentation_model.ResNet18UNet.predict", side_effect=RuntimeError("secret path should not leak")):
        value = observe_checkpoint(image, "fixture_fabric", config)
    assert value.status == "error" and value.reason == "prediction_failed"
    assert "secret" not in value.model_dump_json()
