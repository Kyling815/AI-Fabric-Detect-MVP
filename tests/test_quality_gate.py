from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pytest

from backend.app.ai.quality_gate import CaptureStatus, ImageQualityGate, InvalidCaptureError, QualityGateResult
from backend.app.ai.preprocessing.quality import QualityMetrics
from backend.app.core.acquisition import AcquisitionMetadata, QualityGateConfig
from backend.app.core.inspection_contracts import InspectionInput
from tests.quality_gate_fixtures import capture_pixels, corrupt_png_checksum, write_capture


@pytest.mark.parametrize("kind, width, height, reason", [
    ("valid", 128, 96, None), ("black", 128, 96, "image_too_dark"),
    ("white", 128, 96, "image_too_bright"), ("blurred", 128, 96, "image_too_blurry"),
    ("valid", 16, 16, "image_width_too_small"),
])
def test_synthetic_capture_quality_is_deterministic_and_read_only(tmp_path: Path, kind, width, height, reason) -> None:
    path = write_capture(tmp_path, kind, width, height)
    before = path.read_bytes()
    value = InspectionInput(path, "fabric_001")
    first = ImageQualityGate().evaluate(value, QualityGateConfig())
    second = ImageQualityGate().evaluate(value, QualityGateConfig())
    assert first == second
    assert first.metrics.width == width and first.metrics.height == height
    assert path.read_bytes() == before
    if reason is None:
        assert first.valid and first.status == CaptureStatus.VALID_CAPTURE
        assert first.reasons == ()
        assert first.metrics.dark_ratio == first.metrics.bright_ratio == 0
    else:
        assert not first.valid and first.status == CaptureStatus.INVALID_CAPTURE
        assert reason in first.reasons


@pytest.mark.parametrize("missing", [False, True])
def test_corrupt_or_missing_image_returns_invalid_capture(tmp_path: Path, missing: bool) -> None:
    path = tmp_path / "invalid.png"
    if not missing:
        path.write_bytes(b"not an image")
    result = ImageQualityGate().evaluate(InspectionInput(path, "fabric_001"), QualityGateConfig())
    assert result.status == CaptureStatus.INVALID_CAPTURE
    assert not result.metrics.decode_success
    assert result.reasons == ("cannot_decode",)


def test_bad_png_checksum_is_invalid_even_when_pixels_are_decodable(tmp_path: Path) -> None:
    path = write_capture(tmp_path, "valid")
    path.write_bytes(corrupt_png_checksum(path.read_bytes()))
    result = ImageQualityGate().evaluate(InspectionInput(path, "fabric_001"), QualityGateConfig())
    assert result.status == CaptureStatus.INVALID_CAPTURE
    assert result.reasons == ("cannot_decode",)


@pytest.mark.parametrize("roi, valid", [
    ((0, 0, 128, 96), True), ((120, 88, 8, 8), True),
    ((-1, 0, 8, 8), False), ((0, 0, 0, 8), False),
    ((120, 0, 9, 8), False), ((0, 90, 8, 7), False),
    ((0, 0, 8), False), ((False, 0, 8, 8), False), ((0.5, 0, 8, 8), False),
])
def test_roi_validation_uses_full_oriented_source_pixels(tmp_path: Path, roi, valid: bool) -> None:
    value = InspectionInput(write_capture(tmp_path, "valid"), "fabric_001", AcquisitionMetadata(roi=roi))
    result = ImageQualityGate().evaluate(value, QualityGateConfig())
    assert result.valid is valid
    assert result.metrics.width == 128 and result.metrics.height == 96
    assert ("invalid_roi" in result.reasons) is not valid


def test_exif_orientation_matches_dense_decoder(tmp_path: Path) -> None:
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(capture_pixels("valid")).save(path, exif=exif)
    value = InspectionInput(path, "fabric_001", AcquisitionMetadata(roi=(0, 0, 96, 128)))
    result = ImageQualityGate().evaluate(value, QualityGateConfig())
    assert result.valid
    assert (result.metrics.width, result.metrics.height) == (96, 128)


def test_nonstandard_extension_and_grayscale_are_decoded_like_existing_runtime(tmp_path: Path) -> None:
    path = tmp_path / "capture.img"
    Image.fromarray(capture_pixels("valid")[:, :, 0]).save(path, format="PNG")
    result = ImageQualityGate().evaluate(InspectionInput(path, "fabric_001"), QualityGateConfig())
    assert result.valid and result.metrics.channel_count == 3


def test_optional_metadata_is_not_required_for_uploads(tmp_path: Path) -> None:
    result = ImageQualityGate().evaluate(InspectionInput(write_capture(tmp_path, "valid"), "fabric_001"),
                                        QualityGateConfig(check_metadata=True))
    assert result.valid


def test_explicit_metadata_requirements_are_checked_without_guessing(tmp_path: Path) -> None:
    result = ImageQualityGate().evaluate(InspectionInput(write_capture(tmp_path, "valid"), "fabric_001"),
                                        QualityGateConfig(check_metadata=True, required_metadata=("camera_id", "roll_id")))
    assert result.reasons == ("missing_camera_id", "missing_roll_id")


@pytest.mark.parametrize("metadata, reason", [
    (AcquisitionMetadata(camera_id="CAM_1"), "camera_id_mismatch"),
    (AcquisitionMetadata(camera_id="CAM_2", lot_id=" "), "invalid_lot_id"),
    (AcquisitionMetadata(camera_id="CAM_2", frame_id=123), "invalid_frame_id"),
    (AcquisitionMetadata(camera_id="CAM_2", encoder_position=-1), "invalid_encoder_position"),
    (AcquisitionMetadata(camera_id="CAM_2", encoder_position=float("nan")), "invalid_encoder_position"),
    (AcquisitionMetadata(camera_id="CAM_2", encoder_position=float("inf")), "invalid_encoder_position"),
    (AcquisitionMetadata(camera_id="CAM_2", encoder_position=10 ** 1000), "invalid_encoder_position"),
    (AcquisitionMetadata(camera_id="CAM_2", timestamp=datetime(2026, 10, 13)), "invalid_timestamp"),
])
def test_optional_metadata_consistency(tmp_path: Path, metadata, reason) -> None:
    result = ImageQualityGate().evaluate(InspectionInput(write_capture(tmp_path, "valid"), "fabric_001", metadata),
                                        QualityGateConfig(check_metadata=True, expected_camera_id="CAM_2"))
    assert not result.valid and reason in result.reasons


def test_valid_metadata_is_preserved_without_physical_conversion(tmp_path: Path) -> None:
    metadata = AcquisitionMetadata("LOT_1", "ROLL_1", "CAM_2", "FRAME_1", datetime(2026, 10, 13, tzinfo=timezone.utc), 123.5)
    before = asdict(metadata)
    result = ImageQualityGate().evaluate(InspectionInput(write_capture(tmp_path, "valid"), "fabric_001", metadata),
                                        QualityGateConfig(check_metadata=True, required_metadata=("camera_id", "timestamp"), expected_camera_id="CAM_2"))
    assert result.valid and asdict(metadata) == before


def test_plain_low_texture_image_can_use_calibrated_blur_override(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    Image.new("RGB", (128, 96), (128, 128, 128)).save(path)
    result = ImageQualityGate().evaluate(InspectionInput(path, "fabric_001"), QualityGateConfig(blur_threshold=0))
    assert result.valid and result.metrics.laplacian_variance == 0


def test_invalid_capture_error_carries_quality_evidence_not_qc_label() -> None:
    result = QualityGateResult(QualityMetrics(), ("cannot_decode",))
    error = InvalidCaptureError(result)
    assert error.result is result
    assert str(error) == "INVALID_CAPTURE: cannot_decode"
    assert not hasattr(result, "final_result")


@pytest.mark.parametrize("values", [
    {"min_width": 0}, {"blur_threshold": -1}, {"blur_threshold": float("nan")},
    {"max_dark_ratio": 1.1}, {"dark_pixel_threshold": 240, "bright_pixel_threshold": 235},
    {"required_metadata": ("camera_id",)}, {"check_metadata": True, "required_metadata": ("unknown",)},
    {"check_metadata": True, "expected_camera_id": " "}, {"unknown_setting": True},
])
def test_invalid_gate_configuration_is_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        QualityGateConfig(**values)
