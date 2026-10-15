"""Capture-quality evidence, separate from fabric defect classification."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.app.ai.preprocessing.quality import QualityMetrics, calculate_quality_metrics, validate_quality
from backend.app.ai.preprocess import ImageReadError, inspect_image_file
from backend.app.core.acquisition import AcquisitionMetadata, QualityGateConfig
from backend.app.core.inspection_contracts import InspectionInput


class CaptureStatus(str, Enum):
    VALID_CAPTURE = "VALID_CAPTURE"
    INVALID_CAPTURE = "INVALID_CAPTURE"


@dataclass(frozen=True)
class QualityGateResult:
    metrics: QualityMetrics
    reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.metrics.decode_success and not self.reasons

    @property
    def status(self) -> CaptureStatus:
        return CaptureStatus.VALID_CAPTURE if self.valid else CaptureStatus.INVALID_CAPTURE


class InvalidCaptureError(ValueError):
    def __init__(self, result: QualityGateResult) -> None:
        self.result = result
        super().__init__(f"INVALID_CAPTURE: {', '.join(result.reasons) or 'cannot_decode'}")


class QualityGate(Protocol):
    def evaluate(self, inspection: InspectionInput, config: QualityGateConfig) -> QualityGateResult:
        ...


class ImageQualityGate:
    def evaluate(self, inspection: InspectionInput, config: QualityGateConfig) -> QualityGateResult:
        # Match the dense input decoder's RGB/EXIF convention, including images
        # with nonstandard filename extensions. This does not alter the source.
        try:
            inspect_image_file(inspection.image_path)
            with Image.open(inspection.image_path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except (ImageReadError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            image = None
        policy = config.image_quality_config()
        metrics = calculate_quality_metrics(image, policy)
        reasons = list(validate_quality(metrics, policy).reasons)
        metadata = inspection.metadata
        if metadata.roi is not None and metrics.decode_success:
            roi = metadata.roi
            if (not isinstance(roi, tuple) or len(roi) != 4
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in roi)):
                reasons.append("invalid_roi")
            else:
                x, y, width, height = roi
                if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > metrics.width or y + height > metrics.height:
                    reasons.append("invalid_roi")
        if config.check_metadata:
            reasons.extend(_metadata_reasons(metadata, config))
        return QualityGateResult(metrics, tuple(reasons))


def _metadata_reasons(metadata: AcquisitionMetadata, config: QualityGateConfig) -> list[str]:
    reasons: list[str] = []
    for name in config.required_metadata:
        if getattr(metadata, name) is None:
            reasons.append(f"missing_{name}")
    for name in ("lot_id", "roll_id", "camera_id", "frame_id"):
        value = getattr(metadata, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            reasons.append(f"invalid_{name}")
    if config.expected_camera_id is not None and metadata.camera_id != config.expected_camera_id:
        reasons.append("camera_id_mismatch")
    value = metadata.encoder_position
    if value is not None and not _valid_encoder_position(value):
        reasons.append("invalid_encoder_position")
    timestamp = metadata.timestamp
    if timestamp is not None and (not isinstance(timestamp, datetime) or timestamp.utcoffset() is None):
        reasons.append("invalid_timestamp")
    return reasons


def _valid_encoder_position(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False
