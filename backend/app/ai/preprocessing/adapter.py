"""Shared v1/v2 preparation adapter for training, references, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from backend.app.ai.patching import ImagePatch, PatchBox, build_patch_grid, extract_image_patches
from backend.app.ai.preprocess import (
    ImageReadError,
    build_preprocess_plan,
    inspect_image_file,
    load_resized_rgb_image,
)
from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.ai.preprocessing.patches import extract_overlapping_patches
from backend.app.ai.preprocessing.pipeline import preprocess_path
from backend.app.ai.preprocessing.resize import ResizeMetadata, map_box_to_source


@dataclass(frozen=True)
class CoordinateTransform:
    source_width: int
    source_height: int
    roi_x: int
    roi_y: int
    resize: ResizeMetadata
    orientation_correction_degrees: float = 0.0
    orientation_applied: bool = False

    def map_box_to_original(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Map an axis-aligned processed-image box back to the uploaded image."""
        x1, y1, x2, y2 = map_box_to_source(box, self.resize)
        if self.orientation_applied:
            center = (self.resize.source_width / 2.0, self.resize.source_height / 2.0)
            forward = cv2.getRotationMatrix2D(center, self.orientation_correction_degrees, 1.0)
            inverse = cv2.invertAffineTransform(forward)
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float64,
            ).reshape(-1, 1, 2)
            mapped = cv2.transform(corners, inverse).reshape(-1, 2)
            x1 = math.floor(float(mapped[:, 0].min()))
            y1 = math.floor(float(mapped[:, 1].min()))
            x2 = math.ceil(float(mapped[:, 0].max()))
            y2 = math.ceil(float(mapped[:, 1].max()))

        return (
            max(0, min(self.source_width, x1 + self.roi_x)),
            max(0, min(self.source_height, y1 + self.roi_y)),
            max(0, min(self.source_width, x2 + self.roi_x)),
            max(0, min(self.source_height, y2 + self.roi_y)),
        )


@dataclass(frozen=True)
class PreparedImage:
    patches: list[ImagePatch]
    width: int
    height: int
    preprocessing_version: str
    configuration_hash: str | None
    preprocessing_config: dict[str, Any]
    coordinate_transform: CoordinateTransform


def prepare_image(
    path: Path,
    *,
    preprocessing_version: str,
    image_max_size: int,
    patch_size: int,
    patch_stride: int,
    preprocessing_config: Mapping[str, Any] | None = None,
    fabric_type: str | None = None,
    split: str = "unspecified",
) -> PreparedImage:
    version = preprocessing_version.lower()
    if version == "v1":
        info = inspect_image_file(path)
        plan = build_preprocess_plan(
            width=info.width,
            height=info.height,
            target_max_size=image_max_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        image = load_resized_rgb_image(path, plan)
        grid = build_patch_grid(image.width, image.height, patch_size, patch_stride)
        return PreparedImage(
            patches=extract_image_patches(image, grid),
            width=image.width,
            height=image.height,
            preprocessing_version="v1",
            configuration_hash=None,
            preprocessing_config={
                "image_max_size": image_max_size,
                "patch_size": patch_size,
                "patch_stride": patch_stride,
            },
            coordinate_transform=CoordinateTransform(
                source_width=info.width,
                source_height=info.height,
                roi_x=0,
                roi_y=0,
                resize=ResizeMetadata(
                    source_width=info.width,
                    source_height=info.height,
                    resized_width=plan.resized_width,
                    resized_height=plan.resized_height,
                    output_width=plan.resized_width,
                    output_height=plan.resized_height,
                    scale=plan.scale,
                ),
            ),
        )
    if version != "v2":
        raise ValueError(f"Unsupported preprocessing version: {preprocessing_version}")

    values = dict(preprocessing_config or {})
    values["preprocessing_version"] = "v2"
    values.setdefault("target_max_size", image_max_size)
    values.setdefault("patch_sizes", (patch_size,))
    values.setdefault("patch_overlap_ratio", _overlap_from_stride(patch_size, patch_stride))
    config = PreprocessConfig.from_mapping(values)
    result = preprocess_path(path, config, fabric_type=fabric_type, split=split)
    if result.error:
        raise ValueError(result.error)
    if result.rejection_reasons:
        raise ImageReadError(f"Image rejected by preprocessing v2: {', '.join(result.rejection_reasons)}")
    if result.processed_image is None:
        raise ImageReadError(f"Preprocessing v2 produced no image: {path}")
    extracted = extract_overlapping_patches(
        result.processed_image,
        patch_sizes=config.patch_sizes,
        overlap_ratio=config.patch_overlap_ratio,
        source_image=path,
        fabric_type=fabric_type,
        split=split,
        source_scale=result.resize_metadata.scale if result.resize_metadata else 1.0,
        minimum_fabric_coverage=config.minimum_fabric_coverage,
    )
    patches = [
        ImagePatch(
            box=PatchBox(
                x=item.metadata.x1,
                y=item.metadata.y1,
                w=item.image.shape[1],
                h=item.image.shape[0],
            ),
            image=Image.fromarray(item.image[:, :, ::-1], mode="RGB"),
        )
        for item in extracted
    ]
    height, width = result.processed_image.shape[:2]
    resize = result.resize_metadata
    if resize is None:
        raise ImageReadError(f"Preprocessing v2 produced no resize metadata: {path}")
    roi = result.roi_metadata
    orientation = result.orientation_metadata
    return PreparedImage(
        patches=patches,
        width=width,
        height=height,
        preprocessing_version="v2",
        configuration_hash=result.configuration_hash,
        preprocessing_config=config.to_dict(),
        coordinate_transform=CoordinateTransform(
            source_width=result.quality_metrics.width,
            source_height=result.quality_metrics.height,
            roi_x=roi.x1 if roi else 0,
            roi_y=roi.y1 if roi else 0,
            resize=resize,
            orientation_correction_degrees=orientation.correction_degrees if orientation else 0.0,
            orientation_applied=bool(orientation and orientation.applied),
        ),
    )


def configuration_from_reference_metadata(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    version = str(metadata.get("preprocessing_version") or metadata.get("preprocess", {}).get("version") or "v1")
    preprocess = metadata.get("preprocess") or {}
    config = preprocess.get("configuration") or metadata.get("preprocessing_config") or {}
    return version, dict(config)


def assert_reference_compatibility(
    *,
    expected_version: str,
    expected_hash: str | None,
    reference_metadata: Mapping[str, Any],
) -> None:
    actual_version, _ = configuration_from_reference_metadata(reference_metadata)
    if actual_version != expected_version:
        raise ValueError(
            f"Preprocessing mismatch: query uses {expected_version}, reference uses {actual_version}."
        )
    actual_hash = reference_metadata.get("configuration_hash")
    if expected_version == "v2" and expected_hash and actual_hash and expected_hash != actual_hash:
        raise ValueError("Preprocessing v2 configuration hash does not match the reference index.")


def _overlap_from_stride(patch_size: int, patch_stride: int) -> float:
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("Patch size and stride must be positive.")
    return max(0.0, min(0.99, 1.0 - patch_stride / patch_size))
