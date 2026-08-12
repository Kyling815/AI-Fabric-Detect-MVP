"""End-to-end and batch orchestration for preprocessing v2."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable

import cv2
import numpy as np

from backend.app.ai.preprocessing.color import normalize_fabric_color
from backend.app.ai.preprocessing.config import (
    PreprocessConfig,
    load_registry_overrides,
)
from backend.app.ai.preprocessing.contrast import apply_luminance_clahe
from backend.app.ai.preprocessing.denoise import bilateral_denoise
from backend.app.ai.preprocessing.fft_features import calculate_fft_features
from backend.app.ai.preprocessing.illumination import correct_illumination
from backend.app.ai.preprocessing.orientation import OrientationMetadata, normalize_orientation
from backend.app.ai.preprocessing.patches import ExtractedPatch, extract_overlapping_patches
from backend.app.ai.preprocessing.quality import (
    QualityMetrics,
    calculate_quality_metrics,
    load_image,
    validate_quality,
)
from backend.app.ai.preprocessing.resize import ResizeMetadata, letterbox, resize_preserving_aspect_ratio
from backend.app.ai.preprocessing.roi import RoiMetadata, extract_roi
from backend.app.ai.preprocessing.sharpen import unsharp_mask
from backend.app.ai.preprocessing.texture_view import generate_texture_view

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    processed_image: np.ndarray | None
    quality_metrics: QualityMetrics
    rejection_reasons: tuple[str, ...]
    roi_metadata: RoiMetadata | None
    orientation_metadata: OrientationMetadata | None
    resize_metadata: ResizeMetadata | None
    operations_applied: tuple[str, ...]
    processing_time_ms: float
    configuration_hash: str
    preprocessing_version: str = "v2"
    texture_view: np.ndarray | None = field(default=None, repr=False)
    illumination_corrected_image: np.ndarray | None = field(default=None, repr=False)
    error: str | None = None

    @property
    def accepted(self) -> bool:
        return self.processed_image is not None and not self.rejection_reasons and self.error is None


@dataclass(frozen=True)
class BatchPreprocessSummary:
    discovered: int
    processed: int
    rejected: int
    skipped_existing: int
    failed: int
    patch_count: int
    metadata_path: Path
    quality_report_path: Path
    error_report_path: Path
    output_root: Path
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("metadata_path", "quality_report_path", "error_report_path", "output_root"):
            result[key] = str(result[key])
        return result


def preprocess_fabric_image(
    image: np.ndarray,
    config: PreprocessConfig,
    fabric_type: str | None = None,
    split: str = "unspecified",
    registry: dict[str, Any] | None = None,
) -> PreprocessResult:
    """Process one already-decoded uint8 BGR fabric image deterministically."""

    del fabric_type, split, registry  # Config is resolved before this pure pipeline call.
    started = time.perf_counter()
    metrics = calculate_quality_metrics(image, config)
    validation = validate_quality(metrics, config)
    if not validation.valid:
        return PreprocessResult(
            processed_image=None,
            quality_metrics=metrics,
            rejection_reasons=validation.reasons,
            roi_metadata=None,
            orientation_metadata=None,
            resize_metadata=None,
            operations_applied=("quality_validation",),
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            configuration_hash=config.configuration_hash,
        )

    operations = ["quality_validation"]
    roi_image, roi_metadata = extract_roi(image, config)
    if roi_metadata.applied:
        operations.append("roi")
    oriented, orientation_metadata = normalize_orientation(roi_image, config)
    if orientation_metadata.applied:
        operations.append("orientation_normalization")

    current = oriented
    illumination_image = current.copy()
    if config.apply_illumination_correction:
        current = correct_illumination(current, config.illumination_sigma_ratio)
        illumination_image = current.copy()
        operations.append("illumination_correction")
    if config.apply_color_normalization:
        assert config.lab_reference_mean is not None
        assert config.lab_reference_std is not None
        current = normalize_fabric_color(
            current,
            config.lab_reference_mean,
            config.lab_reference_std,
            config.luminance_strength,
            config.chromatic_strength,
        )
        operations.append("color_normalization")
    if config.apply_denoising:
        current = bilateral_denoise(
            current,
            config.bilateral_diameter,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )
        operations.append("bilateral_denoising")
    if config.apply_clahe:
        current = apply_luminance_clahe(current, config.clahe_clip_limit, config.clahe_grid_size)
        operations.append("clahe")
    if config.apply_sharpening:
        current = unsharp_mask(current, config.sharpen_alpha, config.sharpen_sigma)
        operations.append("unsharp_mask")

    resize_metadata: ResizeMetadata | None = None
    if config.target_max_size is not None:
        current, resize_metadata = resize_preserving_aspect_ratio(current, config.target_max_size)
        if resize_metadata.scale != 1.0:
            operations.append("aspect_ratio_resize")
    if config.letterbox_size is not None:
        current, letterbox_metadata = letterbox(current, config.letterbox_size, reflection=True)
        resize_metadata = _combine_resize_metadata(resize_metadata, letterbox_metadata)
        operations.append("reflection_letterbox")
    if resize_metadata is None:
        height, width = current.shape[:2]
        resize_metadata = ResizeMetadata(width, height, width, height, width, height, 1.0)

    texture = generate_texture_view(current) if config.generate_texture_view else None
    if texture is not None:
        operations.append("texture_view")
    return PreprocessResult(
        processed_image=current,
        quality_metrics=metrics,
        rejection_reasons=(),
        roi_metadata=roi_metadata,
        orientation_metadata=orientation_metadata,
        resize_metadata=resize_metadata,
        operations_applied=tuple(operations),
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
        configuration_hash=config.configuration_hash,
        texture_view=texture,
        illumination_corrected_image=illumination_image,
    )


def preprocess_path(
    path: Path,
    config: PreprocessConfig,
    fabric_type: str | None = None,
    split: str = "unspecified",
) -> PreprocessResult:
    started = time.perf_counter()
    image, error = load_image(path)
    if image is None:
        return PreprocessResult(
            processed_image=None,
            quality_metrics=QualityMetrics(),
            rejection_reasons=(error or "cannot_decode",),
            roi_metadata=None,
            orientation_metadata=None,
            resize_metadata=None,
            operations_applied=("quality_validation",),
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            configuration_hash=config.configuration_hash,
        )
    try:
        return preprocess_fabric_image(image, config, fabric_type=fabric_type, split=split)
    except (cv2.error, ValueError, FloatingPointError) as exc:
        logger.exception("Preprocessing failed for %s", path)
        return PreprocessResult(
            processed_image=None,
            quality_metrics=calculate_quality_metrics(image, config),
            rejection_reasons=(),
            roi_metadata=None,
            orientation_metadata=None,
            resize_metadata=None,
            operations_applied=("quality_validation",),
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            configuration_hash=config.configuration_hash,
            error=f"{type(exc).__name__}: {exc}",
        )


def process_directory(
    input_root: Path,
    output_root: Path,
    config: PreprocessConfig,
    *,
    registry_path: Path | None = None,
    metadata_path: Path | None = None,
    fabric_type: str | None = None,
    split: str = "unspecified",
    extract_patches: bool = False,
    save_texture_view: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BatchPreprocessSummary:
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if split.lower() not in {"train", "validation", "test", "unspecified"}:
        raise ValueError("split must be train, validation, test, or unspecified.")
    overrides = load_registry_overrides(registry_path, fabric_type)
    resolved = config.with_overrides(overrides) if overrides else config
    if save_texture_view and not resolved.generate_texture_view:
        resolved = resolved.with_overrides({"generate_texture_view": True})
    image_paths = sorted(
        (path for path in input_root.rglob("*") if path.is_file()),
        key=lambda item: str(item).casefold(),
    )
    metadata_path = metadata_path or output_root / "metadata" / "preprocessing_v2.csv"
    quality_path = metadata_path.with_name("quality_report_v2.csv")
    error_path = metadata_path.with_name("preprocessing_v2_errors.csv")
    rgb_root = output_root / "rgb" / split
    texture_root = output_root / "texture_view" / split
    rejected_root = output_root / "rejected_quality" / split
    patch_root = output_root.parent / "patches_v2" / split
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    counts = {"processed": 0, "rejected": 0, "skipped_existing": 0, "failed": 0, "patch_count": 0}
    git_commit = _git_commit()
    registry_version = _registry_version(registry_path)

    claimed_outputs: set[Path] = set()
    for path in image_paths:
        relative = path.relative_to(input_root)
        output_path = (rgb_root / relative).with_suffix(".png")
        if output_path in claimed_outputs:
            result = preprocess_path(path, resolved, fabric_type=fabric_type, split=split)
            result = replace_result_error(result, f"Duplicate output path: {output_path}")
            status = "failed"
            counts["failed"] += 1
            error_rows.append({"source_path": str(path), "error": result.error})
            rows.append(
                _metadata_row(
                    path,
                    output_path,
                    result,
                    status=status,
                    fabric_type=fabric_type,
                    split=split,
                    git_commit=git_commit,
                    random_seed=resolved.random_seed,
                    registry_version=registry_version,
                )
            )
            quality_rows.append(_quality_row(path, result, status))
            continue
        claimed_outputs.add(output_path)
        result = preprocess_path(path, resolved, fabric_type=fabric_type, split=split)
        status = "processed"
        if result.error:
            status = "failed"
            counts["failed"] += 1
            error_rows.append({"source_path": str(path), "error": result.error})
        elif result.rejection_reasons:
            status = "rejected"
            counts["rejected"] += 1
            if not dry_run:
                rejected_path = rejected_root / relative
                try:
                    if overwrite or not rejected_path.exists():
                        rejected_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, rejected_path)
                except OSError as exc:
                    counts["rejected"] -= 1
                    counts["failed"] += 1
                    status = "failed"
                    result = replace_result_error(result, f"{type(exc).__name__}: {exc}")
                    error_rows.append({"source_path": str(path), "error": result.error})
        elif output_path.exists() and not overwrite:
            status = "skipped_existing"
            counts["skipped_existing"] += 1
        else:
            counts["processed"] += 1
            patches: list[ExtractedPatch] = []
            if result.processed_image is not None and extract_patches:
                patches = extract_overlapping_patches(
                    result.processed_image,
                    patch_sizes=resolved.patch_sizes,
                    overlap_ratio=resolved.patch_overlap_ratio,
                    source_image=path,
                    fabric_type=fabric_type,
                    split=split,
                    source_scale=result.resize_metadata.scale if result.resize_metadata else 1.0,
                    minimum_fabric_coverage=resolved.minimum_fabric_coverage,
                )
                patches = [_map_patch_to_original(item, result) for item in patches]
                counts["patch_count"] += len(patches)
            if not dry_run and result.processed_image is not None:
                try:
                    _write_image(output_path, result.processed_image)
                    if result.texture_view is not None:
                        _write_image((texture_root / relative).with_suffix(".png"), result.texture_view)
                    _save_patches(
                        patches,
                        patch_root=patch_root,
                        relative_source=relative,
                        calculate_fft=resolved.calculate_fft_features,
                    )
                except (OSError, PermissionError, ValueError, cv2.error) as exc:
                    counts["processed"] -= 1
                    counts["patch_count"] -= len(patches)
                    counts["failed"] += 1
                    status = "failed"
                    result = replace_result_error(result, f"{type(exc).__name__}: {exc}")
                    error_rows.append({"source_path": str(path), "error": result.error})
        row = _metadata_row(
            path,
            output_path,
            result,
            status=status,
            fabric_type=fabric_type,
            split=split,
            git_commit=git_commit,
            random_seed=resolved.random_seed,
            registry_version=registry_version,
        )
        rows.append(row)
        quality_rows.append(_quality_row(path, result, status))

    if not dry_run:
        _write_csv(metadata_path, rows)
        _write_csv(quality_path, quality_rows)
        _write_csv(error_path, error_rows, fallback_fields=("source_path", "error"))
    return BatchPreprocessSummary(
        discovered=len(image_paths),
        processed=counts["processed"],
        rejected=counts["rejected"],
        skipped_existing=counts["skipped_existing"],
        failed=counts["failed"],
        patch_count=counts["patch_count"],
        metadata_path=metadata_path,
        quality_report_path=quality_path,
        error_report_path=error_path,
        output_root=output_root,
        configuration_hash=resolved.configuration_hash,
    )


def _combine_resize_metadata(
    first: ResizeMetadata | None,
    second: ResizeMetadata,
) -> ResizeMetadata:
    if first is None:
        return second
    return ResizeMetadata(
        source_width=first.source_width,
        source_height=first.source_height,
        resized_width=second.resized_width,
        resized_height=second.resized_height,
        output_width=second.output_width,
        output_height=second.output_height,
        scale=first.scale * second.scale,
        pad_left=second.pad_left,
        pad_top=second.pad_top,
        pad_right=second.pad_right,
        pad_bottom=second.pad_bottom,
    )


def replace_result_error(result: PreprocessResult, error: str) -> PreprocessResult:
    return PreprocessResult(
        processed_image=result.processed_image,
        quality_metrics=result.quality_metrics,
        rejection_reasons=result.rejection_reasons,
        roi_metadata=result.roi_metadata,
        orientation_metadata=result.orientation_metadata,
        resize_metadata=result.resize_metadata,
        operations_applied=result.operations_applied,
        processing_time_ms=result.processing_time_ms,
        configuration_hash=result.configuration_hash,
        preprocessing_version=result.preprocessing_version,
        texture_view=result.texture_view,
        illumination_corrected_image=result.illumination_corrected_image,
        error=error,
    )


def _map_patch_to_original(item: ExtractedPatch, result: PreprocessResult) -> ExtractedPatch:
    resize = result.resize_metadata
    roi = result.roi_metadata
    if resize is None:
        return item
    scale = max(resize.scale, 1e-12)
    origin_x = roi.x1 if roi else 0
    origin_y = roi.y1 if roi else 0
    source_width = result.quality_metrics.width
    source_height = result.quality_metrics.height
    metadata = item.metadata
    x1 = origin_x + int(np.floor((metadata.x1 - resize.pad_left) / scale))
    y1 = origin_y + int(np.floor((metadata.y1 - resize.pad_top) / scale))
    x2 = origin_x + int(np.ceil((metadata.x2 - resize.pad_left) / scale))
    y2 = origin_y + int(np.ceil((metadata.y2 - resize.pad_top) / scale))
    mapped = replace(
        metadata,
        x1=max(0, min(source_width, x1)),
        y1=max(0, min(source_height, y1)),
        x2=max(0, min(source_width, x2)),
        y2=max(0, min(source_height, y2)),
    )
    return ExtractedPatch(image=item.image, metadata=mapped)


def _metadata_row(
    source_path: Path,
    output_path: Path,
    result: PreprocessResult,
    *,
    status: str,
    fabric_type: str | None,
    split: str,
    git_commit: str | None,
    random_seed: int,
    registry_version: str,
) -> dict[str, Any]:
    metrics = result.quality_metrics
    roi = result.roi_metadata
    orientation = result.orientation_metadata
    processed_height, processed_width = (
        result.processed_image.shape[:2] if result.processed_image is not None else (0, 0)
    )
    operations = set(result.operations_applied)
    return {
        "source_path": str(source_path),
        "output_path": str(output_path) if result.accepted else "",
        "fabric_type": fabric_type or "",
        "split": split,
        "preprocessing_version": result.preprocessing_version,
        "configuration_hash": result.configuration_hash,
        "code_version": git_commit or "",
        "registry_version": registry_version,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "status": status,
        "width_before": metrics.width,
        "height_before": metrics.height,
        "width_after": processed_width,
        "height_after": processed_height,
        "blur_score": metrics.laplacian_variance,
        "dark_ratio": metrics.dark_ratio,
        "bright_ratio": metrics.bright_ratio,
        "quality_valid": not result.rejection_reasons,
        "rejection_reasons": "|".join(result.rejection_reasons),
        "roi_mode": roi.mode if roi else "",
        "roi_metadata": json.dumps(roi.to_dict(), sort_keys=True) if roi else "",
        "orientation_angle": orientation.estimated_angle_degrees if orientation else "",
        "orientation_confidence": orientation.confidence if orientation else "",
        "orientation_applied": orientation.applied if orientation else False,
        "illumination_applied": "illumination_correction" in operations,
        "color_normalization_applied": "color_normalization" in operations,
        "denoising_applied": "bilateral_denoising" in operations,
        "clahe_applied": "clahe" in operations,
        "sharpening_applied": "unsharp_mask" in operations,
        "operations_applied": "|".join(result.operations_applied),
        "processing_time_ms": round(result.processing_time_ms, 3),
        "error": result.error or "",
    }


def _quality_row(path: Path, result: PreprocessResult, status: str) -> dict[str, Any]:
    metrics = result.quality_metrics
    return {
        "image_path": str(path),
        "width": metrics.width,
        "height": metrics.height,
        "blur_score": metrics.laplacian_variance,
        "dark_ratio": metrics.dark_ratio,
        "bright_ratio": metrics.bright_ratio,
        "status": status,
        "reason": "|".join(result.rejection_reasons) or (result.error or ""),
    }


def _save_patches(
    patches: Iterable[ExtractedPatch],
    *,
    patch_root: Path,
    relative_source: Path,
    calculate_fft: bool,
) -> None:
    metadata_rows: list[dict[str, Any]] = []
    fft_rows: list[dict[str, Any]] = []
    source_stem = relative_source.with_suffix("")
    for item in patches:
        name = f"{source_stem.name}_s{item.metadata.patch_size}_p{item.metadata.patch_index:05d}.png"
        path = patch_root / source_stem.parent / source_stem.name / name
        _write_image(path, item.image)
        record = item.metadata.to_dict()
        record["patch_path"] = str(path)
        metadata_rows.append(record)
        if calculate_fft:
            fft_rows.append({"patch_path": str(path), **calculate_fft_features(item.image).to_dict()})
    if metadata_rows:
        _append_csv(patch_root / "patch_metadata.csv", metadata_rows)
    if fft_rows:
        _append_csv(patch_root / "fft_features.csv", fft_rows)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(path)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fallback_fields: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _registry_version(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("schema_version", ""))
