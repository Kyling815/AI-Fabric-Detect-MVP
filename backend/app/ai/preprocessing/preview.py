"""Generate visual before/after preprocessing-v2 comparison grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.ai.preprocessing.pipeline import _write_image, preprocess_path
from backend.app.ai.preprocessing.quality import SUPPORTED_IMAGE_EXTENSIONS, load_image
from backend.app.ai.preprocessing.texture_view import generate_texture_view


def create_preview(source_path: Path, output_path: Path, config: PreprocessConfig) -> bool:
    original, _ = load_image(source_path)
    result = preprocess_path(source_path, config)
    if original is None or not result.accepted or result.processed_image is None:
        return False
    target_height = 320
    panels = [
        _fit_panel(original, target_height),
        _fit_panel(
            result.illumination_corrected_image
            if result.illumination_corrected_image is not None
            else original,
            target_height,
        ),
        _fit_panel(result.processed_image, target_height),
        _fit_panel(
            result.texture_view
            if result.texture_view is not None
            else generate_texture_view(result.processed_image),
            target_height,
        ),
    ]
    panel_width = max(panel.shape[1] for panel in panels)
    padded = [
        cv2.copyMakeBorder(panel, 32, 0, 0, panel_width - panel.shape[1], cv2.BORDER_CONSTANT)
        for panel in panels
    ]
    labels = ("Original", "Illumination corrected", "Final normalized RGB", "Texture view")
    for panel, label in zip(padded, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.concatenate(padded, axis=1)
    metrics = result.quality_metrics
    cv2.putText(
        grid,
        f"{source_path.name} | blur={metrics.laplacian_variance:.1f} dark={metrics.dark_ratio:.3f} bright={metrics.bright_ratio:.3f}",
        (8, grid.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    _write_image(output_path, grid)
    return True


def _fit_panel(image: np.ndarray, target_height: int) -> np.ndarray:
    scale = target_height / image.shape[0]
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), target_height), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create preprocessing-v2 visual comparison grids.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("storage/annotated/preprocessing_v2_preview"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    config = PreprocessConfig.load(args.config) if args.config else PreprocessConfig(generate_texture_view=True)
    sources = (
        [args.input]
        if args.input.is_file()
        else sorted(
            (path for path in args.input.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
            key=lambda item: str(item).casefold(),
        )[: args.limit]
    )
    created = sum(
        create_preview(path, args.output / f"{path.stem}_preview.png", config)
        for path in sources
    )
    print(f"Created {created} preview(s) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
