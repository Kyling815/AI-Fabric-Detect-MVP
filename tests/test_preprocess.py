from pathlib import Path

import pytest
from PIL import Image

from backend.app.ai.preprocess import (
    ImageReadError,
    build_preprocess_plan,
    inspect_image_file,
    load_resized_rgb_image,
)


def test_inspect_image_file_reads_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 160), color=(255, 255, 255)).save(image_path)

    metadata = inspect_image_file(image_path)

    assert metadata.width == 320
    assert metadata.height == 160
    assert metadata.mode == "RGB"


def test_inspect_image_file_rejects_invalid_image(tmp_path: Path) -> None:
    image_path = tmp_path / "invalid.png"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageReadError):
        inspect_image_file(image_path)


def test_image_read_and_resize_apply_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20), color=(20, 40, 60)).save(image_path, exif=exif)

    metadata = inspect_image_file(image_path)
    plan = build_preprocess_plan(
        width=metadata.width,
        height=metadata.height,
        target_max_size=100,
        patch_size=16,
        patch_stride=8,
    )
    resized = load_resized_rgb_image(image_path, plan)

    assert (metadata.width, metadata.height) == (20, 40)
    assert resized.size == (20, 40)


def test_build_preprocess_plan_downscales_long_side() -> None:
    plan = build_preprocess_plan(
        width=1600,
        height=800,
        target_max_size=800,
        patch_size=96,
        patch_stride=48,
    )

    assert plan.resized_width == 800
    assert plan.resized_height == 400
    assert plan.scale == 0.5
    assert plan.patch_size == 96
