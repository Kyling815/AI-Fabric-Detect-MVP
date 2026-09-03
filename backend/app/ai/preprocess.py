from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageReadResult:
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class PreprocessPlanResult:
    target_max_size: int
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float
    patch_size: int
    patch_stride: int
    color_mode: str = "RGB"


class ImageReadError(ValueError):
    """Raised when an uploaded image cannot be decoded."""


def inspect_image_file(path: Path) -> ImageReadResult:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            return ImageReadResult(width=oriented.width, height=oriented.height, mode=oriented.mode)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageReadError(f"File is not a readable image: {path}") from exc


def build_preprocess_plan(
    width: int,
    height: int,
    target_max_size: int,
    patch_size: int,
    patch_stride: int,
) -> PreprocessPlanResult:
    longest_side = max(width, height)
    if longest_side <= 0:
        raise ValueError("Image dimensions must be positive.")

    scale = min(1.0, target_max_size / longest_side)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    return PreprocessPlanResult(
        target_max_size=target_max_size,
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale=scale,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )


def load_resized_rgb_image(path: Path, plan: PreprocessPlanResult) -> Image.Image:
    with Image.open(path) as image:
        rgb_image = ImageOps.exif_transpose(image).convert(plan.color_mode)
        if rgb_image.size == (plan.resized_width, plan.resized_height):
            return rgb_image.copy()
        return rgb_image.resize((plan.resized_width, plan.resized_height), Image.Resampling.BILINEAR)
