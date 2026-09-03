from pathlib import Path

from PIL import Image

from backend.app.ai.patching import build_patch_grid, extract_image_patches
from scripts.train_extended_projection_head import select_masked_defect_patches


def test_select_masked_defect_patches_keeps_only_intersecting_patches(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 48), color="gray")
    grid = build_patch_grid(96, 48, patch_size=48, patch_stride=48)
    patches = extract_image_patches(image, grid)
    mask = Image.new("L", (96, 48), color=0)
    for x in range(60, 70):
        for y in range(10, 20):
            mask.putpixel((x, y), 255)
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path)

    selected = select_masked_defect_patches(
        patches,
        mask_path=mask_path,
        output_size=image.size,
    )

    assert len(selected) == 1
    assert selected[0].box.x == 48
