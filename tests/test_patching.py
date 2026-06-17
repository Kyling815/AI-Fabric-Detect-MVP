from PIL import Image

from backend.app.ai.patching import build_patch_grid, extract_image_patches


def test_build_patch_grid_includes_right_and_bottom_edges() -> None:
    grid = build_patch_grid(
        image_width=220,
        image_height=220,
        patch_size=96,
        patch_stride=64,
    )

    assert grid.rows == 3
    assert grid.cols == 3
    assert grid.patch_count == 9
    assert grid.boxes[-1].x == 124
    assert grid.boxes[-1].y == 124
    assert grid.coverage_ratio == 1.0


def test_extract_image_patches_returns_expected_crop_sizes() -> None:
    image = Image.new("RGB", (180, 120), color=(255, 255, 255))
    grid = build_patch_grid(
        image_width=180,
        image_height=120,
        patch_size=96,
        patch_stride=96,
    )

    patches = extract_image_patches(image, grid)

    assert len(patches) == 4
    assert all(patch.image.size == (96, 96) for patch in patches)
