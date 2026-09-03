import numpy as np

from backend.app.ai.anomaly_detector import PatchAnomaly
from backend.app.ai.bbox import (
    localize_pixel_defects,
    localize_suspicious_patches,
    merge_suspicious_patches,
    scale_boxes_to_original,
)
from backend.app.ai.patching import PatchBox


def test_merge_suspicious_patches_groups_touching_boxes() -> None:
    patches = [
        PatchAnomaly(0, 10, 0.2, 0.8, True, PatchBox(x=0, y=0, w=32, h=32)),
        PatchAnomaly(1, 11, 0.2, 0.9, True, PatchBox(x=32, y=0, w=32, h=32)),
        PatchAnomaly(2, 12, 0.9, 0.1, False, PatchBox(x=96, y=0, w=32, h=32)),
    ]

    boxes = merge_suspicious_patches(patches)

    assert len(boxes) == 1
    assert boxes[0].x == 0
    assert boxes[0].y == 0
    assert boxes[0].w == 64
    assert boxes[0].h == 32
    assert boxes[0].score == 0.9
    assert boxes[0].patch_count == 2


def test_scale_boxes_to_original_uses_inverse_resize_scale() -> None:
    boxes = merge_suspicious_patches(
        [PatchAnomaly(0, 10, 0.2, 0.8, True, PatchBox(x=10, y=20, w=30, h=40))]
    )

    scaled = scale_boxes_to_original(boxes, scale=0.5)

    assert scaled[0].x == 20
    assert scaled[0].y == 40
    assert scaled[0].w == 60
    assert scaled[0].h == 80


def test_localization_refines_dense_full_frame_activation_to_score_peak() -> None:
    patches = []
    patch_index = 0
    for row, y in enumerate(range(0, 50, 10)):
        for col, x in enumerate(range(0, 50, 10)):
            anomaly_score = 1.0
            if 1 <= row <= 3 and 1 <= col <= 3:
                anomaly_score = 1.2
            if row == 2 and col == 2:
                anomaly_score = 2.0
            patches.append(
                PatchAnomaly(
                    patch_index,
                    patch_index,
                    1.0 - anomaly_score,
                    anomaly_score,
                    True,
                    PatchBox(x=x, y=y, w=20, h=20),
                )
            )
            patch_index += 1

    boxes = localize_suspicious_patches(
        patches,
        image_width=60,
        image_height=60,
        min_patch_count=6,
    )

    assert len(boxes) == 1
    assert boxes[0].x == 15
    assert boxes[0].y == 15
    assert boxes[0].w == 30
    assert boxes[0].h == 30
    assert boxes[0].patch_count == 9
    assert boxes[0].label == "localized_defect"


def test_localization_does_not_bridge_across_a_missing_grid_cell() -> None:
    patches = [
        PatchAnomaly(0, 0, 0.0, 0.8, True, PatchBox(x=0, y=0, w=20, h=20)),
        PatchAnomaly(1, 1, 0.0, 0.1, False, PatchBox(x=10, y=0, w=20, h=20)),
        PatchAnomaly(2, 2, 0.0, 0.9, True, PatchBox(x=20, y=0, w=20, h=20)),
    ]

    boxes = localize_suspicious_patches(
        patches,
        image_width=40,
        image_height=20,
        min_patch_count=1,
    )

    assert len(boxes) == 2
    assert [(box.x, box.w) for box in boxes] == [(0, 15), (25, 15)]


def test_pixel_localization_keeps_dominant_compact_residual() -> None:
    image = np.full((160, 120, 3), 120, dtype=np.uint8)
    image[70:78, 45:55] = 245

    boxes = localize_pixel_defects(image, anomaly_score=1.2)

    assert len(boxes) == 1
    assert boxes[0].x <= 45
    assert boxes[0].y <= 70
    assert boxes[0].x + boxes[0].w >= 55
    assert boxes[0].y + boxes[0].h >= 78
    assert boxes[0].w < image.shape[1] // 2
    assert boxes[0].h < image.shape[0] // 2
    assert boxes[0].label == "pixel_localized_defect"


def test_pixel_localization_merges_a_long_vertical_yarn_defect() -> None:
    rng = np.random.default_rng(42)
    texture = rng.normal(120, 8, (240, 180)).clip(0, 255).astype(np.uint8)
    image = np.repeat(texture[:, :, None], 3, axis=2)
    image[:, 86:91] = np.clip(image[:, 86:91].astype(np.int16) + 55, 0, 255)

    boxes = localize_pixel_defects(image, anomaly_score=1.1)

    assert len(boxes) == 1
    assert boxes[0].x <= 86
    assert boxes[0].x + boxes[0].w >= 91
    assert boxes[0].w < image.shape[1] // 4
    assert boxes[0].h >= image.shape[0] * 0.9
    assert boxes[0].label == "elongated_pixel_defect"


def test_pixel_localization_does_not_box_uniform_texture() -> None:
    rng = np.random.default_rng(7)
    texture = rng.normal(120, 8, (180, 160)).clip(0, 255).astype(np.uint8)
    image = np.repeat(texture[:, :, None], 3, axis=2)

    assert localize_pixel_defects(image, anomaly_score=1.1) == []
