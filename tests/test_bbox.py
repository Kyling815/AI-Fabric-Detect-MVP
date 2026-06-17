from backend.app.ai.anomaly_detector import PatchAnomaly
from backend.app.ai.bbox import merge_suspicious_patches, scale_boxes_to_original
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
