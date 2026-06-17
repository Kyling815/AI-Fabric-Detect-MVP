from pathlib import Path

from PIL import Image

from scripts.evaluate_wfdd_fabrics import collect_wfdd_test_images


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_collect_wfdd_test_images_labels_good_as_normal(tmp_path: Path) -> None:
    fabric_dir = tmp_path / "grey_cloth"
    _write_image(fabric_dir / "test" / "good" / "normal_1.png")
    _write_image(fabric_dir / "test" / "defect" / "defect_1.png")

    images = collect_wfdd_test_images(fabric_dir, max_images_per_class=None)

    assert len(images) == 2
    labels = {path.parent.name: expected for path, expected in images}
    assert labels["good"] is False
    assert labels["defect"] is True


def test_collect_wfdd_test_images_limits_each_class(tmp_path: Path) -> None:
    fabric_dir = tmp_path / "grid_cloth"
    for index in range(3):
        _write_image(fabric_dir / "test" / "good" / f"normal_{index}.png")
        _write_image(fabric_dir / "test" / "defect" / f"defect_{index}.png")

    images = collect_wfdd_test_images(fabric_dir, max_images_per_class=1)

    assert len(images) == 2
