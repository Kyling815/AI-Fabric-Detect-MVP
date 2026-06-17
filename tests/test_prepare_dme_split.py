from pathlib import Path

from PIL import Image

from scripts.evaluate_pipeline import collect_manifest_evaluation_images
from scripts.prepare_dme_split import build_dme_split


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_build_dme_split_keeps_reference_normals_out_of_evaluation(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    for index in range(10):
        _write_image(root / "non defective" / f"normal_{index}.jpg")
    for index in range(3):
        _write_image(root / "holes" / f"hole_{index}.jpg")

    split = build_dme_split(root, reference_normal_ratio=0.7, seed=7)
    reference_paths = {item["path"] for item in split["reference_normal"]}
    eval_paths = {item["path"] for item in split["evaluation"]}

    assert len(split["reference_normal"]) == 7
    assert len([item for item in split["evaluation"] if not item["expected_defective"]]) == 3
    assert len([item for item in split["evaluation"] if item["expected_defective"]]) == 3
    assert reference_paths.isdisjoint(eval_paths)


def test_collect_manifest_evaluation_images_limits_per_class(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        """
{
  "evaluation": [
    {"path": "normal_a.jpg", "class_name": "non defective", "expected_defective": false},
    {"path": "normal_b.jpg", "class_name": "non defective", "expected_defective": false},
    {"path": "hole_a.jpg", "class_name": "holes", "expected_defective": true},
    {"path": "hole_b.jpg", "class_name": "holes", "expected_defective": true}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    images = collect_manifest_evaluation_images(manifest_path, max_images_per_class=1)

    assert len(images) == 2
    assert {expected for _, expected in images} == {False, True}
