import csv
from pathlib import Path

import pytest

from scripts.prepare_company_demo_manifests import (
    development_source_overlap,
    discover_curated_manifest,
    source_leakage,
)


MANIFEST_FIELDS = [
    "subset",
    "quality",
    "file_path",
    "mask_path",
    "defect_category",
    "lot",
    "source_image",
    "crop_box_xywh",
    "fabric_type_estimate",
    "notes",
]


def _write_manifest(root: Path, records: list[dict[str, str]]) -> Path:
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return manifest_path


def _record(
    subset: str,
    quality: str,
    image_path: str,
    source_image: str,
    mask_path: str = "",
) -> dict[str, str]:
    return {
        "subset": subset,
        "quality": quality,
        "file_path": image_path,
        "mask_path": mask_path,
        "defect_category": "hole_or_cut" if quality == "defect" else "",
        "lot": "lot_001",
        "source_image": source_image,
        "crop_box_xywh": "(0, 0, 16, 16)",
        "fabric_type_estimate": "warp_knit_mesh",
        "notes": "test",
    }


def test_curated_manifest_maps_subsets_and_keeps_locked_separate(tmp_path: Path) -> None:
    records = [
        _record("reference_real_good", "good", "images/reference.png", "dev.png"),
        _record(
            "adapter_train",
            "defect",
            "images/adapter.png",
            "dev.png",
            "masks/adapter.png",
        ),
        _record("validation_good", "good", "images/validation_good.png", "val.png"),
        _record(
            "validation",
            "defect",
            "images/validation_defect.png",
            "val.png",
            "masks/validation_defect.png",
        ),
        _record("locked_good", "good", "images/locked_good.png", "locked.png"),
        _record(
            "locked",
            "defect",
            "images/locked_defect.png",
            "locked.png",
            "masks/locked_defect.png",
        ),
    ]
    for record in records:
        image_path = tmp_path / record["file_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        if record["mask_path"]:
            mask_path = tmp_path / record["mask_path"]
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.touch()

    rows = discover_curated_manifest(
        "fabric_001",
        {"display_name": "Fabric A"},
        _write_manifest(tmp_path, records),
    )

    assert [row["split"] for row in rows] == [
        "reference",
        "adapter_train",
        "validation",
        "validation",
        "locked_test",
        "locked_test",
    ]
    assert source_leakage(rows) == []
    assert development_source_overlap(rows) == [
        {
            "fabric_id": "fabric_001",
            "source_original": "dev.png",
            "splits": ["adapter_train", "reference"],
        }
    ]


def test_curated_manifest_rejects_path_outside_fabric_root(tmp_path: Path) -> None:
    fabric_root = tmp_path / "fabric"
    fabric_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.touch()
    record = _record("reference_real_good", "good", "../outside.png", "source.png")

    with pytest.raises(ValueError, match="escapes fabric directory"):
        discover_curated_manifest(
            "fabric_001",
            {"display_name": "Fabric A"},
            _write_manifest(fabric_root, [record]),
        )


def test_source_leakage_flags_development_to_validation_crossing() -> None:
    rows = [
        {"fabric_id": "fabric_001", "source_original": "same.png", "split": "reference"},
        {"fabric_id": "fabric_001", "source_original": "same.png", "split": "validation"},
    ]

    assert source_leakage(rows) == [
        {
            "fabric_id": "fabric_001",
            "source_original": "same.png",
            "splits": ["development", "validation"],
        }
    ]
