import csv
from pathlib import Path

from PIL import Image

from scripts.prepare_extended_datasets import discover_company, discover_dme, write_csv


def test_prepare_extended_datasets_writes_dme_manifest_rows(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    normal = root / "non defective"
    defect = root / "holes"
    normal.mkdir(parents=True)
    defect.mkdir(parents=True)
    Image.new("RGB", (12, 8), color=(240, 240, 240)).save(normal / "good.png")
    Image.new("RGB", (10, 6), color=(10, 10, 10)).save(defect / "bad.png")

    rows = discover_dme(root, warnings=[])
    output = tmp_path / "dme_all.csv"
    write_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    assert len(manifest_rows) == 2
    labels = {row["label"] for row in manifest_rows}
    assert labels == {"normal", "defective"}
    defect_row = next(row for row in manifest_rows if row["label"] == "defective")
    assert defect_row["defect_type"] == "holes"
    assert defect_row["width"] == "10"
    assert defect_row["height"] == "6"


def test_prepare_extended_datasets_includes_every_company_image(tmp_path: Path) -> None:
    root = tmp_path / "Company"
    root.mkdir()
    normal = root / "Good_only"
    normal_2 = root / "Good_only(1)"
    normal.mkdir()
    normal_2.mkdir()
    Image.new("RGB", (12, 8), color="black").save(root / "IQC_Stain 1.png")
    Image.new("RGB", (12, 8), color="black").save(root / "unlabeled_capture.png")
    Image.new("RGB", (12, 8), color="white").save(normal / "company_good.png")
    Image.new("RGB", (12, 8), color="white").save(normal_2 / "company_good_2.png")

    rows = discover_company(root, warnings=[], default_label="defective")

    assert len(rows) == 4
    assert [row.label for row in rows].count("defective") == 2
    assert [row.label for row in rows].count("normal") == 2
    assert {row.split_source for row in rows} == {"company_real"}
    assert {row.defect_type for row in rows} == {"", "stain", "company_marked_defect"}


def test_prepare_company_separates_synthetic_test_images_and_masks(tmp_path: Path) -> None:
    root = tmp_path / "Company"
    real_good = root / "real" / "fabric_A" / "good"
    synthetic = root / "fabric_synthetic_testset" / "fabric_A"
    real_good.mkdir(parents=True)
    (synthetic / "original").mkdir(parents=True)
    (synthetic / "synthetic_good").mkdir()
    (synthetic / "synthetic_defect").mkdir()
    (synthetic / "defect_masks").mkdir()
    Image.new("RGB", (12, 8), color="gray").save(
        root / "fabric_synthetic_testset" / "fabric_A_synthetic_good_contact.jpg"
    )
    Image.new("RGB", (12, 8), color="white").save(real_good / "source.png")
    Image.new("RGB", (12, 8), color="white").save(synthetic / "original" / "source.png")
    Image.new("RGB", (12, 8), color="white").save(synthetic / "synthetic_good" / "good_001.png")
    Image.new("RGB", (12, 8), color="black").save(
        synthetic / "synthetic_defect" / "defect_01_hole.png"
    )
    Image.new("L", (12, 8), color="white").save(
        synthetic / "defect_masks" / "defect_01_hole_mask.png"
    )

    rows = discover_company(root, warnings=[])

    assert len(rows) == 3
    assert [row.split_source for row in rows].count("company_real") == 1
    assert [row.split_source for row in rows].count("company_synthetic_test") == 2
    defect = next(row for row in rows if row.defect_type == "hole")
    assert defect.mask_path.endswith("defect_01_hole_mask.png")
    assert defect.fabric_group == "company_iqc_fabric_a"
