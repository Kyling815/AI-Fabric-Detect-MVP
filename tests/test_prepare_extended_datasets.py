import csv
from pathlib import Path

from PIL import Image

from scripts.prepare_extended_datasets import discover_dme, write_csv


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
