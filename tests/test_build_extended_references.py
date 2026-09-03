import csv
from pathlib import Path

from scripts.build_extended_references import reference_images_from_manifest


def test_reference_images_from_manifest_uses_only_matching_normal_rows(tmp_path: Path) -> None:
    normal = tmp_path / "normal.png"
    defect = tmp_path / "defect.png"
    normal.write_bytes(b"normal")
    defect.write_bytes(b"defect")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "dataset_name", "fabric_group"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(normal),
                "label": "normal",
                "dataset_name": "company_iqc",
                "fabric_group": "company_iqc",
            }
        )
        writer.writerow(
            {
                "image_path": str(defect),
                "label": "defective",
                "dataset_name": "company_iqc",
                "fabric_group": "company_iqc",
            }
        )

    assert reference_images_from_manifest(
        manifest,
        dataset_name="company_iqc",
        fabric_type=None,
    ) == [normal]
