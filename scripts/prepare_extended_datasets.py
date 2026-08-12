import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


NORMAL_NAMES = {"good", "normal", "non defective", "non_defective", "nondefective", "nondefects", "ok"}
DEFECT_NAMES = {"defect", "defective", "bad", "broken thread", "damage", "holes", "stain"}


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    dataset_name: str
    fabric_group: str
    image_path: str
    mask_path: str
    label: str
    defect_type: str
    split_source: str
    width: int
    height: int
    file_ext: str


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Create one manifest from all fabric photos in Data/raw/Company and Data/raw/external."
    )
    parser.add_argument("--company-root", default=settings.raw_company_dir)
    parser.add_argument("--dme-root", default=settings.dme_dataset_dir)
    parser.add_argument("--wfdd-root", default=settings.wfdd_dataset_dir)
    parser.add_argument("--lusitano-root", default=settings.lusitano_dataset_dir)
    parser.add_argument(
        "--company-default-label",
        choices=("normal", "defective", "unknown"),
        default="defective",
        help="Label for loose Company images whose filename/folder does not expose a label.",
    )
    parser.add_argument("--output-dir", default="storage/metadata/extended_dataset_manifests")
    args = parser.parse_args()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    company_rows = discover_company(
        resolve_project_path(args.company_root),
        warnings,
        default_label=args.company_default_label,
    )
    dme_rows = discover_dme(resolve_project_path(args.dme_root), warnings)
    wfdd_rows = discover_wfdd(resolve_project_path(args.wfdd_root), warnings)
    lusitano_rows = discover_lusitano(resolve_project_path(args.lusitano_root), warnings)
    all_rows = company_rows + dme_rows + wfdd_rows + lusitano_rows

    write_csv(output_dir / "company_all.csv", company_rows)
    write_csv(output_dir / "dme_all.csv", dme_rows)
    write_csv(output_dir / "wfdd_all.csv", wfdd_rows)
    write_csv(output_dir / "lusitano_all.csv", lusitano_rows)
    write_csv(output_dir / "extended_all.csv", all_rows)
    write_csv(output_dir / "raw_all.csv", all_rows)
    write_csv(output_dir / "extended_reference_candidates.csv", [row for row in all_rows if row.label == "normal"])
    write_csv(
        output_dir / "extended_eval_candidates.csv",
        [
            row
            for row in all_rows
            if row.split_source in {"company_recursive", "dme_class_folder", "wfdd_test", "lusitano_recursive"}
        ],
    )

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_dir": str(output_dir),
        "counts": {
            "company": len(company_rows),
            "dme": len(dme_rows),
            "wfdd": len(wfdd_rows),
            "lusitano": len(lusitano_rows),
            "extended_all": len(all_rows),
            "reference_candidates": sum(1 for row in all_rows if row.label == "normal"),
            "eval_candidates": sum(
                1
                for row in all_rows
                if row.split_source in {"company_recursive", "dme_class_folder", "wfdd_test", "lusitano_recursive"}
            ),
        },
        "warnings": warnings,
    }
    (output_dir / "manifest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def discover_company(
    root: Path,
    warnings: list[str],
    default_label: str = "defective",
) -> list[ManifestRow]:
    if not root.exists():
        warnings.append(f"Company root not found: {root}")
        return []

    rows: list[ManifestRow] = []
    for image_path in list_images(root):
        label, defect_type = infer_company_label(image_path, root, default_label)
        if label == "unknown":
            warnings.append(f"Could not infer Company label for: {image_path}")
        rows.append(
            build_row(
                image_path=image_path,
                dataset_name="company_iqc",
                fabric_group="company_iqc",
                label=label,
                defect_type=defect_type,
                split_source="company_recursive",
            )
        )
    return rows


def infer_company_label(image_path: Path, root: Path, default_label: str) -> tuple[str, str]:
    relative_parts = {
        part.lower().replace("-", "_")
        for part in image_path.relative_to(root).parts[:-1]
    }
    comparable_parts = relative_parts | {part.replace("_", " ") for part in relative_parts}
    if comparable_parts & NORMAL_NAMES:
        return "normal", ""
    if comparable_parts & DEFECT_NAMES:
        return "defective", _company_defect_type(image_path)

    stem = image_path.stem.lower()
    if stem.startswith("iqc_"):
        return "defective", _company_defect_type(image_path)
    return default_label, "company_marked_defect" if default_label == "defective" else ""


def _company_defect_type(image_path: Path) -> str:
    stem = image_path.stem.lower().removeprefix("iqc_")
    stem = "_".join(part for part in stem.replace("-", "_").split() if not part.isdigit())
    return stem.strip("_") or "company_defect"


def discover_dme(root: Path, warnings: list[str]) -> list[ManifestRow]:
    class_map = {
        "non defective": ("normal", ""),
        "broken thread": ("defective", "broken_thread"),
        "damage": ("defective", "damage"),
        "holes": ("defective", "holes"),
        "stain": ("defective", "stain"),
    }
    if not root.exists():
        warnings.append(f"DME root not found: {root}")
        return []

    rows: list[ManifestRow] = []
    for class_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        label, defect_type = class_map.get(class_dir.name.lower(), ("unknown", class_dir.name.lower().replace(" ", "_")))
        for image_path in list_images(class_dir):
            rows.append(
                build_row(
                    image_path=image_path,
                    dataset_name="dme_20250129",
                    fabric_group="dme_plain",
                    label=label,
                    defect_type=defect_type,
                    split_source="dme_class_folder",
                )
            )
    return rows


def discover_wfdd(root: Path, warnings: list[str]) -> list[ManifestRow]:
    if not root.exists():
        warnings.append(f"WFDD root not found: {root}")
        return []

    rows: list[ManifestRow] = []
    for group_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        fabric_group = f"wfdd_{group_dir.name}"
        train_good = group_dir / "train" / "good"
        for image_path in list_images(train_good):
            rows.append(
                build_row(
                    image_path=image_path,
                    dataset_name="wfdd",
                    fabric_group=fabric_group,
                    label="normal",
                    defect_type="",
                    split_source="wfdd_train_good",
                )
            )

        test_dir = group_dir / "test"
        test_class_dirs = (
            sorted((item for item in test_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower())
            if test_dir.exists()
            else []
        )
        for class_dir in test_class_dirs:
            label = "normal" if class_dir.name.lower() == "good" else "defective"
            defect_type = "" if label == "normal" else class_dir.name.lower().replace(" ", "_")
            for image_path in list_images(class_dir):
                rows.append(
                    build_row(
                        image_path=image_path,
                        dataset_name="wfdd",
                        fabric_group=fabric_group,
                        label=label,
                        defect_type=defect_type,
                        split_source="wfdd_test",
                        mask_path=find_wfdd_mask(group_dir, defect_type, image_path) if label == "defective" else None,
                    )
                )
    return rows


def discover_lusitano(root: Path, warnings: list[str]) -> list[ManifestRow]:
    if not root.exists():
        warnings.append(f"Lusitano root not found: {root}")
        return []

    rows: list[ManifestRow] = []
    for image_path in list_images(root):
        relative_parts = [part.lower() for part in image_path.relative_to(root).parts[:-1]]
        label, defect_type = infer_lusitano_label(relative_parts)
        if label == "unknown":
            warnings.append(f"Could not infer Lusitano label for: {image_path}")
        rows.append(
            build_row(
                image_path=image_path,
                dataset_name="lusitano",
                fabric_group="lusitano_group_auto",
                label=label,
                defect_type=defect_type,
                split_source="lusitano_recursive",
            )
        )
    return rows


def infer_lusitano_label(parts: list[str]) -> tuple[str, str]:
    joined = " ".join(parts).replace("-", "_")
    if any(name in joined for name in NORMAL_NAMES):
        return "normal", ""
    for name in DEFECT_NAMES:
        if name in joined:
            return "defective", name.replace(" ", "_")
    return "unknown", ""


def find_wfdd_mask(group_dir: Path, defect_type: str, image_path: Path) -> Path | None:
    mask_dir = group_dir / "ground_truth" / defect_type
    if not mask_dir.exists():
        return None
    stem = image_path.stem
    for candidate in mask_dir.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.stem.startswith(stem):
            return candidate
    return None


def build_row(
    image_path: Path,
    dataset_name: str,
    fabric_group: str,
    label: str,
    defect_type: str,
    split_source: str,
    mask_path: Path | None = None,
) -> ManifestRow:
    width, height = read_dimensions(image_path)
    return ManifestRow(
        image_id=stable_image_id(image_path),
        dataset_name=dataset_name,
        fabric_group=fabric_group,
        image_path=str(image_path),
        mask_path=str(mask_path) if mask_path else "",
        label=label,
        defect_type=defect_type,
        split_source=split_source,
        width=width,
        height=height,
        file_ext=image_path.suffix.lower(),
    )


def read_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return 0, 0


def stable_image_id(image_path: Path) -> str:
    digest = hashlib.sha1(str(image_path).lower().encode("utf-8")).hexdigest()[:16]
    return digest


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def write_csv(path: Path, rows: list[ManifestRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()
