import argparse
import csv
import hashlib
import json
import re
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

    company_train_rows = [
        row for row in company_rows if row.split_source in {"company_real", "company_synthetic_train"}
    ]
    company_synthetic_rows = [
        row for row in company_rows if row.split_source.startswith("company_synthetic_")
    ]
    company_synthetic_train_rows = [
        row for row in company_rows if row.split_source == "company_synthetic_train"
    ]
    company_synthetic_test_rows = [
        row for row in company_rows if row.split_source == "company_synthetic_test"
    ]
    company_reference_rows = [
        row for row in company_rows if row.label == "normal" and row.split_source == "company_real"
    ]
    company_augmented_reference_rows = [row for row in company_train_rows if row.label == "normal"]

    write_csv(output_dir / "company_all.csv", company_rows)
    write_csv(output_dir / "company_train.csv", company_train_rows)
    write_csv(output_dir / "company_synthetic_all.csv", company_synthetic_rows)
    write_csv(output_dir / "company_synthetic_train.csv", company_synthetic_train_rows)
    write_csv(output_dir / "company_synthetic_test.csv", company_synthetic_test_rows)
    write_csv(output_dir / "company_reference_candidates.csv", company_reference_rows)
    write_csv(
        output_dir / "company_augmented_reference_candidates.csv",
        company_augmented_reference_rows,
    )
    write_csv(output_dir / "dme_all.csv", dme_rows)
    write_csv(output_dir / "wfdd_all.csv", wfdd_rows)
    write_csv(output_dir / "lusitano_all.csv", lusitano_rows)
    write_csv(output_dir / "extended_all.csv", all_rows)
    write_csv(output_dir / "raw_all.csv", all_rows)
    write_csv(
        output_dir / "extended_reference_candidates.csv",
        [
            row
            for row in all_rows
            if row.label == "normal" and not row.split_source.startswith("company_synthetic_")
        ],
    )
    write_csv(
        output_dir / "extended_eval_candidates.csv",
        [
            row
            for row in all_rows
            if row.split_source in {
                "company_real",
                "company_synthetic_test",
                "dme_class_folder",
                "wfdd_test",
                "lusitano_recursive",
            }
        ],
    )

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_dir": str(output_dir),
        "counts": {
            "company": len(company_rows),
            "company_train": len(company_train_rows),
            "company_synthetic_train": len(company_synthetic_train_rows),
            "company_synthetic_test": len(company_synthetic_test_rows),
            "company_reference_candidates": len(company_reference_rows),
            "company_augmented_reference_candidates": len(company_augmented_reference_rows),
            "dme": len(dme_rows),
            "wfdd": len(wfdd_rows),
            "lusitano": len(lusitano_rows),
            "extended_all": len(all_rows),
            "reference_candidates": sum(1 for row in all_rows if row.label == "normal"),
            "eval_candidates": sum(
                1
                for row in all_rows
                if row.split_source in {
                    "company_real",
                    "company_synthetic_test",
                    "dme_class_folder",
                    "wfdd_test",
                    "lusitano_recursive",
                }
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
    synthetic_splits = _load_synthetic_company_splits(root, warnings)
    for image_path in list_images(root):
        relative_parts = image_path.relative_to(root).parts
        normalized_parts = [part.strip().lower() for part in relative_parts]
        if "defect_masks" in normalized_parts:
            continue
        is_synthetic_tree = "fabric_synthetic_testset" in normalized_parts
        if is_synthetic_tree and "original" in normalized_parts:
            # These are byte-identical provenance copies of real Company
            # images. Keep them on disk, but never duplicate them in a model
            # or evaluation manifest.
            continue

        synthetic = _synthetic_company_metadata(image_path, root, synthetic_splits)
        if is_synthetic_tree and synthetic is None:
            # Contact sheets and other package-level preview assets are
            # documentation, not individual model samples.
            continue
        if synthetic is None:
            label, defect_type = infer_company_label(image_path, root, default_label)
            fabric_group = _company_fabric_group(image_path, root)
            split_source = "company_real"
            mask_path = None
        else:
            label, defect_type, fabric_group, mask_path, split_source = synthetic
        if label == "unknown":
            warnings.append(f"Could not infer Company label for: {image_path}")
        rows.append(
            build_row(
                image_path=image_path,
                dataset_name="company_iqc",
                fabric_group=fabric_group,
                label=label,
                defect_type=defect_type,
                split_source=split_source,
                mask_path=mask_path,
            )
        )
    return rows


def _synthetic_company_metadata(
    image_path: Path,
    root: Path,
    synthetic_splits: dict[tuple[str, str], str],
) -> tuple[str, str, str, Path | None, str] | None:
    parts = image_path.relative_to(root).parts
    lowered = [part.lower() for part in parts]
    try:
        synthetic_index = lowered.index("fabric_synthetic_testset")
    except ValueError:
        return None
    if len(parts) <= synthetic_index + 2:
        return None

    fabric_name = parts[synthetic_index + 1]
    class_name = lowered[synthetic_index + 2]
    fabric_group = f"company_iqc_{fabric_name.lower()}"
    split_source = synthetic_splits.get(
        (fabric_name.lower(), image_path.name.lower()),
        "company_synthetic_test",
    )
    if class_name == "synthetic_good":
        return "normal", "", fabric_group, None, split_source
    if class_name != "synthetic_defect":
        return None

    stem = re.sub(r"^defect_\d+_?", "", image_path.stem.lower()).strip("_")
    mask_name = f"{image_path.stem}_mask{image_path.suffix}"
    mask_path = root / "fabric_synthetic_testset" / fabric_name / "defect_masks" / mask_name
    return (
        "defective",
        stem or "synthetic_defect",
        fabric_group,
        mask_path if mask_path.exists() else None,
        split_source,
    )


def _load_synthetic_company_splits(
    root: Path,
    warnings: list[str],
) -> dict[tuple[str, str], str]:
    manifest_path = root / "fabric_synthetic_testset" / "manifest.csv"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        warnings.append(f"Could not read Company synthetic manifest: {exc}")
        return {}

    sources_by_fabric: dict[str, set[str]] = {}
    for record in records:
        fabric = (record.get("fabric_set") or "").strip().lower()
        source = (record.get("source_original") or "").strip().lower()
        if fabric and source:
            sources_by_fabric.setdefault(fabric, set()).add(source)

    held_out_sources = {
        (fabric, source)
        for fabric, sources in sources_by_fabric.items()
        for index, source in enumerate(sorted(sources))
        if index % 5 == 0
    }
    splits: dict[tuple[str, str], str] = {}
    for record in records:
        fabric = (record.get("fabric_set") or "").strip().lower()
        filename = (record.get("filename") or "").strip().lower()
        source = (record.get("source_original") or "").strip().lower()
        if not fabric or not filename:
            continue
        splits[(fabric, filename)] = (
            "company_synthetic_test"
            if (fabric, source) in held_out_sources
            else "company_synthetic_train"
        )
    return splits


def _company_fabric_group(image_path: Path, root: Path) -> str:
    parts = image_path.relative_to(root).parts
    lowered = [part.lower() for part in parts]
    for index, part in enumerate(lowered[:-1]):
        if part == "real" and index + 1 < len(parts) - 1:
            fabric_name = parts[index + 1].strip().lower()
            if fabric_name != "unclassified":
                return f"company_iqc_{fabric_name}"
    return "company_iqc_unclassified"


def infer_company_label(image_path: Path, root: Path, default_label: str) -> tuple[str, str]:
    relative_parts = image_path.relative_to(root).parts[:-1]
    if any(_is_company_normal_folder(part) for part in relative_parts):
        return "normal", ""
    comparable_parts = {
        re.sub(r"[\s-]+", "_", part.strip().lower()).strip("_")
        for part in relative_parts
    }
    if comparable_parts & {name.replace(" ", "_") for name in DEFECT_NAMES}:
        return "defective", _company_defect_type(image_path)

    stem = image_path.stem.lower()
    if stem.startswith("iqc_"):
        return "defective", _company_defect_type(image_path)
    return default_label, "company_marked_defect" if default_label == "defective" else ""


def _is_company_normal_folder(name: str) -> bool:
    normalized = re.sub(r"\(\d+\)$", "", name.strip().lower())
    normalized = re.sub(r"[\s-]+", "_", normalized).strip("_")
    normalized_names = {item.replace(" ", "_") for item in NORMAL_NAMES}
    return normalized in normalized_names or normalized.startswith("good_")


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
