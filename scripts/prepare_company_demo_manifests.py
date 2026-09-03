import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


FIELDS = [
    "image_id",
    "fabric_id",
    "display_name",
    "image_path",
    "mask_path",
    "label",
    "defect_type",
    "origin",
    "source_original",
    "split",
]

CURATED_SUBSETS = {
    "reference_real_good": ("reference", "good"),
    "validation_good": ("validation", "good"),
    "locked_good": ("locked_test", "good"),
    "adapter_train": ("adapter_train", "defect"),
    "validation": ("validation", "defect"),
    "locked": ("locked_test", "defect"),
}

DEVELOPMENT_SPLITS = {"reference", "adapter_train", "development_only"}


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build source-disjoint Company demo manifests.")
    parser.add_argument("--config", default=settings.fabric_demo_config_path)
    parser.add_argument("--output-dir", default="storage/metadata/company_demo_manifests")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for fabric_id, fabric in sorted(config["fabrics"].items()):
        rows.extend(discover_fabric(fabric_id, fabric, warnings))

    write_csv(output_dir / "all.csv", rows)
    for fabric_id in sorted(config["fabrics"]):
        fabric_rows = [row for row in rows if row["fabric_id"] == fabric_id]
        write_csv(output_dir / f"{fabric_id}_all.csv", fabric_rows)
        for split in ("reference", "adapter_train", "validation", "locked_test", "development_only"):
            write_csv(
                output_dir / f"{fabric_id}_{split}.csv",
                [row for row in fabric_rows if row["split"] == split],
            )

    leakage = source_leakage(rows)
    if leakage:
        raise RuntimeError(f"Source leakage detected: {leakage}")
    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_version": config["dataset_version"],
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "counts": counts(rows),
        "source_leakage": leakage,
        "development_source_overlap": development_source_overlap(rows),
        "warnings": warnings,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def discover_fabric(fabric_id: str, fabric: dict, warnings: list[str]) -> list[dict[str, str]]:
    dataset_mode = fabric.get("dataset_mode", "legacy_directories")
    if dataset_mode == "curated_manifest":
        manifest_value = fabric.get("manifest_path")
        if not manifest_value:
            raise ValueError(f"manifest_path is required for {fabric_id} in curated_manifest mode")
        return discover_curated_manifest(
            fabric_id,
            fabric,
            resolve_project_path(manifest_value),
        )
    if dataset_mode != "legacy_directories":
        raise ValueError(f"Unsupported dataset_mode for {fabric_id}: {dataset_mode}")

    real_dir = resolve_project_path(fabric["real_good_dir"])
    synthetic_dir = resolve_project_path(fabric["synthetic_dir"])
    validation_sources = {value.lower() for value in fabric["validation_sources"]}
    adapter_training_sources = {value.lower() for value in fabric.get("adapter_training_sources", [])}
    test_sources = {value.lower() for value in fabric["locked_test_sources"]}
    overlap = (
        (adapter_training_sources & validation_sources)
        | (adapter_training_sources & test_sources)
        | (validation_sources & test_sources)
    )
    if overlap:
        raise ValueError(f"Validation/test sources overlap for {fabric_id}: {sorted(overlap)}")

    real_images = {
        path.name.lower(): path
        for path in sorted(real_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    rows: list[dict[str, str]] = []
    for source_name, image_path in real_images.items():
        split = source_split(source_name, adapter_training_sources, validation_sources, test_sources)
        rows.append(
            row_for(
                fabric_id=fabric_id,
                display_name=fabric["display_name"],
                image_path=image_path,
                mask_path=None,
                label="normal",
                defect_type="",
                origin="real",
                source_original=source_name,
                split=split,
            )
        )

    package_manifest = synthetic_dir.parent / "manifest.csv"
    with package_manifest.open(newline="", encoding="utf-8") as handle:
        synthetic_records = [
            record
            for record in csv.DictReader(handle)
            if (record.get("fabric_set") or "").strip() == fabric["source_name"]
        ]
    for record in synthetic_records:
        source_name = (record.get("source_original") or "").strip().lower()
        if source_name not in real_images:
            warnings.append(f"{fabric_id}: synthetic source not found in real Good: {source_name}")
        inherited_split = source_split(source_name, adapter_training_sources, validation_sources, test_sources)
        split = inherited_split if inherited_split != "reference" else "development_only"
        class_name = (record.get("class") or "").strip().lower()
        filename = (record.get("filename") or "").strip()
        subdir = "synthetic_defect" if class_name == "defect" else "synthetic_good"
        image_path = synthetic_dir / subdir / filename
        mask_path = synthetic_dir / "defect_masks" / f"{Path(filename).stem}_mask{Path(filename).suffix}"
        rows.append(
            row_for(
                fabric_id=fabric_id,
                display_name=fabric["display_name"],
                image_path=image_path,
                mask_path=mask_path if class_name == "defect" and mask_path.exists() else None,
                label="defective" if class_name == "defect" else "normal",
                defect_type=(record.get("defect_type") or "").strip(),
                origin="synthetic",
                source_original=source_name,
                split=split,
            )
        )
    return rows


def discover_curated_manifest(
    fabric_id: str,
    fabric: dict,
    manifest_path: Path,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    fabric_root = manifest_path.parent.resolve()
    required_fields = {
        "subset",
        "quality",
        "file_path",
        "mask_path",
        "defect_category",
        "source_image",
    }
    rows: list[dict[str, str]] = []
    seen_images: set[Path] = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Missing fields in curated manifest for {fabric_id}: {sorted(missing_fields)}"
            )
        for line_number, record in enumerate(reader, start=2):
            subset = (record.get("subset") or "").strip().lower()
            quality = (record.get("quality") or "").strip().lower()
            if subset not in CURATED_SUBSETS:
                raise ValueError(
                    f"Unsupported subset in {manifest_path}:{line_number}: {subset!r}"
                )
            split, expected_quality = CURATED_SUBSETS[subset]
            if quality != expected_quality:
                raise ValueError(
                    f"Quality/subset mismatch in {manifest_path}:{line_number}: "
                    f"{subset!r} requires {expected_quality!r}, got {quality!r}"
                )

            image_path = resolve_manifest_member(
                fabric_root,
                record.get("file_path"),
                manifest_path,
                line_number,
                "file_path",
            )
            if image_path in seen_images:
                raise ValueError(f"Duplicate file_path in {manifest_path}:{line_number}: {image_path}")
            seen_images.add(image_path)

            mask_value = (record.get("mask_path") or "").strip()
            mask_path = (
                resolve_manifest_member(
                    fabric_root,
                    mask_value,
                    manifest_path,
                    line_number,
                    "mask_path",
                )
                if mask_value
                else None
            )
            if quality == "defect" and mask_path is None:
                raise ValueError(f"Defect row has no mask in {manifest_path}:{line_number}")
            if quality == "good" and mask_path is not None:
                raise ValueError(f"Good row unexpectedly has a mask in {manifest_path}:{line_number}")

            source_name = (record.get("source_image") or "").strip().lower()
            if not source_name:
                raise ValueError(f"Missing source_image in {manifest_path}:{line_number}")
            defect_type = (record.get("defect_category") or "").strip()
            if quality == "defect" and not defect_type:
                raise ValueError(f"Defect row has no defect_category in {manifest_path}:{line_number}")

            rows.append(
                row_for(
                    fabric_id=fabric_id,
                    display_name=fabric["display_name"],
                    image_path=image_path,
                    mask_path=mask_path,
                    label="defective" if quality == "defect" else "normal",
                    defect_type=defect_type,
                    origin="synthetic" if quality == "defect" else "real",
                    source_original=source_name,
                    split=split,
                )
            )
    return rows


def resolve_manifest_member(
    root: Path,
    raw_value: str | None,
    manifest_path: Path,
    line_number: int,
    field_name: str,
) -> Path:
    value = (raw_value or "").strip()
    if not value:
        raise ValueError(f"Missing {field_name} in {manifest_path}:{line_number}")
    path = (root / Path(value)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} escapes fabric directory in {manifest_path}:{line_number}: {value}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_split(
    source: str,
    adapter_training_sources: set[str],
    validation_sources: set[str],
    test_sources: set[str],
) -> str:
    lowered = source.lower()
    if lowered in adapter_training_sources:
        return "adapter_train"
    if lowered in validation_sources:
        return "validation"
    if lowered in test_sources:
        return "locked_test"
    return "reference"


def row_for(**values) -> dict[str, str]:
    image_path: Path = values.pop("image_path")
    mask_path: Path | None = values.pop("mask_path")
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    digest = hashlib.sha1(str(image_path).lower().encode("utf-8")).hexdigest()[:16]
    return {
        "image_id": digest,
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path else "",
        **{key: str(value) for key, value in values.items()},
    }


def source_leakage(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_source: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        split = row["split"]
        partition = "development" if split in DEVELOPMENT_SPLITS else split
        by_source.setdefault((row["fabric_id"], row["source_original"]), set()).add(partition)
    return [
        {"fabric_id": key[0], "source_original": key[1], "splits": sorted(splits)}
        for key, splits in by_source.items()
        if len(splits) > 1
    ]


def development_source_overlap(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_source: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row["split"] not in DEVELOPMENT_SPLITS:
            continue
        by_source.setdefault((row["fabric_id"], row["source_original"]), set()).add(row["split"])
    return [
        {"fabric_id": key[0], "source_original": key[1], "splits": sorted(splits)}
        for key, splits in sorted(by_source.items())
        if len(splits) > 1
    ]


def counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter = Counter(
        f"{row['fabric_id']}:{row['split']}:{row['origin']}:{row['label']}" for row in rows
    )
    return dict(sorted(counter.items()))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
