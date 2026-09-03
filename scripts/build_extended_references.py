import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import DatasetService
from backend.app.services.reference_registry import ReferenceRegistry, ReferenceRegistryEntry


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build separate reference libraries from every available normal set.")
    parser.add_argument("--variant", default="v1", help="Reference id suffix, e.g. v1 or extended_v1.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap per reference. Omit to use every normal image.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete, compatible references already registered for this variant.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show what would be skipped/built without extracting embeddings.",
    )
    lusitano_group = parser.add_mutually_exclusive_group()
    lusitano_group.add_argument("--include-lusitano", dest="include_lusitano", action="store_true")
    lusitano_group.add_argument("--exclude-lusitano", dest="include_lusitano", action="store_false")
    parser.set_defaults(include_lusitano=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--only-reference",
        default=None,
        help="Build only one reference family, e.g. company_iqc, dme_plain or wfdd_grey_cloth.",
    )
    parser.add_argument(
        "--projection-head-path",
        default=None,
        help="Override FABRIC_PROJECTION_HEAD_PATH for this build.",
    )
    parser.add_argument(
        "--reference-manifest-csv",
        default=None,
        help="Optional CSV whose normal rows define the exact reference image set.",
    )
    args = parser.parse_args()

    datasets = DatasetService()
    paths = datasets.list_configured_paths()
    references = [
        {
            "reference_id": reference_id("dme_plain", args.variant),
            "dataset_name": "dme_20250129",
            "fabric_type": None,
            "normal_path": paths["dme_20250129"] / "non defective",
            "source_layer": "raw_dataset",
        },
        *[
            {
                "reference_id": reference_id(f"wfdd_{fabric}", args.variant),
                "dataset_name": "wfdd",
                "fabric_type": fabric,
                "normal_path": paths["wfdd"] / fabric / "train" / "good",
                "source_layer": "raw_dataset",
            }
            for fabric in ["grey_cloth", "grid_cloth", "pink_flower", "yellow_cloth"]
        ],
    ]
    if args.include_lusitano:
        lusitano_normal = resolve_project_path(settings.lusitano_dataset_dir) / "nondefects"
        references.append(
            {
                "reference_id": reference_id("lusitano_group_auto", args.variant),
                "dataset_name": "lusitano",
                "fabric_type": "group_auto",
                "normal_path": lusitano_normal,
                "source_layer": "raw_dataset",
            }
        )

    company_normal = datasets.resolve_reference_normal_path("company_iqc", None, None)
    if company_normal is not None:
        company_normal_dirs = datasets._company_normal_dirs(paths["raw_company"])
        references.append(
            {
                "reference_id": reference_id("company_iqc", args.variant),
                "dataset_name": "company_iqc",
                "fabric_type": None,
                "normal_path": paths["raw_company"],
                "normal_paths": company_normal_dirs,
                "source_layer": "raw_dataset",
            }
        )

    if args.only_reference:
        requested = args.only_reference.strip().lower()
        references = [
            item
            for item in references
            if item["reference_id"].startswith(requested) or item["dataset_name"].lower() == requested
        ]
        if not references:
            raise SystemExit(f"No reference matched --only-reference={args.only_reference!r}")

    builder = ReferenceLibraryBuilder(
        indexes_dir=resolve_project_path(settings.indexes_dir),
        metadata_dir=resolve_project_path(settings.metadata_dir),
    )
    registry = ReferenceRegistry()
    max_images = args.max_images
    configured_projection_path = args.projection_head_path or settings.projection_head_path
    projection_head_path = (
        resolve_project_path(configured_projection_path)
        if settings.use_projection_head and configured_projection_path
        else None
    )
    results = []
    for item in references:
        normal_path = item["normal_path"]
        if not normal_path.exists():
            results.append({**serializable_item(item), "status": "path_not_found"})
            continue
        normal_paths = item.get("normal_paths") or [normal_path]
        if args.reference_manifest_csv:
            image_paths = reference_images_from_manifest(
                resolve_project_path(args.reference_manifest_csv),
                dataset_name=item["dataset_name"],
                fabric_type=item["fabric_type"],
            )
        else:
            image_paths = sorted(
                {
                    image_path
                    for source_path in normal_paths
                    for image_path in datasets.list_images(source_path)
                },
                key=lambda path: str(path).lower(),
            )
        if max_images is not None:
            image_paths = image_paths[:max_images]
        if not image_paths:
            results.append({**serializable_item(item), "status": "empty_reference_path"})
            continue

        reusable = None
        if args.resume:
            reusable = reusable_reference(
                registry=registry,
                item=item,
                image_paths=image_paths,
                max_images=max_images,
                settings=settings,
                projection_head_path=projection_head_path,
            )
        if reusable is not None:
            print(f"[resume] skip {item['reference_id']} ({len(image_paths)} source images)", flush=True)
            results.append(reusable)
            continue
        if args.plan:
            print(f"[plan] build {item['reference_id']} ({len(image_paths)} source images)", flush=True)
            results.append(
                {
                    **serializable_item(item),
                    "status": "would_build",
                    "source_image_count": len(image_paths),
                }
            )
            continue

        print(f"[build] start {item['reference_id']} ({len(image_paths)} source images)", flush=True)
        reference_version = versioned_name(settings.model_name, item["reference_id"])
        artifact = builder.build(
            ReferenceBuildConfig(
                dataset_name=item["dataset_name"],
                fabric_type=item["fabric_type"],
                normal_path=normal_path,
                reference_version=reference_version,
                model_name=settings.model_name,
                device=settings.device,
                use_pretrained_weights=settings.use_pretrained_weights,
                image_max_size=settings.image_max_size,
                model_input_size=settings.model_input_size,
                patch_size=settings.patch_size,
                patch_stride=settings.patch_stride,
                batch_size=settings.batch_size,
                max_images=max_images,
                source_layer=item["source_layer"],
                reference_id=item["reference_id"],
                threshold_config=current_threshold_config(settings),
                projection_head_path=projection_head_path,
            ),
            image_paths=image_paths,
        )
        results.append(
            {
                **serializable_item(item),
                "status": "built",
                "reference_version": artifact.reference_version,
                "processed_image_count": artifact.processed_image_count,
                "skipped_image_count": artifact.skipped_image_count,
                "patch_count": artifact.patch_count,
                "embedding_dim": artifact.embedding_dim,
                "embeddings_path": str(artifact.embeddings_path),
                "faiss_index_path": str(artifact.faiss_index_path),
                "metadata_path": str(artifact.metadata_path),
            }
        )
        print(
            f"[build] done {item['reference_id']} "
            f"(processed={artifact.processed_image_count}, skipped={artifact.skipped_image_count}, "
            f"patches={artifact.patch_count})",
            flush=True,
        )

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "variant": args.variant,
        "max_images": max_images,
        "resume": args.resume,
        "plan_only": args.plan,
        "projection_head_path": str(projection_head_path) if projection_head_path else "",
        "references": results,
    }
    default_name = f"build_extended_references_{args.variant}{'_plan' if args.plan else ''}.json"
    output_path = Path(args.output) if args.output else resolve_project_path(settings.metadata_dir) / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "references": results}, indent=2))


def reference_id(prefix: str, variant: str) -> str:
    return f"{prefix}_{variant}".lower()


def versioned_name(model_name: str, reference_id_value: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    raw = f"{model_name}_{reference_id_value}_{timestamp}".lower()
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def current_threshold_config(settings) -> dict:
    return {
        "anomaly_threshold": settings.anomaly_threshold,
        "good_area_threshold": settings.good_area_threshold,
        "bad_area_threshold": settings.bad_area_threshold,
        "min_defect_cluster_patches": settings.min_defect_cluster_patches,
    }


def reference_images_from_manifest(
    manifest_path: Path,
    *,
    dataset_name: str,
    fabric_type: str | None,
) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = {
        Path(row["image_path"])
        for row in rows
        if row.get("label") == "normal"
        and row.get("dataset_name") == dataset_name
        and (not fabric_type or row.get("fabric_group", "").endswith(fabric_type))
    }
    missing = sorted(path for path in paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"Reference manifest contains {len(missing)} missing image(s): {missing[0]}")
    return sorted(paths, key=lambda path: str(path).lower())


def reusable_reference(
    registry: ReferenceRegistry,
    item: dict,
    image_paths: list[Path],
    max_images: int | None,
    settings,
    projection_head_path: Path | None,
) -> dict | None:
    try:
        entry = registry.get(item["reference_id"])
    except KeyError:
        return None
    if entry.dataset_name != item["dataset_name"] or entry.fabric_type != item["fabric_type"]:
        return None
    if any(not path.exists() or path.stat().st_size <= 0 for path in (entry.metadata_path, entry.embeddings_path, entry.faiss_index_path)):
        return None

    header = read_metadata_header(entry.metadata_path)
    expected_projection = str(projection_head_path.resolve()) if projection_head_path else None
    actual_projection = metadata_header_value(header, "checkpoint_path")
    if actual_projection is not None:
        actual_projection = str(resolve_project_path(str(actual_projection)).resolve())
    if actual_projection != expected_projection:
        return None
    if metadata_header_value(header, "model_name") != settings.model_name:
        return None
    if metadata_header_value(header, "use_pretrained_weights") != settings.use_pretrained_weights:
        return None
    if metadata_header_value(header, "max_images") != max_images:
        return None

    processed = metadata_header_value(header, "processed_image_count")
    skipped = metadata_header_value(header, "skipped_image_count")
    if not isinstance(processed, int) or not isinstance(skipped, int):
        return None
    if processed + skipped != len(image_paths):
        return None

    return {
        **serializable_item(item),
        "status": "skipped_existing",
        "reference_version": entry.reference_version,
        "processed_image_count": processed,
        "skipped_image_count": skipped,
        "patch_count": entry.patch_count,
        "embedding_dim": entry.embedding_dim,
        "embeddings_path": str(entry.embeddings_path),
        "faiss_index_path": str(entry.faiss_index_path),
        "metadata_path": str(entry.metadata_path),
    }


def read_metadata_header(path: Path, max_bytes: int = 262_144) -> str:
    with path.open(encoding="utf-8") as handle:
        return handle.read(max_bytes).split('"images"', 1)[0]


def metadata_header_value(header: str, key: str):
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*(null|true|false|-?\d+(?:\.\d+)?|"(?:\\.|[^"\\])*")',
        header,
    )
    return json.loads(match.group(1)) if match else None


def serializable_item(item: dict) -> dict:
    return {
        key: (
            str(value)
            if isinstance(value, Path)
            else [str(entry) if isinstance(entry, Path) else entry for entry in value]
            if isinstance(value, list)
            else value
        )
        for key, value in item.items()
    }


if __name__ == "__main__":
    main()
