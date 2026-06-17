import argparse
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


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build separate reference libraries for DME, WFDD, and Lusitano.")
    parser.add_argument("--variant", default="v1", help="Reference id suffix, e.g. v1 or extended_v1.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--include-lusitano", action="store_true")
    parser.add_argument("--output", default=None)
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
        lusitano_normal = resolve_project_path("Data/raw/external/Lusitano/nondefects")
        references.append(
            {
                "reference_id": reference_id("lusitano_group_auto", args.variant),
                "dataset_name": "lusitano",
                "fabric_type": "group_auto",
                "normal_path": lusitano_normal,
                "source_layer": "raw_dataset",
            }
        )

    builder = ReferenceLibraryBuilder(
        indexes_dir=resolve_project_path(settings.indexes_dir),
        metadata_dir=resolve_project_path(settings.metadata_dir),
    )
    max_images = args.max_images or settings.reference_max_images
    results = []
    for item in references:
        normal_path = item["normal_path"]
        if not normal_path.exists():
            results.append({**serializable_item(item), "status": "path_not_found"})
            continue
        image_paths = datasets.list_images(normal_path, max_images=max_images)
        if not image_paths:
            results.append({**serializable_item(item), "status": "empty_reference_path"})
            continue

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
                projection_head_path=resolve_project_path(settings.projection_head_path)
                if settings.use_projection_head and settings.projection_head_path
                else None,
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

    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "variant": args.variant,
        "max_images": max_images,
        "projection_head_path": settings.projection_head_path if settings.use_projection_head else "",
        "references": results,
    }
    output_path = Path(args.output) if args.output else resolve_project_path(settings.metadata_dir) / f"build_extended_references_{args.variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "references": results}, indent=2))


def reference_id(prefix: str, variant: str) -> str:
    return f"{prefix}_{variant}".lower()


def versioned_name(model_name: str, reference_id_value: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    raw = f"{model_name}_{reference_id_value}_{timestamp}".lower()
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")


def current_threshold_config(settings) -> dict:
    return {
        "anomaly_threshold": settings.anomaly_threshold,
        "good_area_threshold": settings.good_area_threshold,
        "bad_area_threshold": settings.bad_area_threshold,
        "min_defect_cluster_patches": settings.min_defect_cluster_patches,
    }


def serializable_item(item: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in item.items()}


if __name__ == "__main__":
    main()
