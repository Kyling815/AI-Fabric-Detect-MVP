import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.schemas import ReferenceBuildRequest
from backend.app.services.reference_service import ReferenceService
from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normal-reference embeddings and metadata.")
    parser.add_argument("--dataset-name", default="dme_20250129")
    parser.add_argument("--fabric-type", default=None)
    parser.add_argument("--normal-path", default=None)
    parser.add_argument("--reference-id", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--split-manifest", default=None)
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.split_manifest:
            response = build_from_split_manifest(args)
        else:
            service = ReferenceService(db)
            response = service.enqueue_build(
                ReferenceBuildRequest(
                    dataset_name=args.dataset_name,
                    fabric_type=args.fabric_type,
                    normal_path=args.normal_path,
                    reference_id=args.reference_id,
                    max_images=args.max_images,
                )
            )
            response = response.model_dump()
        print(json.dumps(response, indent=2))
    finally:
        db.close()


def build_from_split_manifest(args: argparse.Namespace) -> dict:
    settings = get_settings()
    manifest_path = Path(args.split_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_paths = [Path(item["path"]) for item in manifest["reference_normal"]]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("Split manifest contains no reference_normal images.")

    timestamp = manifest["created_at"].replace("-", "").replace(":", "").replace("Z", "").replace("T", "_")
    reference_version = f"{settings.model_name}_{args.dataset_name}_split_seed{manifest['seed']}_{timestamp}".lower()
    builder = ReferenceLibraryBuilder(
        indexes_dir=resolve_project_path(settings.indexes_dir),
        metadata_dir=resolve_project_path(settings.metadata_dir),
    )
    artifact = builder.build(
        ReferenceBuildConfig(
            dataset_name=args.dataset_name,
            fabric_type=args.fabric_type,
            normal_path=Path(manifest["dataset_root"]),
            reference_version=reference_version,
            model_name=settings.model_name,
            device=settings.device,
            use_pretrained_weights=settings.use_pretrained_weights,
            image_max_size=settings.image_max_size,
            model_input_size=settings.model_input_size,
            patch_size=settings.patch_size,
            patch_stride=settings.patch_stride,
            batch_size=settings.batch_size,
            max_images=args.max_images,
            source_layer="split_manifest",
            projection_head_path=resolve_project_path(settings.projection_head_path)
            if settings.use_projection_head and settings.projection_head_path
            else None,
            reference_id=args.reference_id,
        ),
        image_paths=image_paths,
    )
    return {
        "status": "built",
        "message": "Reference library built from split manifest reference_normal images.",
        "dataset_name": args.dataset_name,
        "reference_id": args.reference_id,
        "reference_version": artifact.reference_version,
        "split_manifest": str(manifest_path),
        "normal_image_count": len(manifest["reference_normal"]),
        "processed_image_count": artifact.processed_image_count,
        "skipped_image_count": artifact.skipped_image_count,
        "patch_count": artifact.patch_count,
        "embedding_dim": artifact.embedding_dim,
        "embeddings_path": str(artifact.embeddings_path),
        "faiss_index_path": str(artifact.faiss_index_path),
        "faiss_metric": artifact.faiss_metric,
        "metadata_path": str(artifact.metadata_path),
    }


if __name__ == "__main__":
    main()
