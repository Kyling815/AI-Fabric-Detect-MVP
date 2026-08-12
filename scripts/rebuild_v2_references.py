"""Build versioned v2 FAISS reference artifacts without touching v1 indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder
from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.core.config import get_settings
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild v2 normal/defective FAISS references.")
    parser.add_argument("--normal-path", required=True, type=Path)
    parser.add_argument("--defective-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("storage/references/v2"))
    parser.add_argument("--config", type=Path, default=Path("config/preprocessing_v2.json"))
    parser.add_argument("--fabric-type")
    parser.add_argument("--dataset-name", default="fabric")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = [args.output / "normal.index", args.output / "metadata.json"]
    if args.defective_path:
        required.append(args.output / "defective.index")
    existing = [path for path in required if path.exists()]
    if existing and not args.overwrite:
        print(f"Refusing to overwrite existing artifacts: {', '.join(map(str, existing))}", file=sys.stderr)
        return 2
    config = PreprocessConfig.load(args.config)
    settings = get_settings()
    args.output.mkdir(parents=True, exist_ok=True)
    builder = ReferenceLibraryBuilder(indexes_dir=args.output, metadata_dir=args.output)
    normal = _build(
        builder,
        args.normal_path,
        "normal_v2",
        args.dataset_name,
        args.fabric_type,
        args.max_images,
        config,
        settings,
    )
    shutil.copy2(normal.faiss_index_path, args.output / "normal.index")
    defective = None
    if args.defective_path:
        defective = _build(
            builder,
            args.defective_path,
            "defective_v2",
            args.dataset_name,
            args.fabric_type,
            args.max_images,
            config,
            settings,
        )
        shutil.copy2(defective.faiss_index_path, args.output / "defective.index")
    metadata = json.loads(normal.metadata_path.read_text(encoding="utf-8"))
    summary = {
        **metadata,
        "preprocessing_version": "v2",
        "configuration_hash": config.configuration_hash,
        "backbone": settings.model_name,
        "embedding_dimension": normal.embedding_dim,
        "patch_sizes": list(config.patch_sizes),
        "source_count": normal.processed_image_count,
        "patch_count": normal.patch_count,
        "created_at": metadata["created_at"],
        "faiss_index_path": str(args.output / "normal.index"),
        "normal_index_path": str(args.output / "normal.index"),
        "defective_index_path": str(args.output / "defective.index") if defective else None,
        "normal_metadata_path": str(normal.metadata_path),
        "defective_metadata_path": str(defective.metadata_path) if defective else None,
    }
    (args.output / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _build(
    builder: ReferenceLibraryBuilder,
    source: Path,
    version: str,
    dataset_name: str,
    fabric_type: str | None,
    max_images: int | None,
    config: PreprocessConfig,
    settings,
):
    if not source.exists():
        raise FileNotFoundError(source)
    paths = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda item: str(item).casefold(),
    )
    if max_images is not None:
        paths = paths[:max_images]
    return builder.build(
        ReferenceBuildConfig(
            dataset_name=dataset_name,
            fabric_type=fabric_type,
            normal_path=source,
            reference_version=version,
            model_name=settings.model_name,
            device=settings.device,
            use_pretrained_weights=settings.use_pretrained_weights,
            image_max_size=settings.image_max_size,
            model_input_size=settings.model_input_size,
            patch_size=config.patch_sizes[0],
            patch_stride=max(1, round(config.patch_sizes[0] * (1.0 - config.patch_overlap_ratio))),
            batch_size=settings.batch_size,
            max_images=max_images,
            preprocessing_version="v2",
            preprocessing_config=config.to_dict(),
        ),
        paths,
    )


if __name__ == "__main__":
    raise SystemExit(main())
