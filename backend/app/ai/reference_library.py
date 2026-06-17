import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.patching import build_patch_grid, extract_image_patches
from backend.app.ai.preprocess import ImageReadError, build_preprocess_plan, inspect_image_file, load_resized_rgb_image
from backend.app.ai.projection_head import project_embeddings


@dataclass(frozen=True)
class ReferenceBuildConfig:
    dataset_name: str
    fabric_type: str | None
    normal_path: Path
    reference_version: str
    model_name: str
    device: str
    use_pretrained_weights: bool
    image_max_size: int
    model_input_size: int
    patch_size: int
    patch_stride: int
    batch_size: int
    max_images: int | None
    source_layer: str = "raw_dataset"
    projection_head_path: Path | None = None
    reference_id: str | None = None
    threshold_config: dict | None = None


@dataclass(frozen=True)
class ReferenceBuildArtifact:
    reference_version: str
    embeddings_path: Path
    faiss_index_path: Path
    metadata_path: Path
    processed_image_count: int
    skipped_image_count: int
    patch_count: int
    embedding_dim: int
    faiss_metric: str


class ReferenceLibraryBuilder:
    def __init__(self, indexes_dir: Path, metadata_dir: Path) -> None:
        self.indexes_dir = indexes_dir
        self.metadata_dir = metadata_dir
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def build(self, config: ReferenceBuildConfig, image_paths: list[Path]) -> ReferenceBuildArtifact:
        from backend.app.ai.feature_extractor import CNNFeatureExtractor

        extractor = CNNFeatureExtractor(
            model_name=config.model_name,
            device=config.device,
            use_pretrained_weights=config.use_pretrained_weights,
            model_input_size=config.model_input_size,
        )
        all_embeddings: list[np.ndarray] = []
        image_records: list[dict] = []
        skipped_records: list[dict] = []
        embedding_offset = 0

        for image_path in image_paths:
            try:
                image_info = inspect_image_file(image_path)
                plan = build_preprocess_plan(
                    width=image_info.width,
                    height=image_info.height,
                    target_max_size=config.image_max_size,
                    patch_size=config.patch_size,
                    patch_stride=config.patch_stride,
                )
                resized_image = load_resized_rgb_image(image_path, plan)
                grid = build_patch_grid(
                    image_width=resized_image.width,
                    image_height=resized_image.height,
                    patch_size=plan.patch_size,
                    patch_stride=plan.patch_stride,
                )
                patches = extract_image_patches(resized_image, grid)
                embeddings = extractor.extract((patch.image for patch in patches), batch_size=config.batch_size)
            except (ImageReadError, OSError, ValueError) as exc:
                skipped_records.append({"path": str(image_path), "reason": str(exc)})
                continue

            all_embeddings.append(embeddings)
            image_records.append(
                {
                    "path": str(image_path),
                    "width": image_info.width,
                    "height": image_info.height,
                    "mode": image_info.mode,
                    "resized_width": plan.resized_width,
                    "resized_height": plan.resized_height,
                    "patch_count": len(patches),
                    "embedding_start": embedding_offset,
                    "embedding_end": embedding_offset + len(patches),
                    "patches": [
                        {
                            "x": patch.box.x,
                            "y": patch.box.y,
                            "w": patch.box.w,
                            "h": patch.box.h,
                            "embedding_offset": embedding_offset + index,
                        }
                        for index, patch in enumerate(patches)
                    ],
                }
            )
            embedding_offset += len(patches)

        embedding_dim = extractor.info.embedding_dim
        if all_embeddings:
            embedding_matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        else:
            embedding_matrix = np.empty((0, embedding_dim), dtype=np.float32)

        projection_metadata = None
        if config.projection_head_path:
            embedding_matrix, projection_info = project_embeddings(
                embedding_matrix,
                checkpoint_path=config.projection_head_path,
                device=extractor.info.device,
                batch_size=max(config.batch_size, 1),
            )
            embedding_dim = projection_info.projection_dim
            projection_metadata = asdict(projection_info)

        embeddings_path = self.indexes_dir / f"{config.reference_version}_embeddings.npy"
        faiss_index_path = self.indexes_dir / f"{config.reference_version}_faiss.index"
        metadata_path = self.metadata_dir / f"{config.reference_version}_metadata.json"
        np.save(embeddings_path, embedding_matrix)
        faiss_info = FaissIndexStore(metric="cosine").save(embedding_matrix, faiss_index_path)

        metadata = {
            "reference_id": config.reference_id or config.reference_version,
            "reference_version": config.reference_version,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dataset_name": config.dataset_name,
            "fabric_type": config.fabric_type,
            "source_layer": config.source_layer,
            "normal_path": str(config.normal_path),
            "model": asdict(extractor.info),
            "projection_head": projection_metadata,
            "preprocess": {
                "image_max_size": config.image_max_size,
                "model_input_size": config.model_input_size,
                "patch_size": config.patch_size,
                "patch_stride": config.patch_stride,
                "batch_size": config.batch_size,
            },
            "max_images": config.max_images,
            "processed_image_count": len(image_records),
            "skipped_image_count": len(skipped_records),
            "patch_count": int(embedding_matrix.shape[0]),
            "embedding_dim": embedding_dim,
            "embeddings_path": str(embeddings_path),
            "faiss_index_path": str(faiss_info.path),
            "faiss_metric": faiss_info.metric,
            "faiss_vector_count": faiss_info.vector_count,
            "thresholds": config.threshold_config or {},
            "images": image_records,
            "skipped": skipped_records,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.metadata_dir / "latest_reference.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._update_registry(metadata, metadata_path)

        return ReferenceBuildArtifact(
            reference_version=config.reference_version,
            embeddings_path=embeddings_path,
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
            processed_image_count=len(image_records),
            skipped_image_count=len(skipped_records),
            patch_count=int(embedding_matrix.shape[0]),
            embedding_dim=embedding_dim,
            faiss_metric=faiss_info.metric,
        )

    def _update_registry(self, metadata: dict, metadata_path: Path) -> None:
        registry_path = self.metadata_dir / "reference_registry.json"
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                registry = {}
        else:
            registry = {}

        references = [
            item
            for item in registry.get("references", [])
            if item.get("reference_version") != metadata["reference_version"]
        ]
        references.append(
            {
                "reference_version": metadata["reference_version"],
                "reference_id": metadata.get("reference_id", metadata["reference_version"]),
                "dataset_name": metadata["dataset_name"],
                "fabric_type": metadata.get("fabric_type"),
                "source_layer": metadata.get("source_layer", "raw_dataset"),
                "normal_path": metadata["normal_path"],
                "metadata_path": str(metadata_path),
                "embeddings_path": metadata["embeddings_path"],
                "faiss_index_path": metadata["faiss_index_path"],
                "faiss_metric": metadata["faiss_metric"],
                "embedding_dim": metadata["embedding_dim"],
                "patch_count": metadata["patch_count"],
                "processed_image_count": metadata["processed_image_count"],
                "created_at": metadata["created_at"],
                "thresholds": metadata.get("thresholds", {}),
            }
        )
        registry = {
            "schema_version": 1,
            "active_reference_version": metadata["reference_version"],
            "active_metadata_path": str(metadata_path),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "references": sorted(references, key=lambda item: item["created_at"]),
        }
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
