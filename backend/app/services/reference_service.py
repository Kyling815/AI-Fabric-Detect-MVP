import json
import re
from datetime import datetime

from sqlalchemy.orm import Session

from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder
from backend.app.ai.preprocessing.config import PreprocessConfig, load_registry_overrides
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.db import crud
from backend.app.db.models import ModelVersion
from backend.app.db.schemas import (
    ReferenceBuildRequest,
    ReferenceBuildResponse,
    ReferenceRegistryItem,
    ReferenceRegistryListResponse,
    ReferenceStatusResponse,
)
from backend.app.services.dataset_service import DatasetService
from backend.app.services.reference_registry import ReferenceRegistry


class ReferenceService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.datasets = DatasetService()

    def enqueue_build(self, request: ReferenceBuildRequest) -> ReferenceBuildResponse:
        normal_path = self.datasets.resolve_reference_normal_path(
            request.dataset_name,
            request.fabric_type,
            request.normal_path,
        )
        if normal_path is None:
            return ReferenceBuildResponse(
                status="needs_normal_path",
                message="Provide normal_path, or provide fabric_type for WFDD/MVTec AD.",
                dataset_name=request.dataset_name,
            )
        if not normal_path.exists():
            return ReferenceBuildResponse(
                status="path_not_found",
                message=f"Normal-reference path does not exist: {normal_path}",
                dataset_name=request.dataset_name,
                normal_path=str(normal_path),
            )

        image_count = self.datasets.count_images(normal_path)
        if image_count == 0:
            return ReferenceBuildResponse(
                status="empty_reference_path",
                message=f"Normal-reference path exists but contains no supported images: {normal_path}",
                dataset_name=request.dataset_name,
                normal_path=str(normal_path),
            )

        max_images = request.max_images or self.settings.reference_max_images
        image_paths = self.datasets.list_images(normal_path, max_images=max_images)
        preprocessing_version = request.preprocessing_version or self.settings.preprocessing_version
        preprocessing_config = self._preprocessing_config(
            preprocessing_version,
            request.preprocessing_config_path,
            request.fabric_type,
        )
        reference_version = self._reference_version(
            request.dataset_name,
            request.fabric_type,
            preprocessing_version,
        )
        builder = ReferenceLibraryBuilder(
            indexes_dir=resolve_project_path(self.settings.indexes_dir),
            metadata_dir=resolve_project_path(self.settings.metadata_dir),
        )
        artifact = builder.build(
            ReferenceBuildConfig(
                dataset_name=request.dataset_name,
                fabric_type=request.fabric_type,
                normal_path=normal_path,
                reference_version=reference_version,
                model_name=self.settings.model_name,
                device=self.settings.device,
                use_pretrained_weights=self.settings.use_pretrained_weights,
                image_max_size=self.settings.image_max_size,
                model_input_size=self.settings.model_input_size,
                patch_size=self.settings.patch_size,
                patch_stride=self.settings.patch_stride,
                batch_size=self.settings.batch_size,
                max_images=max_images,
                source_layer=self._source_layer(normal_path),
                reference_id=request.reference_id,
                projection_head_path=resolve_project_path(self.settings.projection_head_path)
                if self.settings.use_projection_head and self.settings.projection_head_path
                else None,
                preprocessing_version=preprocessing_version,
                preprocessing_config=preprocessing_config,
            ),
            image_paths=image_paths,
        )
        if artifact.processed_image_count > 0:
            crud.create_model_version(
                self.db,
                ModelVersion(
                    model_name=self.settings.model_name,
                    patch_size=self.settings.patch_size,
                    patch_stride=self.settings.patch_stride,
                    image_max_size=self.settings.image_max_size,
                    anomaly_threshold=self.settings.anomaly_threshold,
                    reference_version=artifact.reference_version,
                ),
            )

        status = "built" if artifact.processed_image_count > 0 else "no_valid_images"
        message = (
            "Normal-reference embeddings, FAISS index, and metadata were saved."
            if artifact.processed_image_count > 0
            else "No readable images were found; empty embeddings, FAISS index, and metadata were saved for diagnostics."
        )
        return ReferenceBuildResponse(
            status=status,
            message=message,
            dataset_name=request.dataset_name,
            reference_id=request.reference_id or artifact.reference_version,
            reference_version=artifact.reference_version,
            normal_path=str(normal_path),
            normal_image_count=image_count,
            processed_image_count=artifact.processed_image_count,
            skipped_image_count=artifact.skipped_image_count,
            patch_count=artifact.patch_count,
            embedding_dim=artifact.embedding_dim,
            embeddings_path=str(artifact.embeddings_path),
            faiss_index_path=str(artifact.faiss_index_path),
            faiss_metric=artifact.faiss_metric,
            metadata_path=str(artifact.metadata_path),
        )

    def get_status(self) -> ReferenceStatusResponse:
        latest_path = resolve_project_path(self.settings.metadata_dir) / "latest_reference.json"
        if latest_path.exists():
            metadata = json.loads(latest_path.read_text(encoding="utf-8"))
            thresholds = metadata.get("thresholds") or {}
            preprocess = metadata.get("preprocess") or {}
            model = metadata.get("model") or {}
            return ReferenceStatusResponse(
                status="built",
                reference_id=metadata.get("reference_id"),
                active_reference_version=metadata.get("reference_version"),
                model_name=str(model.get("model_name", self.settings.model_name)),
                patch_size=int(preprocess.get("patch_size", self.settings.patch_size)),
                patch_stride=int(preprocess.get("patch_stride", self.settings.patch_stride)),
                anomaly_threshold=float(thresholds.get("anomaly_threshold", self.settings.anomaly_threshold)),
                embeddings_path=metadata.get("embeddings_path"),
                faiss_index_path=metadata.get("faiss_index_path"),
                faiss_metric=metadata.get("faiss_metric"),
                metadata_path=str(latest_path),
                patch_count=int(metadata.get("patch_count", 0)),
                image_count=int(metadata.get("processed_image_count", 0)),
            )
        return ReferenceStatusResponse(
            status="not_built",
            reference_id=None,
            active_reference_version=None,
            model_name=self.settings.model_name,
            patch_size=self.settings.patch_size,
            patch_stride=self.settings.patch_stride,
            anomaly_threshold=self.settings.anomaly_threshold,
        )

    def list_references(self) -> ReferenceRegistryListResponse:
        registry = ReferenceRegistry()
        references = [
            ReferenceRegistryItem(
                reference_id=item.reference_id,
                reference_version=item.reference_version,
                dataset_name=item.dataset_name,
                fabric_type=item.fabric_type,
                source_layer=item.source_layer,
                metadata_path=str(item.metadata_path),
                embeddings_path=str(item.embeddings_path),
                faiss_index_path=str(item.faiss_index_path),
                faiss_metric=item.faiss_metric,
                embedding_dim=item.embedding_dim,
                patch_count=item.patch_count,
                processed_image_count=item.processed_image_count,
                created_at=item.created_at,
                thresholds=item.thresholds,
                preprocessing_version=item.preprocessing_version,
                configuration_hash=item.configuration_hash,
            )
            for item in registry.list_latest_entries()
        ]
        return ReferenceRegistryListResponse(
            active_reference_version=registry.active_reference_version,
            default_reference_id=registry.default_reference_id() if references else None,
            references=references,
        )

    def get_reference_status(self, reference_id: str) -> ReferenceStatusResponse:
        try:
            entry = ReferenceRegistry().get(reference_id)
        except KeyError:
            return ReferenceStatusResponse(
                status="not_found",
                reference_id=reference_id,
                active_reference_version=None,
                model_name=self.settings.model_name,
                patch_size=self.settings.patch_size,
                patch_stride=self.settings.patch_stride,
                anomaly_threshold=self.settings.anomaly_threshold,
            )
        metadata = {}
        if entry.metadata_path.exists():
            try:
                metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        preprocess = metadata.get("preprocess") or {}
        model = metadata.get("model") or {}
        threshold = float(entry.thresholds.get("anomaly_threshold", self.settings.anomaly_threshold))
        return ReferenceStatusResponse(
            status="built",
            reference_id=entry.reference_id,
            active_reference_version=entry.reference_version,
            model_name=str(model.get("model_name", self.settings.model_name)),
            patch_size=int(preprocess.get("patch_size", self.settings.patch_size)),
            patch_stride=int(preprocess.get("patch_stride", self.settings.patch_stride)),
            anomaly_threshold=threshold,
            embeddings_path=str(entry.embeddings_path),
            faiss_index_path=str(entry.faiss_index_path),
            faiss_metric=entry.faiss_metric,
            metadata_path=str(entry.metadata_path),
            patch_count=entry.patch_count,
            image_count=entry.processed_image_count,
        )

    def _reference_version(
        self,
        dataset_name: str,
        fabric_type: str | None,
        preprocessing_version: str = "v1",
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        parts = [self.settings.model_name, dataset_name, preprocessing_version]
        if fabric_type:
            parts.append(fabric_type)
        parts.append(timestamp)
        raw = "_".join(parts).lower()
        return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")

    def _preprocessing_config(
        self,
        version: str,
        config_path: str | None,
        fabric_type: str | None,
    ) -> dict | None:
        if version == "v1":
            return None
        path_value = config_path or self.settings.preprocessing_config_path
        config = PreprocessConfig.load(resolve_project_path(path_value)) if path_value else PreprocessConfig()
        registry_path = resolve_project_path(self.settings.reference_registry_path)
        overrides = load_registry_overrides(registry_path, fabric_type)
        return (config.with_overrides(overrides) if overrides else config).to_dict()

    def _source_layer(self, normal_path) -> str:
        reference_ready = resolve_project_path(self.settings.reference_ready_dir)
        try:
            normal_path.resolve().relative_to(reference_ready.resolve())
        except ValueError:
            return "raw_dataset"
        return "processed_reference_ready"
