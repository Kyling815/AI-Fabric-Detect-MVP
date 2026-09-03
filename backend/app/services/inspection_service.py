from dataclasses import dataclass
import json
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from pathlib import Path

import numpy as np

from backend.app.ai.anomaly_detector import AnomalyDetectionResult, FaissAnomalyDetector, ReferenceNotReadyError
from backend.app.ai.bbox import MergedBox, localize_pixel_defects, localize_suspicious_patches
from backend.app.ai.classifier import classify_inspection
from backend.app.ai.dense_anomaly import DenseReferenceDetector
from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.fabric_decision import decide_fabric_inspection
from backend.app.ai.patching import ImagePatch, PatchGrid, build_patch_grid
from backend.app.ai.preprocess import ImageReadError, build_preprocess_plan, inspect_image_file
from backend.app.ai.preprocessing.adapter import CoordinateTransform, PreparedImage, configuration_from_reference_metadata, prepare_image
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.db import crud
from backend.app.db.models import DetectedDefect, Inspection
from backend.app.db.schemas import (
    AnomalySummary,
    DefectBox,
    EmbeddingSummary,
    ImageMetadata,
    InspectionCreateResponse,
    InspectionListItem,
    InspectionListResponse,
    PatchBox,
    PatchGridSummary,
    PreprocessPlan,
    SuspiciousPatch,
)
from backend.app.services.storage_service import StorageService, UploadTooLargeError
from backend.app.services.fabric_registry import FabricRegistry, FabricRegistryError
from backend.app.services.reference_registry import ReferenceRegistry
from backend.app.utils.visualization import save_annotated_image


@dataclass(frozen=True)
class RuntimeEmbeddingResult:
    summary: EmbeddingSummary
    embeddings: np.ndarray
    patches: list[ImagePatch]
    coordinate_transform: CoordinateTransform
    prepared_image: PreparedImage


@dataclass(frozen=True)
class RuntimeAnomalyResult:
    summary: AnomalySummary
    raw_result: AnomalyDetectionResult | None


@dataclass(frozen=True)
class RuntimeReferenceContext:
    reference_id: str
    metadata_path: Path
    anomaly_threshold: float
    good_area_threshold: float
    bad_area_threshold: float
    min_defect_cluster_patches: int


class InspectionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = StorageService()
        self.settings = get_settings()

    def create_fabric_inspection(self, image: UploadFile, fabric_id: str) -> InspectionCreateResponse:
        """Run the demo dense-anomaly pipeline against exactly one active fabric reference."""
        try:
            registry = FabricRegistry()
            reference = registry.get_active_reference(fabric_id)
        except FabricRegistryError as exc:
            status_code = 400 if str(exc).startswith("Unknown fabric_id") else 409
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

        backbone = registry.shared_backbone
        try:
            stored_upload = self.storage.save_upload(image)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        annotated = None
        persisted = False
        try:
            try:
                image_info = inspect_image_file(stored_upload.absolute_path)
            except ImageReadError as exc:
                raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.") from exc

            extractor = get_shared_dense_extractor(
                str(backbone["backbone_version"]),
                self.settings.device,
                bool(backbone.get("use_pretrained_weights", True)),
                str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
            )
            features = extractor.extract_path(
                stored_upload.absolute_path,
                image_max_size=int(backbone.get("image_max_size", self.settings.image_max_size)),
            )
            try:
                detector = DenseReferenceDetector(reference.metadata_path)
                anomaly = detector.score(
                    features.embeddings,
                    features.boxes,
                    grid_rows=features.grid_rows,
                    grid_cols=features.grid_cols,
                    image_width=features.processed_width,
                    image_height=features.processed_height,
                    thresholds=reference.thresholds,
                )
            except (OSError, KeyError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Active fabric reference is invalid or incompatible: {exc}",
                ) from exc

            original_boxes = self._map_dense_boxes_to_original(
                list(anomaly.boxes),
                scale=features.scale,
                width=features.original_width,
                height=features.original_height,
            )
            decision = decide_fabric_inspection(
                image_score=anomaly.image_score,
                boxes=original_boxes,
                thresholds=reference.thresholds,
                localized_score=anomaly.localized_score,
            )
            defect_area_ratio = self._box_area_ratio(
                original_boxes,
                features.original_width,
                features.original_height,
            )
            inspection_id = str(uuid4())
            annotated = self.storage.annotated_output_for(inspection_id, image.filename or "upload.jpg")
            save_annotated_image(stored_upload.absolute_path, original_boxes, annotated.absolute_path)

            inspection = Inspection(
                id=inspection_id,
                filename=image.filename or "upload",
                original_image_path=stored_upload.relative_path,
                annotated_image_path=annotated.relative_path,
                content_type=image.content_type,
                file_size_bytes=stored_upload.file_size_bytes,
                image_width=image_info.width,
                image_height=image_info.height,
                image_mode=image_info.mode,
                final_result=decision.label,
                confidence=None,
                defect_area_ratio=defect_area_ratio,
                defect_count=len(original_boxes),
                model_version=reference.reference_id,
                fabric_id=reference.fabric_id,
                fabric_name=reference.display_name,
                reference_id=reference.reference_id,
                backbone_version=reference.backbone_version,
                anomaly_score=anomaly.image_score,
                decision_reason=decision.reason,
            )
            defects = self._detected_defects(inspection.id, original_boxes)
            inspection = crud.create_inspection_with_defects(self.db, inspection, defects)
            persisted = True
            pixel_threshold = float(reference.thresholds["pixel_threshold"])
            return self._to_response(
                inspection,
                embedding_summary=EmbeddingSummary(
                    model_name=extractor.info.architecture,
                    device=extractor.info.device,
                    requested_device=self.settings.device,
                    use_pretrained_weights=extractor.info.use_pretrained_weights,
                    embedding_dim=extractor.info.feature_dim,
                    patch_count=int(features.embeddings.shape[0]),
                    batch_size=1,
                    model_input_size=int(backbone.get("image_max_size", self.settings.image_max_size)),
                    l2_normalized=True,
                ),
                anomaly_summary=AnomalySummary(
                    status="scored",
                    reference_version=reference.reference_id,
                    faiss_metric="cosine",
                    threshold=float(reference.thresholds["image_review_threshold"]),
                    patch_count=int(anomaly.patch_scores.size),
                    suspicious_patch_count=int((anomaly.patch_scores >= pixel_threshold).sum()),
                    max_anomaly_score=float(anomaly.patch_scores.max()),
                    mean_anomaly_score=float(anomaly.patch_scores.mean()),
                    defect_area_ratio_estimate=defect_area_ratio,
                ),
                boxes=self._defect_boxes(original_boxes),
            )
        except Exception:
            if not persisted:
                self.storage.delete_stored_file(stored_upload.relative_path)
                if annotated is not None:
                    self.storage.delete_stored_file(annotated.relative_path)
            raise

    def create_placeholder_inspection(self, image: UploadFile, reference_id: str | None = None) -> InspectionCreateResponse:
        reference_context = self._resolve_reference_context(reference_id)
        try:
            stored_upload = self.storage.save_upload(image)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        annotated = None
        persisted = False
        try:
            try:
                image_info = inspect_image_file(stored_upload.absolute_path)
            except ImageReadError as exc:
                raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.") from exc

            inspection_id = str(uuid4())
            embedding_result = self._extract_patch_embeddings(
                stored_upload.absolute_path,
                image_info.width,
                image_info.height,
                reference_context.metadata_path,
            )
            anomaly_result = self._score_anomalies(embedding_result, reference_context)
            original_boxes = self._merged_original_boxes(
                anomaly_result.raw_result,
                embedding_result.coordinate_transform,
                reference_context.min_defect_cluster_patches,
                embedding_result.prepared_image.processed_rgb,
            )
            defect_area_ratio = self._box_area_ratio(original_boxes, image_info.width, image_info.height)
            classification = (
                classify_inspection(
                    defect_area_ratio=defect_area_ratio,
                    defect_count=len(original_boxes),
                    max_anomaly_score=anomaly_result.summary.max_anomaly_score,
                    good_area_threshold=reference_context.good_area_threshold,
                    bad_area_threshold=reference_context.bad_area_threshold,
                )
                if anomaly_result.summary.status == "scored"
                else None
            )
            if anomaly_result.summary.status == "scored":
                annotated = self.storage.annotated_output_for(inspection_id, image.filename or "upload.jpg")
                save_annotated_image(stored_upload.absolute_path, original_boxes, annotated.absolute_path)

            inspection = Inspection(
                id=inspection_id,
                filename=image.filename or "upload",
                original_image_path=stored_upload.relative_path,
                annotated_image_path=annotated.relative_path if annotated else None,
                content_type=image.content_type,
                file_size_bytes=stored_upload.file_size_bytes,
                image_width=image_info.width,
                image_height=image_info.height,
                image_mode=image_info.mode,
                final_result=classification.label if classification else "ReferenceMissing",
                confidence=classification.confidence if classification else None,
                defect_area_ratio=defect_area_ratio,
                defect_count=len(original_boxes),
                model_version=anomaly_result.summary.reference_version or "reference_not_built",
            )
            defects = self._detected_defects(inspection.id, original_boxes)
            inspection = crud.create_inspection_with_defects(self.db, inspection, defects)
            persisted = True
            return self._to_response(
                inspection,
                reference_id=reference_context.reference_id,
                embedding_summary=embedding_result.summary,
                anomaly_summary=anomaly_result.summary,
                boxes=self._defect_boxes(original_boxes),
                prepared_image=embedding_result.prepared_image,
            )
        except Exception:
            if not persisted:
                self.storage.delete_stored_file(stored_upload.relative_path)
                if annotated is not None:
                    self.storage.delete_stored_file(annotated.relative_path)
            raise

    def list_inspections(self) -> InspectionListResponse:
        inspections = crud.list_inspections(self.db)
        return InspectionListResponse(
            items=[
                InspectionListItem(
                    inspection_id=item.id,
                    filename=item.filename,
                    final_result=item.final_result,
                    defect_area_ratio=item.defect_area_ratio,
                    defect_count=item.defect_count,
                    fabric_id=item.fabric_id,
                    fabric_name=item.fabric_name,
                    reference_id=item.reference_id,
                    created_at=item.created_at,
                )
                for item in inspections
            ]
        )

    def get_inspection(self, inspection_id: str) -> InspectionCreateResponse | None:
        inspection = crud.get_inspection(self.db, inspection_id)
        if inspection is None:
            return None
        return self._to_response(inspection)

    def _to_response(
        self,
        inspection: Inspection,
        reference_id: str | None = None,
        embedding_summary: EmbeddingSummary | None = None,
        anomaly_summary: AnomalySummary | None = None,
        boxes: list[DefectBox] | None = None,
        prepared_image: PreparedImage | None = None,
    ) -> InspectionCreateResponse:
        return InspectionCreateResponse(
            inspection_id=inspection.id,
            fabric_id=inspection.fabric_id,
            fabric_name=inspection.fabric_name,
            reference_id=reference_id or inspection.reference_id or inspection.model_version,
            backbone_version=inspection.backbone_version,
            final_result=inspection.final_result,
            anomaly_score=inspection.anomaly_score,
            reason=inspection.decision_reason,
            confidence=inspection.confidence,
            defect_area_ratio=inspection.defect_area_ratio,
            defect_count=inspection.defect_count,
            boxes=boxes if boxes is not None else self._defect_boxes_from_inspection(inspection),
            original_image_url=self.storage.to_public_url(inspection.original_image_path),
            annotated_image_url=self.storage.to_public_url(inspection.annotated_image_path),
            image_metadata=self._image_metadata(inspection),
            preprocess_plan=(
                self._preprocess_plan_from_prepared(inspection, prepared_image)
                if prepared_image is not None
                else self._preprocess_plan(inspection)
            ),
            patch_grid=(
                self._patch_grid_summary_from_prepared(prepared_image)
                if prepared_image is not None
                else self._patch_grid_summary(inspection)
            ),
            embedding_summary=embedding_summary,
            anomaly_summary=anomaly_summary,
            model_version=inspection.model_version,
            created_at=inspection.created_at,
        )

    def _map_dense_boxes_to_original(
        self,
        boxes: list[MergedBox],
        *,
        scale: float,
        width: int,
        height: int,
    ) -> list[MergedBox]:
        if scale <= 0:
            raise ValueError("Dense feature scale must be positive")
        mapped: list[MergedBox] = []
        for box in boxes:
            x1 = max(0, min(width, round(box.x / scale)))
            y1 = max(0, min(height, round(box.y / scale)))
            x2 = max(x1, min(width, round((box.x + box.w) / scale)))
            y2 = max(y1, min(height, round((box.y + box.h) / scale)))
            if x2 <= x1 or y2 <= y1:
                continue
            mapped.append(
                MergedBox(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    score=box.score,
                    patch_count=box.patch_count,
                    label=box.label,
                )
            )
        return mapped

    def _image_metadata(self, inspection: Inspection) -> ImageMetadata | None:
        if inspection.image_width is None or inspection.image_height is None or inspection.file_size_bytes is None:
            return None
        return ImageMetadata(
            width=inspection.image_width,
            height=inspection.image_height,
            mode=inspection.image_mode or "unknown",
            content_type=inspection.content_type,
            file_size_bytes=inspection.file_size_bytes,
        )

    def _preprocess_plan(self, inspection: Inspection) -> PreprocessPlan | None:
        if inspection.image_width is None or inspection.image_height is None:
            return None
        plan = build_preprocess_plan(
            width=inspection.image_width,
            height=inspection.image_height,
            target_max_size=self.settings.image_max_size,
            patch_size=self.settings.patch_size,
            patch_stride=self.settings.patch_stride,
        )
        return PreprocessPlan(
            target_max_size=plan.target_max_size,
            original_width=plan.original_width,
            original_height=plan.original_height,
            resized_width=plan.resized_width,
            resized_height=plan.resized_height,
            scale=plan.scale,
            patch_size=plan.patch_size,
            patch_stride=plan.patch_stride,
            color_mode=plan.color_mode,
        )

    def _patch_grid_summary(self, inspection: Inspection) -> PatchGridSummary | None:
        plan = self._preprocess_plan(inspection)
        if plan is None:
            return None

        grid = build_patch_grid(
            image_width=plan.resized_width,
            image_height=plan.resized_height,
            patch_size=plan.patch_size,
            patch_stride=plan.patch_stride,
        )
        return self._grid_to_summary(grid)

    def _extract_patch_embeddings(self, image_path: Path, width: int, height: int, metadata_path: Path) -> RuntimeEmbeddingResult:
        from backend.app.ai.feature_extractor import get_cached_feature_extractor

        try:
            prepared = self._prepare_patches(image_path, width, height, metadata_path)
        except (ImageReadError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {exc}") from exc
        patches = prepared.patches
        if not patches:
            raise HTTPException(status_code=400, detail="Image produced no patches.")

        reference_metadata = self._load_reference_metadata(metadata_path)
        model_metadata = reference_metadata.get("model") or {}
        extractor = get_cached_feature_extractor(
            model_name=str(model_metadata.get("model_name", self.settings.model_name)),
            device=self.settings.device,
            use_pretrained_weights=bool(
                model_metadata.get("use_pretrained_weights", self.settings.use_pretrained_weights)
            ),
            model_input_size=int(model_metadata.get("model_input_size", self.settings.model_input_size)),
        )
        embeddings = extractor.extract((patch.image for patch in patches), batch_size=self.settings.batch_size)
        info = extractor.info
        projection_path = self._active_projection_head_path(metadata_path)
        if projection_path:
            from backend.app.ai.projection_head import project_embeddings

            try:
                embeddings, projection_info = project_embeddings(
                    embeddings,
                    checkpoint_path=projection_path,
                    device=info.device,
                    batch_size=max(self.settings.batch_size, 1),
                )
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Reference projection head is missing or incompatible: {exc}",
                ) from exc
            embedding_dim = projection_info.projection_dim
        else:
            embedding_dim = info.embedding_dim
        return RuntimeEmbeddingResult(
            summary=EmbeddingSummary(
                model_name=info.model_name,
                device=info.device,
                requested_device=info.requested_device,
                use_pretrained_weights=info.use_pretrained_weights,
                embedding_dim=embedding_dim,
                patch_count=int(embeddings.shape[0]),
                batch_size=self.settings.batch_size,
                model_input_size=info.model_input_size,
                l2_normalized=info.l2_normalized,
            ),
            embeddings=embeddings,
            patches=patches,
            coordinate_transform=prepared.coordinate_transform,
            prepared_image=prepared,
        )

    def _active_projection_head_path(self, metadata_path: Path) -> Path | None:
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                projection = metadata.get("projection_head")
                if projection and projection.get("checkpoint_path"):
                    return resolve_project_path(str(projection["checkpoint_path"]))
            except (OSError, json.JSONDecodeError):
                pass
        if self.settings.use_projection_head and self.settings.projection_head_path:
            return resolve_project_path(self.settings.projection_head_path)
        return None

    def _score_anomalies(
        self,
        embedding_result: RuntimeEmbeddingResult,
        reference_context: RuntimeReferenceContext,
    ) -> RuntimeAnomalyResult:
        patch_boxes = [patch.box for patch in embedding_result.patches]
        try:
            detector = FaissAnomalyDetector(
                metadata_path=reference_context.metadata_path,
                threshold=reference_context.anomaly_threshold,
            )
            expected_dim = int(detector.index.d)
            actual_dim = int(embedding_result.embeddings.shape[1])
            if actual_dim != expected_dim:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Reference/model embedding mismatch: "
                        f"reference expects {expected_dim}, query produced {actual_dim}. Rebuild the reference."
                    ),
                )
            result = detector.score(embedding_result.embeddings, patch_boxes)
        except (ReferenceNotReadyError, FileNotFoundError, KeyError):
            return RuntimeAnomalyResult(
                summary=AnomalySummary(
                    status="reference_missing",
                    reference_version=None,
                    faiss_metric=None,
                    threshold=reference_context.anomaly_threshold,
                    patch_count=len(patch_boxes),
                    suspicious_patch_count=0,
                    max_anomaly_score=None,
                    mean_anomaly_score=None,
                    defect_area_ratio_estimate=0.0,
                ),
                raw_result=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=f"Reference index is invalid or incompatible: {exc}") from exc

        return RuntimeAnomalyResult(
            summary=self._anomaly_summary_from_result(result, patch_count=len(patch_boxes)),
            raw_result=result,
        )

    def _anomaly_summary_from_result(self, result: AnomalyDetectionResult, patch_count: int) -> AnomalySummary:
        suspicious_patches = [item for item in result.patch_scores if item.suspicious]
        return AnomalySummary(
            status=result.status,
            reference_version=result.reference_version,
            faiss_metric=result.faiss_metric,
            threshold=result.threshold,
            patch_count=patch_count,
            suspicious_patch_count=result.suspicious_patch_count,
            max_anomaly_score=result.max_anomaly_score,
            mean_anomaly_score=result.mean_anomaly_score,
            defect_area_ratio_estimate=self._defect_area_ratio_estimate(result),
            sample_suspicious_patches=[
                SuspiciousPatch(
                    patch_index=item.patch_index,
                    reference_offset=item.reference_offset,
                    score=item.score,
                    anomaly_score=item.anomaly_score,
                    suspicious=item.suspicious,
                    box=PatchBox(x=item.box.x, y=item.box.y, w=item.box.w, h=item.box.h),
                )
                for item in suspicious_patches[:20]
            ],
        )

    def _defect_area_ratio_estimate(self, result: AnomalyDetectionResult) -> float:
        suspicious_patches = [item for item in result.patch_scores if item.suspicious]
        if not suspicious_patches:
            return 0.0
        if not result.patch_scores:
            return 0.0
        image_width = max(item.box.x + item.box.w for item in result.patch_scores)
        image_height = max(item.box.y + item.box.h for item in result.patch_scores)
        total_area = image_width * image_height
        suspicious_area = sum(item.box.w * item.box.h for item in suspicious_patches)
        return min(1.0, suspicious_area / total_area)

    def _merged_original_boxes(
        self,
        result: AnomalyDetectionResult | None,
        coordinate_transform: CoordinateTransform,
        min_cluster_patches: int,
        processed_rgb: np.ndarray | None = None,
    ) -> list[MergedBox]:
        if result is None:
            return []
        patch_scores = list(result.patch_scores)
        suspicious_ratio = result.suspicious_patch_count / len(patch_scores) if patch_scores else 0.0
        resized_boxes = []
        if processed_rgb is not None and suspicious_ratio >= 0.5:
            resized_boxes = localize_pixel_defects(
                processed_rgb,
                anomaly_score=result.max_anomaly_score or 0.0,
            )
        if not resized_boxes:
            resized_boxes = localize_suspicious_patches(
                patch_scores,
                image_width=coordinate_transform.resize.output_width,
                image_height=coordinate_transform.resize.output_height,
                min_patch_count=min_cluster_patches,
            )
        if not resized_boxes:
            return []
        mapped: list[MergedBox] = []
        for box in resized_boxes:
            x1, y1, x2, y2 = coordinate_transform.map_box_to_original(
                (box.x, box.y, box.x + box.w, box.y + box.h)
            )
            if x2 <= x1 or y2 <= y1:
                continue
            mapped.append(
                MergedBox(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    score=box.score,
                    patch_count=box.patch_count,
                    label=box.label,
                )
            )
        return mapped

    def _resolve_reference_context(self, reference_id: str | None) -> RuntimeReferenceContext:
        try:
            entry = ReferenceRegistry().get(reference_id)
        except KeyError as exc:
            if reference_id:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            latest_metadata = resolve_project_path(self.settings.metadata_dir) / "latest_reference.json"
            if not latest_metadata.exists():
                return RuntimeReferenceContext(
                    reference_id="latest_reference",
                    metadata_path=latest_metadata,
                    anomaly_threshold=self.settings.anomaly_threshold,
                    good_area_threshold=self.settings.good_area_threshold,
                    bad_area_threshold=self.settings.bad_area_threshold,
                    min_defect_cluster_patches=self.settings.min_defect_cluster_patches,
                )
            metadata = self._load_reference_metadata(latest_metadata)
            thresholds = metadata.get("thresholds") or {}
            if metadata.get("preprocessing_version", "v1") == "v2" and "anomaly_threshold" not in thresholds:
                raise HTTPException(
                    status_code=409,
                    detail="The selected v2 reference has no calibrated anomaly threshold.",
                )
            return RuntimeReferenceContext(
                reference_id=metadata.get("reference_id") or metadata.get("reference_version") or "latest_reference",
                metadata_path=latest_metadata,
                anomaly_threshold=float(thresholds.get("anomaly_threshold", self.settings.anomaly_threshold)),
                good_area_threshold=float(thresholds.get("good_area_threshold", self.settings.good_area_threshold)),
                bad_area_threshold=float(thresholds.get("bad_area_threshold", self.settings.bad_area_threshold)),
                min_defect_cluster_patches=int(
                    thresholds.get("min_defect_cluster_patches", self.settings.min_defect_cluster_patches)
                ),
            )

        if entry.preprocessing_version == "v2" and "anomaly_threshold" not in entry.thresholds:
            raise HTTPException(
                status_code=409,
                detail="The selected v2 reference has no calibrated anomaly threshold.",
            )
        return RuntimeReferenceContext(
            reference_id=entry.reference_id,
            metadata_path=entry.metadata_path,
            anomaly_threshold=float(entry.thresholds.get("anomaly_threshold", self.settings.anomaly_threshold)),
            good_area_threshold=float(entry.thresholds.get("good_area_threshold", self.settings.good_area_threshold)),
            bad_area_threshold=float(entry.thresholds.get("bad_area_threshold", self.settings.bad_area_threshold)),
            min_defect_cluster_patches=int(
                entry.thresholds.get("min_defect_cluster_patches", self.settings.min_defect_cluster_patches)
            ),
        )

    def _box_area_ratio(self, boxes: list[MergedBox], width: int, height: int) -> float:
        if not boxes or width <= 0 or height <= 0:
            return 0.0
        return min(1.0, sum(box.area for box in boxes) / (width * height))

    def _detected_defects(self, inspection_id: str, boxes: list[MergedBox]) -> list[DetectedDefect]:
        return [
            DetectedDefect(
                inspection_id=inspection_id,
                x=box.x,
                y=box.y,
                w=box.w,
                h=box.h,
                score=box.score,
                label=box.label,
                area_pixels=box.area,
            )
            for box in boxes
        ]

    def _defect_boxes(self, boxes: list[MergedBox]) -> list[DefectBox]:
        return [
            DefectBox(x=box.x, y=box.y, w=box.w, h=box.h, score=box.score, label=box.label)
            for box in boxes
        ]

    def _defect_boxes_from_inspection(self, inspection: Inspection) -> list[DefectBox]:
        return [
            DefectBox(x=item.x, y=item.y, w=item.w, h=item.h, score=item.score, label=item.label)
            for item in inspection.defects
        ]

    def _prepare_patches(
        self,
        image_path: Path,
        width: int,
        height: int,
        metadata_path: Path | None = None,
    ) -> PreparedImage:
        del width, height
        metadata = self._load_reference_metadata(metadata_path) if metadata_path else {}
        version, config = configuration_from_reference_metadata(metadata)
        prepared = prepare_image(
            image_path,
            preprocessing_version=version,
            image_max_size=int(metadata.get("preprocess", {}).get("image_max_size", self.settings.image_max_size)),
            patch_size=int(metadata.get("preprocess", {}).get("patch_size", self.settings.patch_size)),
            patch_stride=int(metadata.get("preprocess", {}).get("patch_stride", self.settings.patch_stride)),
            preprocessing_config=config,
            fabric_type=metadata.get("fabric_type"),
            split="inference",
        )
        return prepared

    def _load_reference_metadata(self, metadata_path: Path) -> dict:
        if not metadata_path.exists():
            return {}
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=f"Reference metadata is invalid: {exc}") from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=409, detail="Reference metadata must be a JSON object.")
        return value

    def _grid_to_summary(self, grid: PatchGrid) -> PatchGridSummary:
        sample_boxes = [
            PatchBox(x=box.x, y=box.y, w=box.w, h=box.h)
            for box in grid.boxes[:20]
        ]
        return PatchGridSummary(
            image_width=grid.image_width,
            image_height=grid.image_height,
            patch_size=grid.patch_size,
            patch_stride=grid.patch_stride,
            rows=grid.rows,
            cols=grid.cols,
            patch_count=grid.patch_count,
            coverage_ratio=grid.coverage_ratio,
            sample_patches=sample_boxes,
        )

    def _preprocess_plan_from_prepared(
        self,
        inspection: Inspection,
        prepared: PreparedImage,
    ) -> PreprocessPlan:
        config = prepared.preprocessing_config
        patch_sizes = config.get("patch_sizes") or [config.get("patch_size", self.settings.patch_size)]
        patch_size = int(patch_sizes[0])
        if prepared.preprocessing_version == "v2":
            overlap = float(config.get("patch_overlap_ratio", 0.0))
            patch_stride = max(1, round(patch_size * (1.0 - overlap)))
            target_max_size = int(config.get("target_max_size") or max(prepared.width, prepared.height))
        else:
            patch_stride = int(config.get("patch_stride", self.settings.patch_stride))
            target_max_size = int(config.get("image_max_size", self.settings.image_max_size))
        return PreprocessPlan(
            target_max_size=target_max_size,
            original_width=inspection.image_width or prepared.coordinate_transform.source_width,
            original_height=inspection.image_height or prepared.coordinate_transform.source_height,
            resized_width=prepared.width,
            resized_height=prepared.height,
            scale=prepared.coordinate_transform.resize.scale,
            patch_size=patch_size,
            patch_stride=patch_stride,
            color_mode="RGB",
        )

    def _patch_grid_summary_from_prepared(self, prepared: PreparedImage) -> PatchGridSummary:
        patches = prepared.patches
        config = prepared.preprocessing_config
        patch_sizes = config.get("patch_sizes") or [config.get("patch_size", self.settings.patch_size)]
        patch_size = int(patch_sizes[0])
        if prepared.preprocessing_version == "v2":
            overlap = float(config.get("patch_overlap_ratio", 0.0))
            patch_stride = max(1, round(patch_size * (1.0 - overlap)))
        else:
            patch_stride = int(config.get("patch_stride", self.settings.patch_stride))
        coverage = np.zeros((prepared.height, prepared.width), dtype=bool)
        for patch in patches:
            x2 = min(prepared.width, patch.box.x + patch.box.w)
            y2 = min(prepared.height, patch.box.y + patch.box.h)
            coverage[patch.box.y:y2, patch.box.x:x2] = True
        return PatchGridSummary(
            image_width=prepared.width,
            image_height=prepared.height,
            patch_size=patch_size,
            patch_stride=patch_stride,
            rows=len({patch.box.y for patch in patches}),
            cols=len({patch.box.x for patch in patches}),
            patch_count=len(patches),
            coverage_ratio=float(coverage.mean()) if coverage.size else 0.0,
            sample_patches=[
                PatchBox(x=patch.box.x, y=patch.box.y, w=patch.box.w, h=patch.box.h)
                for patch in patches[:20]
            ],
        )
