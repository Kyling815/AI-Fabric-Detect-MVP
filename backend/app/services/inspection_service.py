from dataclasses import dataclass
import json
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from pathlib import Path

import numpy as np

from backend.app.ai.anomaly_detector import AnomalyDetectionResult, FaissAnomalyDetector, ReferenceNotReadyError
from backend.app.ai.bbox import MergedBox, filter_boxes_by_cluster_size, merge_suspicious_patches, scale_boxes_to_original
from backend.app.ai.classifier import classify_inspection
from backend.app.ai.patching import ImagePatch, PatchGrid, build_patch_grid, extract_image_patches
from backend.app.ai.preprocess import ImageReadError, build_preprocess_plan, inspect_image_file, load_resized_rgb_image
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
from backend.app.services.storage_service import StorageService
from backend.app.services.reference_registry import ReferenceRegistry
from backend.app.utils.visualization import save_annotated_image


@dataclass(frozen=True)
class RuntimeEmbeddingResult:
    summary: EmbeddingSummary
    embeddings: np.ndarray
    patches: list[ImagePatch]


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

    def create_placeholder_inspection(self, image: UploadFile, reference_id: str | None = None) -> InspectionCreateResponse:
        reference_context = self._resolve_reference_context(reference_id)
        stored_upload = self.storage.save_upload(image)
        try:
            image_info = inspect_image_file(stored_upload.absolute_path)
        except ImageReadError as exc:
            self.storage.delete_stored_file(stored_upload.relative_path)
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
            image_info.width,
            image_info.height,
            reference_context.min_defect_cluster_patches,
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
        annotated = None
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
        inspection = crud.create_inspection(self.db, inspection)
        crud.create_detected_defects(self.db, self._detected_defects(inspection.id, original_boxes))
        return self._to_response(
            inspection,
            reference_id=reference_context.reference_id,
            embedding_summary=embedding_result.summary,
            anomaly_summary=anomaly_result.summary,
            boxes=self._defect_boxes(original_boxes),
        )

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
    ) -> InspectionCreateResponse:
        return InspectionCreateResponse(
            inspection_id=inspection.id,
            reference_id=reference_id or inspection.model_version,
            final_result=inspection.final_result,
            confidence=inspection.confidence,
            defect_area_ratio=inspection.defect_area_ratio,
            defect_count=inspection.defect_count,
            boxes=boxes if boxes is not None else self._defect_boxes_from_inspection(inspection),
            original_image_url=self.storage.to_public_url(inspection.original_image_path),
            annotated_image_url=self.storage.to_public_url(inspection.annotated_image_path),
            image_metadata=self._image_metadata(inspection),
            preprocess_plan=self._preprocess_plan(inspection),
            patch_grid=self._patch_grid_summary(inspection),
            embedding_summary=embedding_summary,
            anomaly_summary=anomaly_summary,
            model_version=inspection.model_version,
            created_at=inspection.created_at,
        )

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
        from backend.app.ai.feature_extractor import CNNFeatureExtractor

        patches = self._prepare_patches(image_path, width, height)
        if not patches:
            raise HTTPException(status_code=400, detail="Image produced no patches.")

        extractor = CNNFeatureExtractor(
            model_name=self.settings.model_name,
            device=self.settings.device,
            use_pretrained_weights=self.settings.use_pretrained_weights,
            model_input_size=self.settings.model_input_size,
        )
        embeddings = extractor.extract((patch.image for patch in patches), batch_size=self.settings.batch_size)
        info = extractor.info
        projection_path = self._active_projection_head_path(metadata_path)
        if projection_path:
            from backend.app.ai.projection_head import project_embeddings

            embeddings, projection_info = project_embeddings(
                embeddings,
                checkpoint_path=projection_path,
                device=info.device,
                batch_size=max(self.settings.batch_size, 1),
            )
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
        )

    def _active_projection_head_path(self, metadata_path: Path) -> Path | None:
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                projection = metadata.get("projection_head")
                if projection and projection.get("checkpoint_path"):
                    return Path(projection["checkpoint_path"])
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
            result = detector.score(embedding_result.embeddings, patch_boxes)
        except (ReferenceNotReadyError, FileNotFoundError, KeyError, ValueError):
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
        width: int,
        height: int,
        min_cluster_patches: int,
    ) -> list[MergedBox]:
        if result is None:
            return []
        resized_boxes = merge_suspicious_patches(list(result.patch_scores), merge_gap=0)
        resized_boxes = filter_boxes_by_cluster_size(resized_boxes, min_cluster_patches)
        if not resized_boxes:
            return []
        plan = build_preprocess_plan(
            width=width,
            height=height,
            target_max_size=self.settings.image_max_size,
            patch_size=self.settings.patch_size,
            patch_stride=self.settings.patch_stride,
        )
        return scale_boxes_to_original(resized_boxes, plan.scale)

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
            metadata = json.loads(latest_metadata.read_text(encoding="utf-8"))
            thresholds = metadata.get("thresholds") or {}
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

    def _prepare_patches(self, image_path: Path, width: int, height: int) -> list[ImagePatch]:
        plan = build_preprocess_plan(
            width=width,
            height=height,
            target_max_size=self.settings.image_max_size,
            patch_size=self.settings.patch_size,
            patch_stride=self.settings.patch_stride,
        )
        resized_image = load_resized_rgb_image(image_path, plan)
        grid = build_patch_grid(
            image_width=resized_image.width,
            image_height=resized_image.height,
            patch_size=plan.patch_size,
            patch_stride=plan.patch_stride,
        )
        return extract_image_patches(resized_image, grid)

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
