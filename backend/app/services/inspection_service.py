from dataclasses import dataclass, replace
import json
import logging
import threading
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from pathlib import Path

import numpy as np

from backend.app.ai.anomaly_detector import AnomalyDetectionResult, FaissAnomalyDetector, ReferenceNotReadyError
from backend.app.ai.anomaly_branch import AnomalyDetector
from backend.app.ai.bbox import MergedBox, localize_pixel_defects, localize_suspicious_patches
from backend.app.ai.classifier import classify_inspection
from backend.app.ai.current_anomaly import CurrentAnomalyDetector, InvalidFabricReferenceError
from backend.app.ai.pattern_period import PatternPeriodEstimator, PatternPeriodResult
from backend.app.ai.fabric_decision import FabricDecision, decide_fabric_inspection
from backend.app.ai.fusion import FusionEngine, fusion_failure
from backend.app.ai.fusion_adapter import fusion_inputs_from_inspection
from backend.app.ai.quality_gate import ImageQualityGate, InvalidCaptureError, QualityGate, QualityGateResult
from backend.app.ai.patching import ImagePatch, PatchGrid, build_patch_grid
from backend.app.ai.preprocess import ImageReadError, build_preprocess_plan, inspect_image_file
from backend.app.ai.preprocessing.adapter import CoordinateTransform, PreparedImage, configuration_from_reference_metadata, prepare_image
from backend.app.core.config import get_settings
from backend.app.core.acquisition import AcquisitionMetadata, QualityGateConfig
from backend.app.core.inspection_contracts import InspectionContext, InspectionInput, InspectionResult
from backend.app.core.pattern import PatternEstimatorConfig
from backend.app.core.segmentation_shadow import SupervisedShadowConfig, SupervisedShadowObservation
from backend.app.core.fusion import DefectEvidence, FusionPolicy
from backend.app.core.qc import QCDecision, QCPolicy
from backend.app.core.paths import resolve_project_path
from backend.app.core.runtime_profiling import RuntimeProfiler, current_profiler, runtime_stage
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
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner, SupervisedShadowRunner
from backend.app.services.qc_decision import QCDecisionEngine, qc_failure, qc_input_from_evidence
from backend.app.utils.visualization import save_annotated_image


logger = logging.getLogger(__name__)
shadow_audit_logger = logging.getLogger("backend.supervised_shadow.audit")
fusion_audit_logger = logging.getLogger("backend.hybrid_fusion.audit")
qc_audit_logger = logging.getLogger("backend.hybrid_qc.audit")
runtime_audit_logger = logging.getLogger("backend.runtime_profile.audit")
_shadow_log_lock = threading.Lock()


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
    def __init__(self, db: Session, *, anomaly_detector: AnomalyDetector | None = None,
                 quality_gate: QualityGate | None = None,
                 pattern_estimator: PatternPeriodEstimator | None = None,
                 supervised_shadow: SupervisedShadowRunner | None = None,
                 fusion_engine: FusionEngine | None = None,
                 qc_engine: QCDecisionEngine | None = None) -> None:
        self.db = db
        self.storage = StorageService()
        self.settings = get_settings()
        self.anomaly_detector = anomaly_detector if anomaly_detector is not None else CurrentAnomalyDetector()
        self.quality_gate = quality_gate if quality_gate is not None else ImageQualityGate()
        self.pattern_estimator = pattern_estimator if pattern_estimator is not None else PatternPeriodEstimator()
        self.supervised_shadow = supervised_shadow if supervised_shadow is not None else ProcessSupervisedShadowRunner()
        self.fusion_engine = fusion_engine
        self.qc_engine = qc_engine

    def create_fabric_inspection(self, image: UploadFile, fabric_id: str, *,
                                 metadata: AcquisitionMetadata | None = None) -> InspectionCreateResponse:
        """Run the demo dense-anomaly pipeline against exactly one active fabric reference."""
        context = self._resolve_fabric_context(fabric_id)
        reference = context.reference
        backbone = context.backbone
        try:
            stored_upload = self.storage.save_upload(image)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        annotated = None
        persisted = False
        try:
            inspection_input = InspectionInput(
                image_path=stored_upload.absolute_path, fabric_id=fabric_id,
                metadata=metadata if metadata is not None else AcquisitionMetadata(),
            )
            try:
                quality = self._evaluate_capture(inspection_input, context)
            except InvalidCaptureError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            try:
                image_info = inspect_image_file(stored_upload.absolute_path)
            except ImageReadError as exc:
                raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.") from exc

            try:
                result = self._inspect_fabric(
                    inspection_input, context, quality_gate_result=quality,
                )
            except InvalidFabricReferenceError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Active fabric reference is invalid or incompatible: {exc}",
                ) from exc

            features = result.evidence.features
            extractor_info = result.evidence.extractor_info
            anomaly = result.evidence.anomaly
            original_boxes = list(result.candidates)
            decision = result.decision
            defect_area_ratio = result.defect_area_ratio
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
            self._log_supervised_shadow(result.known_defects_shadow, inspection_id=inspection_id,
                                        fabric_id=fabric_id, reference_id=reference.reference_id)
            self._log_fusion(result.defect_evidence, inspection_id=inspection_id,
                             fabric_id=fabric_id, legacy_qc_label=(result.legacy_decision or decision).label,
                             hybrid_qc_active=bool(result.qc_evaluation and result.qc_evaluation.applied))
            self._log_qc(result.qc_evaluation, inspection_id=inspection_id, fabric_id=fabric_id,
                         legacy_label=(result.legacy_decision or decision).label, selected_label=decision.label)
            pixel_threshold = float(reference.thresholds["pixel_threshold"])
            return self._to_response(
                inspection,
                embedding_summary=EmbeddingSummary(
                    model_name=extractor_info.architecture,
                    device=extractor_info.device,
                    requested_device=self.settings.device,
                    use_pretrained_weights=extractor_info.use_pretrained_weights,
                    embedding_dim=extractor_info.feature_dim,
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

    def _resolve_fabric_context(self, fabric_id: str) -> InspectionContext:
        try:
            registry = FabricRegistry()
            reference = registry.get_active_reference(fabric_id)
        except FabricRegistryError as exc:
            status_code = 400 if str(exc).startswith("Unknown fabric_id") else 409
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        quality_config = self._quality_gate_policy(registry, fabric_id)
        pattern_config, pattern_error = self._pattern_observation_policy(registry, fabric_id)
        shadow_config, shadow_error = self._supervised_shadow_policy(registry, fabric_id)
        fusion_policy, fusion_error = self._fusion_policy(registry, fabric_id)
        qc_policy, qc_error, qc_mode = self._hybrid_qc_policy(registry, fabric_id)
        return InspectionContext(
            reference=reference,
            backbone=registry.shared_backbone,
            device=self.settings.device,
            default_image_max_size=self.settings.image_max_size,
            quality_gate_config=quality_config,
            pattern_estimator_config=pattern_config,
            pattern_observation_error=pattern_error,
            supervised_shadow_config=shadow_config,
            supervised_shadow_error=shadow_error,
            fusion_policy=fusion_policy,
            fusion_error=fusion_error,
            qc_policy=qc_policy,
            qc_policy_error=qc_error,
            hybrid_qc_mode=qc_mode,
        )

    def _hybrid_qc_policy(self, registry, fabric_id):
        mode = self.settings.hybrid_qc_mode if self.settings.hybrid_qc_enabled else "observe_only"
        try:
            override = registry.get_fabric(fabric_id).get("qc", {})
            if not isinstance(override, dict):
                raise ValueError("fabric qc must be an object")
            enabled = override.get("enabled", self.settings.hybrid_qc_enabled)
            if type(enabled) is not bool:
                mode = "enforce"  # Ambiguous enabled control must not silently retain a GOOD acceptance.
                raise ValueError("qc.enabled must be a boolean")
            if not enabled:
                return None, None, "observe_only"
            mode = override.get("mode", self.settings.hybrid_qc_mode)
            if type(mode) is not str or mode not in {"observe_only", "enforce"}:
                mode = "enforce"
                raise ValueError("Unsupported QC runtime mode")
            values = {}
            if self.settings.hybrid_qc_config_path:
                values = json.loads(resolve_project_path(self.settings.hybrid_qc_config_path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError("QC policy must be an object")
            values.update({key: value for key, value in override.items() if key not in {"enabled", "mode"}})
            policy = QCPolicy.model_validate(values)
            if mode == "enforce" and not policy.allowed_fabric_ids:
                raise ValueError("Enforcement requires an explicit fabric scope")
            return policy, None, mode
        except Exception:
            return None, "QC_CONFIGURATION_ERROR", mode

    def _evaluate_hybrid_qc(self, fabric_id, context, fused):
        if context.qc_policy is None and context.qc_policy_error is None:
            return None
        try:
            if context.qc_policy_error is not None:
                result = qc_failure("CONFIGURATION_ERROR")
            else:
                inputs = qc_input_from_evidence(fabric_id, fused)
                result = (self.qc_engine or QCDecisionEngine(context.qc_policy)).evaluate(inputs)
                result = QCDecision.model_validate(result.model_dump())
                if result.policy != context.qc_policy:
                    raise ValueError("QC evaluation policy mismatch")
        except Exception:
            result = qc_failure("EVALUATION_ERROR")
        fields = result.model_dump()
        fields.update(runtime_mode=context.hybrid_qc_mode, applied=context.hybrid_qc_mode == "enforce")
        return QCDecision.model_validate(fields)

    def _log_qc(self, result, *, inspection_id, fabric_id, legacy_label, selected_label):
        if result is None:
            return
        try:
            payload = {"inspection_id": inspection_id, "fabric_id": fabric_id, "legacy_qc_label": legacy_label,
                       "selected_qc_label": selected_label, "qc_evaluation": result.model_dump(mode="json")}
            with _shadow_log_lock:
                qc_audit_logger.setLevel(logging.INFO)
                if not qc_audit_logger.hasHandlers():
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(message)s"))
                    qc_audit_logger.addHandler(handler)
            qc_audit_logger.info("qc_evaluation %s", json.dumps(payload, ensure_ascii=True, allow_nan=False), extra={"qc_evaluation": payload})
        except Exception:
            pass

    def _fusion_policy(self, registry: FabricRegistry, fabric_id: str) -> tuple[FusionPolicy | None, str | None]:
        try:
            override = registry.get_fabric(fabric_id).get("fusion", {})
            if not isinstance(override, dict):
                raise ValueError("fabric fusion must be an object")
            enabled = override.get("enabled", self.settings.fusion_enabled)
            if type(enabled) is not bool:
                raise ValueError("fusion.enabled must be a boolean")
            if not enabled:
                return None, None
            values = {}
            if self.settings.fusion_config_path:
                values = json.loads(resolve_project_path(self.settings.fusion_config_path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError("Fusion configuration must be an object")
            values.update({key: value for key, value in override.items() if key != "enabled"})
            return FusionPolicy.model_validate(values), None
        except Exception:
            return None, "fusion_configuration_error"

    def _fuse_evidence(self, inspection, context, evidence, original_boxes, quality, shadow) -> DefectEvidence | None:
        if context.fusion_error is not None:
            return fusion_failure("fusion_configuration_error")
        if context.fusion_policy is None:
            return None
        try:
            supervised, anomaly, capture = fusion_inputs_from_inspection(inspection, context, evidence, original_boxes, quality, shadow)
            result = (self.fusion_engine or FusionEngine(context.fusion_policy)).fuse(supervised, anomaly, capture=capture)
            result = DefectEvidence.model_validate(result.model_dump())
            if result.policy != context.fusion_policy:
                raise ValueError("Fusion output policy does not match configuration")
            return result
        except Exception:
            return fusion_failure("fusion_engine_error")

    def _log_fusion(self, result: DefectEvidence | None, *, inspection_id: str, fabric_id: str, legacy_qc_label: str, hybrid_qc_active: bool = False):
        if result is None:
            return
        try:
            payload = {"inspection_id": inspection_id, "fabric_id": fabric_id, "qc_source": "hybrid_qc" if hybrid_qc_active else "legacy_anomaly",
                       "legacy_qc_label": legacy_qc_label, "hybrid_qc_active": hybrid_qc_active,
                       "anomaly_box_count": len(result.anomaly_boxes), "defect_evidence": result.model_dump(mode="json", exclude={"anomaly_boxes"})}
            with _shadow_log_lock:
                fusion_audit_logger.setLevel(logging.INFO)
                if not fusion_audit_logger.hasHandlers():
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(message)s"))
                    fusion_audit_logger.addHandler(handler)
            fusion_audit_logger.info("defect_evidence %s", json.dumps(payload, ensure_ascii=True, allow_nan=False), extra={"defect_evidence": payload})
        except Exception:
            pass  # Diagnostic logging is not a business/QC decision or a persistence prerequisite.

    def _supervised_shadow_policy(self, registry: FabricRegistry, fabric_id: str
                                  ) -> tuple[SupervisedShadowConfig | None, str | None]:
        """Bad optional shadow configuration never invalidates the active anomaly reference."""
        try:
            override = registry.get_fabric(fabric_id).get("supervised", {})
            if not isinstance(override, dict):
                raise ValueError("fabric supervised must be an object")
            enabled = override.get("enabled", self.settings.supervised_shadow_enabled)
            if type(enabled) is not bool:
                raise ValueError("supervised.enabled must be a boolean")
            if not enabled:
                return None, None  # Do not open files, load a model or spawn a process when OFF.
            values = {}
            if self.settings.supervised_shadow_config_path:
                values = json.loads(resolve_project_path(self.settings.supervised_shadow_config_path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError("Shadow configuration must be an object")
            values.update({key: value for key, value in override.items() if key != "enabled"})
            return SupervisedShadowConfig.model_validate(values), None
        except Exception:
            return None, "configuration_error"

    def _observe_supervised_shadow(self, inspection: InspectionInput, context: InspectionContext
                                   ) -> SupervisedShadowObservation | None:
        if context.supervised_shadow_error is not None:
            return SupervisedShadowObservation(status="configuration_error", reason="configuration_error")
        if context.supervised_shadow_config is None:
            return None
        try:
            observation = self.supervised_shadow.observe(inspection.image_path, inspection.fabric_id, context.supervised_shadow_config)
            # Validate even injected/test runners; no malformed shadow payload may break a valid inspection.
            observation = SupervisedShadowObservation.model_validate(observation.model_dump())
            config = context.supervised_shadow_config
            if observation.requested_model_version != config.expected_model_version or (observation.status == "observed" and (
                    observation.checkpoint_sha256 != config.checkpoint_sha256 or inspection.fabric_id not in config.allowed_fabric_ids
                    or not config.trusted_local or (observation.synthetic_checkpoint and not config.allow_synthetic))):
                raise ValueError("Shadow output does not match configured provenance")
            return observation
        except Exception:
            return SupervisedShadowObservation(status="error", reason="observer_error",
                                                requested_model_version=context.supervised_shadow_config.expected_model_version)

    def _log_supervised_shadow(self, observation: SupervisedShadowObservation | None, *,
                               inspection_id: str, fabric_id: str, reference_id: str) -> None:
        if observation is None:
            return
        try:
            payload = {"inspection_id": inspection_id, "fabric_id": fabric_id,
                       "anomaly_reference_id": reference_id, **observation.audit_payload()}
            with _shadow_log_lock:
                shadow_audit_logger.setLevel(logging.INFO)
                if not shadow_audit_logger.hasHandlers():
                    # Uvicorn does not necessarily configure the root logger. No global/root change.
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(message)s"))
                    shadow_audit_logger.addHandler(handler)
            shadow_audit_logger.info("known_defects_shadow %s", json.dumps(payload, ensure_ascii=True, allow_nan=False),
                                     extra={"known_defects_shadow": payload})
        except Exception:
            pass  # Optional diagnostics/log-sink failure must not turn persisted inspection into HTTP 500.

    def _pattern_observation_policy(self, registry: FabricRegistry, fabric_id: str
                                    ) -> tuple[PatternEstimatorConfig | None, str | None]:
        """Observation misconfiguration must never reject an otherwise valid inspection."""
        try:
            override = registry.get_fabric(fabric_id).get("pattern_estimator", {})
            if not isinstance(override, dict):
                raise ValueError("fabric pattern_estimator must be an object")
            enabled = override.get("enabled", self.settings.pattern_estimator_enabled)
            if not isinstance(enabled, bool):
                raise ValueError("pattern_estimator.enabled must be a boolean")
            if not enabled:
                return None, None
            values = {}
            if self.settings.pattern_estimator_config_path:
                values = json.loads(resolve_project_path(self.settings.pattern_estimator_config_path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError("pattern estimator configuration must be an object")
            values.update({key: value for key, value in override.items() if key != "enabled"})
            return PatternEstimatorConfig.model_validate(values), None
        except (OSError, ValueError):
            logger.warning("Pattern observation configuration unavailable: fabric_id=%s", fabric_id)
            return None, "configuration_error"

    def _observe_pattern(self, inspection: InspectionInput, context: InspectionContext
                         ) -> PatternPeriodResult | None:
        if context.pattern_observation_error is not None:
            return PatternPeriodResult(reason=context.pattern_observation_error)
        if context.pattern_estimator_config is None:
            return None
        try:
            observation = self.pattern_estimator.estimate_path(inspection.image_path, context.pattern_estimator_config)
            logger.info(
                "Pattern observation: fabric_id=%s period_x_px=%s period_y_px=%s confidence=%.3f is_periodic=%s",
                inspection.fabric_id, observation.period_x_px, observation.period_y_px,
                observation.confidence, observation.is_periodic,
            )
            return observation
        except Exception:
            logger.warning("Pattern observation failed: fabric_id=%s", inspection.fabric_id)
            return PatternPeriodResult(reason="observer_error")

    def _quality_gate_policy(self, registry: FabricRegistry, fabric_id: str) -> QualityGateConfig | None:
        try:
            override = registry.get_fabric(fabric_id).get("quality_gate", {})
            if not isinstance(override, dict):
                raise ValueError("fabric quality_gate must be an object")
            enabled = override.get("enabled", self.settings.quality_gate_enabled)
            if not isinstance(enabled, bool):
                raise ValueError("quality_gate.enabled must be a boolean")
            if not enabled:
                return None
            values = {}
            if self.settings.quality_gate_config_path:
                values = json.loads(resolve_project_path(self.settings.quality_gate_config_path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError("quality gate configuration must be an object")
            values.update({key: value for key, value in override.items() if key != "enabled"})
            return QualityGateConfig.model_validate(values)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=f"Invalid quality gate configuration: {exc}") from exc

    def _evaluate_capture(self, inspection: InspectionInput, context: InspectionContext) -> QualityGateResult | None:
        if context.quality_gate_config is None:
            return None
        result = self.quality_gate.evaluate(inspection, context.quality_gate_config)
        if not result.valid:
            raise InvalidCaptureError(result)
        return result

    def _inspect_fabric(self, inspection: InspectionInput, context: InspectionContext, *,
                        quality_gate_result: QualityGateResult | None = None) -> InspectionResult:
        if not self.settings.runtime_profiling_enabled or current_profiler() is not None:
            return self._inspect_fabric_core(inspection, context, quality_gate_result=quality_gate_result)
        profiler = RuntimeProfiler(device=context.device)
        try:
            with profiler:
                result = self._inspect_fabric_core(inspection, context, quality_gate_result=quality_gate_result)
            return replace(result, runtime_profile=profiler.profile)
        finally:
            # Internal diagnostic only. Logging failures never affect inspection/cleanup.
            try:
                payload = profiler.profile.to_dict()
                with _shadow_log_lock:
                    runtime_audit_logger.setLevel(logging.INFO)
                    if not runtime_audit_logger.hasHandlers():
                        handler = logging.StreamHandler()
                        handler.setFormatter(logging.Formatter("%(message)s"))
                        runtime_audit_logger.addHandler(handler)
                runtime_audit_logger.info("runtime_profile %s", json.dumps(payload, allow_nan=False), extra={"runtime_profile": payload})
            except Exception:
                pass

    def _inspect_fabric_core(self, inspection: InspectionInput, context: InspectionContext, *,
                             quality_gate_result: QualityGateResult | None = None) -> InspectionResult:
        """Generate evidence, then apply the existing localization and QC policy."""
        if quality_gate_result is None:
            with runtime_stage("preprocessing"):
                quality_gate_result = self._evaluate_capture(inspection, context)
        elif not quality_gate_result.valid:
            raise InvalidCaptureError(quality_gate_result)
        with runtime_stage("anomaly"):
            evidence = self.anomaly_detector.inspect(inspection, context)
        features = evidence.features
        anomaly = evidence.anomaly
        with runtime_stage("postprocessing"):
            original_boxes = self._map_dense_boxes_to_original(
                list(anomaly.boxes), scale=features.scale,
                width=features.original_width, height=features.original_height,
            )
            decision = decide_fabric_inspection(
                image_score=anomaly.image_score, boxes=original_boxes,
                thresholds=context.reference.thresholds, localized_score=anomaly.localized_score,
            )
        if context.pattern_estimator_config is not None or context.pattern_observation_error is not None:
            with runtime_stage("preprocessing"):
                pattern = self._observe_pattern(inspection, context)
        else:
            pattern = self._observe_pattern(inspection, context)
        if context.supervised_shadow_config is not None or context.supervised_shadow_error is not None:
            with runtime_stage("segmentation"):
                shadow = self._observe_supervised_shadow(inspection, context)
        else:
            shadow = self._observe_supervised_shadow(inspection, context)
        if context.fusion_policy is not None or context.fusion_error is not None:
            with runtime_stage("fusion"):
                fused = self._fuse_evidence(inspection, context, evidence, original_boxes, quality_gate_result, shadow)
        else:
            fused = self._fuse_evidence(inspection, context, evidence, original_boxes, quality_gate_result, shadow)
        with runtime_stage("postprocessing"):
            qc = self._evaluate_hybrid_qc(inspection.fabric_id, context, fused)
        legacy_decision = decision
        if qc is not None and qc.applied:
            decision = FabricDecision(label=qc.label, reason="QC_" + qc.reason)
        return InspectionResult(
            evidence=evidence,
            candidates=tuple(original_boxes),
            decision=decision,
            defect_area_ratio=self._box_area_ratio(original_boxes, features.original_width, features.original_height),
            quality_gate_result=quality_gate_result,
            # Deliberately after evidence and QC; no consumer in patching/scoring/decision.
            pattern_period_result=pattern,
            known_defects_shadow=shadow,
            defect_evidence=fused,
            qc_evaluation=qc,
            legacy_decision=legacy_decision if qc is not None else None,
        )

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
