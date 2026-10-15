"""Callable adapter around the current CNN/FAISS pipeline, independent of HTTP."""

from backend.app.ai.dense_anomaly import DenseReferenceDetector
from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.reference_compatibility import validate_current_reference
from backend.app.core.inspection_contracts import DenseAnomalyEvidence, InspectionContext, InspectionInput
from backend.app.core.paths import resolve_project_path
from backend.app.core.runtime_profiling import runtime_stage


class InvalidFabricReferenceError(RuntimeError):
    """The same reference failures previously translated to HTTP 409 by the service."""


class CurrentAnomalyDetector:
    def inspect(self, inspection: InspectionInput, context: InspectionContext) -> DenseAnomalyEvidence:
        with runtime_stage("anomaly"):
            return self._inspect(inspection, context)

    def _inspect(self, inspection: InspectionInput, context: InspectionContext) -> DenseAnomalyEvidence:
        backbone = context.backbone
        extractor = get_shared_dense_extractor(
            str(backbone["backbone_version"]),
            context.device,
            bool(backbone.get("use_pretrained_weights", True)),
            str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
        )
        features = extractor.extract_path(
            inspection.image_path,
            image_max_size=int(backbone.get("image_max_size", context.default_image_max_size)),
        )
        try:
            detector = DenseReferenceDetector(context.reference.metadata_path)
            validate_current_reference(detector, inspection, context)
            anomaly = detector.score(
                features.embeddings,
                features.boxes,
                grid_rows=features.grid_rows,
                grid_cols=features.grid_cols,
                image_width=features.processed_width,
                image_height=features.processed_height,
                thresholds=context.reference.thresholds,
            )
        except (OSError, KeyError, ValueError) as exc:
            # Keep model-loading/native runtime failures outside this translation,
            # exactly as in the baseline service's reference-only try/except.
            raise InvalidFabricReferenceError(str(exc)) from exc
        return DenseAnomalyEvidence(features=features, extractor_info=extractor.info, anomaly=anomaly)
