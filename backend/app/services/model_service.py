from backend.app.core.config import get_settings
from backend.app.db.schemas import ModelInfoResponse


class ModelService:
    def get_model_info(self) -> ModelInfoResponse:
        settings = get_settings()
        return ModelInfoResponse(
            model_name=settings.model_name,
            device=settings.device,
            use_pretrained_weights=settings.use_pretrained_weights,
            use_projection_head=settings.use_projection_head,
            projection_head_path=settings.projection_head_path,
            projection_dim=settings.projection_dim,
            image_max_size=settings.image_max_size,
            model_input_size=settings.model_input_size,
            patch_size=settings.patch_size,
            patch_stride=settings.patch_stride,
            batch_size=settings.batch_size,
            default_reference_id=settings.default_reference_id,
            reference_registry_path=settings.reference_registry_path,
            anomaly_threshold=settings.anomaly_threshold,
            good_area_threshold=settings.good_area_threshold,
            bad_area_threshold=settings.bad_area_threshold,
            detection_core="pretrained CNN / vision embeddings + patch anomaly detection",
        )
