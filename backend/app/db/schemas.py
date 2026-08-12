from datetime import datetime

from pydantic import BaseModel, Field


class DefectBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str = "suspicious_defect"


class ImageMetadata(BaseModel):
    width: int
    height: int
    mode: str
    content_type: str | None = None
    file_size_bytes: int


class PreprocessPlan(BaseModel):
    target_max_size: int
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float
    patch_size: int
    patch_stride: int
    color_mode: str = "RGB"


class PatchBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class PatchGridSummary(BaseModel):
    image_width: int
    image_height: int
    patch_size: int
    patch_stride: int
    rows: int
    cols: int
    patch_count: int
    coverage_ratio: float
    sample_patches: list[PatchBox] = Field(default_factory=list)


class EmbeddingSummary(BaseModel):
    model_name: str
    device: str
    requested_device: str
    use_pretrained_weights: bool
    embedding_dim: int
    patch_count: int
    batch_size: int
    model_input_size: int
    l2_normalized: bool


class SuspiciousPatch(BaseModel):
    patch_index: int
    reference_offset: int
    score: float
    anomaly_score: float
    suspicious: bool
    box: PatchBox


class AnomalySummary(BaseModel):
    status: str
    reference_version: str | None = None
    faiss_metric: str | None = None
    threshold: float
    patch_count: int
    suspicious_patch_count: int
    max_anomaly_score: float | None = None
    mean_anomaly_score: float | None = None
    defect_area_ratio_estimate: float
    sample_suspicious_patches: list[SuspiciousPatch] = Field(default_factory=list)


class InspectionCreateResponse(BaseModel):
    inspection_id: str
    reference_id: str | None = None
    final_result: str
    confidence: float | None = None
    defect_area_ratio: float
    defect_count: int
    boxes: list[DefectBox] = Field(default_factory=list)
    original_image_url: str | None = None
    annotated_image_url: str | None = None
    image_metadata: ImageMetadata | None = None
    preprocess_plan: PreprocessPlan | None = None
    patch_grid: PatchGridSummary | None = None
    embedding_summary: EmbeddingSummary | None = None
    anomaly_summary: AnomalySummary | None = None
    model_version: str
    created_at: datetime | None = None


class InspectionListItem(BaseModel):
    inspection_id: str
    filename: str
    final_result: str
    defect_area_ratio: float
    defect_count: int
    created_at: datetime


class InspectionListResponse(BaseModel):
    items: list[InspectionListItem]


class DatasetPathItem(BaseModel):
    name: str
    path: str
    exists: bool


class DatasetPathsResponse(BaseModel):
    items: list[DatasetPathItem]


class DatasetClassSummary(BaseModel):
    name: str
    role: str
    path: str
    image_count: int
    mask_count: int = 0


class DatasetSummary(BaseModel):
    name: str
    display_name: str
    dataset_type: str
    source_path: str
    exists: bool
    normal_paths: list[str] = Field(default_factory=list)
    classes: list[DatasetClassSummary] = Field(default_factory=list)
    notes: str | None = None


class DatasetDiscoveryResponse(BaseModel):
    datasets: list[DatasetSummary]
    total_images: int


class ReferenceBuildRequest(BaseModel):
    dataset_name: str = "dme_20250129"
    fabric_type: str | None = None
    normal_path: str | None = None
    reference_id: str | None = None
    max_images: int | None = Field(default=None, ge=1)
    preprocessing_version: str | None = Field(default=None, pattern="^v[12]$")
    preprocessing_config_path: str | None = None


class ReferenceBuildResponse(BaseModel):
    status: str
    message: str
    dataset_name: str
    reference_id: str | None = None
    reference_version: str | None = None
    normal_path: str | None = None
    normal_image_count: int = 0
    processed_image_count: int = 0
    skipped_image_count: int = 0
    patch_count: int = 0
    embedding_dim: int | None = None
    embeddings_path: str | None = None
    faiss_index_path: str | None = None
    faiss_metric: str | None = None
    metadata_path: str | None = None


class ReferenceStatusResponse(BaseModel):
    status: str
    reference_id: str | None = None
    active_reference_version: str | None = None
    model_name: str
    patch_size: int
    patch_stride: int
    anomaly_threshold: float
    embeddings_path: str | None = None
    faiss_index_path: str | None = None
    faiss_metric: str | None = None
    metadata_path: str | None = None
    patch_count: int = 0
    image_count: int = 0


class ReferenceRegistryItem(BaseModel):
    reference_id: str
    reference_version: str
    dataset_name: str
    fabric_type: str | None = None
    source_layer: str
    metadata_path: str
    embeddings_path: str
    faiss_index_path: str
    faiss_metric: str
    embedding_dim: int
    patch_count: int
    processed_image_count: int
    created_at: str
    thresholds: dict = Field(default_factory=dict)
    preprocessing_version: str = "v1"
    configuration_hash: str | None = None


class ReferenceRegistryListResponse(BaseModel):
    active_reference_version: str | None = None
    default_reference_id: str | None = None
    references: list[ReferenceRegistryItem] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    inspection_id: str
    user_label: str
    corrected_result: str | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    inspection_id: str
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    device: str
    use_pretrained_weights: bool
    use_projection_head: bool
    projection_head_path: str
    projection_dim: int
    image_max_size: int
    model_input_size: int
    patch_size: int
    patch_stride: int
    batch_size: int
    default_reference_id: str
    reference_registry_path: str
    anomaly_threshold: float
    good_area_threshold: float
    bad_area_threshold: float
    detection_core: str
