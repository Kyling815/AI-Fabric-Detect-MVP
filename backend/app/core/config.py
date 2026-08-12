from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Fabric Detect MVP"
    app_env: str = "local"
    debug: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    database_url: str = "sqlite:///./storage/app.db"

    # Dataset roots:
    # - Data/raw keeps original source datasets untouched.
    # - Data/processed is reserved for optional prepared image datasets.
    # - Data/processed/reference_ready can hold curated normal images for future multi-fabric references.
    data_raw_dir: str = "Data/raw"
    raw_company_dir: str = "Data/raw/Company"
    raw_external_dir: str = "Data/raw/external"
    data_processed_dir: str = "Data/processed"
    reference_ready_dir: str = "Data/processed/reference_ready"

    # Generated AI/runtime artifacts live under storage, not under Data.
    # This includes uploads, annotated outputs, FAISS indexes, metadata registries, and trained checkpoints.
    storage_dir: str = "storage"
    uploads_dir: str = "storage/uploads"
    annotated_dir: str = "storage/annotated"
    indexes_dir: str = "storage/indexes"
    metadata_dir: str = "storage/metadata"
    reference_registry_path: str = "storage/metadata/reference_registry.json"
    default_reference_id: str = ""

    dme_dataset_dir: str = "Data/raw/external/20250129_Fabric_defects/Dataset"
    wfdd_dataset_dir: str = "Data/raw/external/WFDD"
    lusitano_dataset_dir: str = "Data/raw/external/Lusitano"
    mvtec_dataset_dir: str = "Data/Test"

    model_name: str = "resnet18"
    use_pretrained_weights: bool = False
    use_projection_head: bool = False
    projection_head_path: str = ""
    projection_dim: int = 128
    device: str = "auto"
    image_max_size: int = 768
    model_input_size: int = 224
    patch_size: int = 96
    patch_stride: int = 48
    batch_size: int = 16
    max_upload_size_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    preprocessing_version: str = "v1"
    preprocessing_config_path: str = ""
    reference_max_images: int = 25
    anomaly_threshold: float = 0.75
    good_area_threshold: float = 0.001
    bad_area_threshold: float = 0.01
    min_defect_cluster_patches: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FABRIC_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
