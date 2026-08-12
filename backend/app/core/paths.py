from pathlib import Path

from backend.app.core.config import Settings


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def first_existing_project_path(path_value: str, *fallbacks: str) -> Path:
    """Resolve a configured path, falling back to known legacy/default locations."""
    candidates = [resolve_project_path(path_value), *(resolve_project_path(item) for item in fallbacks)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_runtime_dirs(settings: Settings) -> None:
    for value in (
        settings.storage_dir,
        settings.uploads_dir,
        settings.annotated_dir,
        settings.indexes_dir,
        settings.metadata_dir,
    ):
        resolve_project_path(value).mkdir(parents=True, exist_ok=True)
    resolve_project_path(settings.reference_ready_dir).mkdir(parents=True, exist_ok=True)


def dataset_paths(settings: Settings) -> dict[str, Path]:
    # Data/raw contains original datasets. Data/processed/reference_ready is an optional
    # curated image input layer. Storage paths are generated AI artifacts and are not dataset inputs.
    return {
        "raw_root": resolve_project_path(settings.data_raw_dir),
        "raw_company": resolve_project_path(settings.raw_company_dir),
        "raw_external": resolve_project_path(settings.raw_external_dir),
        "processed_root": resolve_project_path(settings.data_processed_dir),
        "reference_ready": resolve_project_path(settings.reference_ready_dir),
        "dme_20250129": first_existing_project_path(
            settings.dme_dataset_dir,
            "Data/Dataset/20250129_Fabric_defects/Dataset",
            "Data/raw/external/20250129_Fabric_defects/Dataset",
        ),
        "wfdd": first_existing_project_path(
            settings.wfdd_dataset_dir,
            "Data/Dataset/WFDD",
            "Data/raw/external/WFDD",
        ),
        "lusitano": resolve_project_path(settings.lusitano_dataset_dir),
        "mvtec_ad": resolve_project_path(settings.mvtec_dataset_dir),
    }
