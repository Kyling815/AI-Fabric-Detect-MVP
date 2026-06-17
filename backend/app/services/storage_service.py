from pathlib import Path
from dataclasses import dataclass
from shutil import copyfileobj
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


@dataclass(frozen=True)
class StoredUpload:
    relative_path: str
    absolute_path: Path
    file_size_bytes: int


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def save_upload_placeholder(self, filename: str) -> str:
        upload_dir = resolve_project_path(self.settings.uploads_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        return str((upload_dir / safe_name).relative_to(resolve_project_path(self.settings.storage_dir).parent))

    def save_upload(self, upload: UploadFile) -> StoredUpload:
        upload_dir = resolve_project_path(self.settings.uploads_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_name = Path(upload.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        stem = Path(original_name).stem or "upload"
        filename = f"{stem}_{uuid4().hex[:12]}{suffix}"
        destination = upload_dir / filename

        upload.file.seek(0)
        with destination.open("wb") as target:
            copyfileobj(upload.file, target)

        return StoredUpload(
            relative_path=str(destination.relative_to(resolve_project_path(self.settings.storage_dir).parent)),
            absolute_path=destination,
            file_size_bytes=destination.stat().st_size,
        )

    def annotated_output_for(self, inspection_id: str, source_filename: str) -> StoredUpload:
        annotated_dir = resolve_project_path(self.settings.annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(source_filename).suffix.lower() or ".jpg"
        destination = annotated_dir / f"{inspection_id}_annotated{suffix}"
        return StoredUpload(
            relative_path=str(destination.relative_to(resolve_project_path(self.settings.storage_dir).parent)),
            absolute_path=destination,
            file_size_bytes=0,
        )

    def to_public_url(self, path: str | None) -> str | None:
        if not path:
            return None
        normalized = path.replace("\\", "/")
        if normalized.startswith("storage/"):
            return f"/{normalized}"
        return f"/storage/{normalized}"

    def delete_stored_file(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        candidate = resolve_project_path(relative_path)
        storage_root = resolve_project_path(self.settings.storage_dir).resolve()
        resolved = candidate.resolve()
        if storage_root in resolved.parents and resolved.exists():
            resolved.unlink()
