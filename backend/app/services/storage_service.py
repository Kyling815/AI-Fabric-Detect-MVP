from pathlib import Path
from dataclasses import dataclass
import re
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


@dataclass(frozen=True)
class StoredUpload:
    relative_path: str
    absolute_path: Path
    file_size_bytes: int


class UploadTooLargeError(ValueError):
    pass


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def save_upload_placeholder(self, filename: str) -> str:
        upload_dir = resolve_project_path(self.settings.uploads_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        return self._relative_to_storage(upload_dir / safe_name)

    def save_upload(self, upload: UploadFile) -> StoredUpload:
        upload_dir = resolve_project_path(self.settings.uploads_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_name = Path(upload.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(original_name).stem).strip("._")[:80] or "upload"
        if len(suffix) > 12 or not re.fullmatch(r"\.[a-zA-Z0-9]+", suffix):
            suffix = ".img"
        filename = f"{stem}_{uuid4().hex[:12]}{suffix}"
        destination = upload_dir / filename

        upload.file.seek(0)
        written = 0
        try:
            with destination.open("wb") as target:
                while chunk := upload.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > self.settings.max_upload_size_bytes:
                        raise UploadTooLargeError(
                            f"Upload exceeds the {self.settings.max_upload_size_bytes}-byte limit."
                        )
                    target.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        return StoredUpload(
            relative_path=self._relative_to_storage(destination),
            absolute_path=destination,
            file_size_bytes=written,
        )

    def annotated_output_for(self, inspection_id: str, source_filename: str) -> StoredUpload:
        annotated_dir = resolve_project_path(self.settings.annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(source_filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            suffix = ".jpg"
        destination = annotated_dir / f"{inspection_id}_annotated{suffix}"
        return StoredUpload(
            relative_path=self._relative_to_storage(destination),
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
        storage_root = resolve_project_path(self.settings.storage_dir).resolve()
        normalized = relative_path.replace("\\", "/")
        if normalized.startswith("storage/"):
            normalized = normalized.removeprefix("storage/")
        candidate = storage_root / normalized
        resolved = candidate.resolve()
        if storage_root in resolved.parents and resolved.exists():
            resolved.unlink()

    def _relative_to_storage(self, path: Path) -> str:
        storage_root = resolve_project_path(self.settings.storage_dir).resolve()
        return path.resolve().relative_to(storage_root).as_posix()
