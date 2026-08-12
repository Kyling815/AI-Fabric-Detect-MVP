from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.db.database import Base, get_db
from backend.main import app


def _png_bytes(width: int = 64, height: int = 32) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    settings = get_settings()
    storage = tmp_path / "storage"
    for name in ("storage_dir", "uploads_dir", "annotated_dir", "indexes_dir", "metadata_dir"):
        value = storage if name == "storage_dir" else storage / name.removesuffix("_dir")
        monkeypatch.setattr(settings, name, str(value))
    monkeypatch.setattr(settings, "reference_registry_path", str(storage / "metadata" / "reference_registry.json"))
    monkeypatch.setattr(settings, "default_reference_id", "")
    monkeypatch.setattr(settings, "use_pretrained_weights", False)
    monkeypatch.setattr(settings, "use_projection_head", False)
    monkeypatch.setattr(settings, "projection_head_path", "")
    monkeypatch.setattr(settings, "max_upload_size_bytes", 25 * 1024 * 1024)
    for directory in (storage, storage / "uploads", storage / "annotated", storage / "indexes", storage / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db() -> Generator[Session, None, None]:
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        app.dependency_overrides.clear()
        engine.dispose()


def test_inspect_upload_returns_image_metadata(isolated_client: TestClient) -> None:
    response = isolated_client.post(
        "/api/inspect",
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["final_result"] == "ReferenceMissing"
    assert body["image_metadata"]["width"] == 64
    assert body["image_metadata"]["height"] == 32
    assert body["preprocess_plan"]["patch_size"] == 96
    assert body["patch_grid"]["patch_count"] == 1
    assert body["patch_grid"]["sample_patches"][0] == {"x": 0, "y": 0, "w": 64, "h": 32}
    assert body["embedding_summary"]["model_name"] == "resnet18"
    assert body["embedding_summary"]["embedding_dim"] == 512
    assert body["embedding_summary"]["patch_count"] == 1
    assert body["embedding_summary"]["device"] in {"cpu", "cuda"}
    assert body["anomaly_summary"]["status"] == "reference_missing"
    assert body["original_image_url"].startswith("/storage/uploads/")


def test_inspect_upload_rejects_unknown_reference_id(isolated_client: TestClient) -> None:
    response = isolated_client.post(
        "/api/inspect",
        data={"reference_id": "missing_reference"},
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400


def test_inspect_upload_enforces_size_limit(isolated_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_size_bytes", 10)
    response = isolated_client.post(
        "/api/inspect",
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 413
