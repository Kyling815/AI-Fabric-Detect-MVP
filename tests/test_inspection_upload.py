from collections.abc import Generator
from io import BytesIO
import json
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
    monkeypatch.setattr(settings, "fabric_registry_path", str(storage / "metadata" / "fabric_registry.json"))
    monkeypatch.setattr(settings, "default_reference_id", "")
    monkeypatch.setattr(settings, "use_pretrained_weights", False)
    monkeypatch.setattr(settings, "use_projection_head", False)
    monkeypatch.setattr(settings, "projection_head_path", "")
    monkeypatch.setattr(settings, "max_upload_size_bytes", 25 * 1024 * 1024)
    for directory in (storage, storage / "uploads", storage / "annotated", storage / "indexes", storage / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)
    (storage / "metadata" / "fabric_registry.json").write_text(
        json.dumps(
            {
                "shared_backbone": {"backbone_version": "shared_resnet18_imagenet1k_v1"},
                "fabrics": {
                    "fabric_001": {
                        "display_name": "Fabric A",
                        "status": "INACTIVE",
                        "active_reference": None,
                        "references": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

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


def test_inspect_upload_requires_fabric_id(isolated_client: TestClient) -> None:
    response = isolated_client.post(
        "/api/inspect",
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 422


def test_inspect_upload_rejects_unknown_fabric_id(isolated_client: TestClient) -> None:
    response = isolated_client.post(
        "/api/inspect",
        data={"fabric_id": "missing_fabric"},
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400


def test_inspect_upload_enforces_size_limit(isolated_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    registry_path = Path(settings.fabric_registry_path)
    artifact_dir = registry_path.parent / "fabric_001" / "v1"
    artifact_dir.mkdir(parents=True)
    metadata_path = artifact_dir / "reference.json"
    memory_bank_path = artifact_dir / "memory_bank.npy"
    index_path = artifact_dir / "memory_bank.faiss"
    for path in (metadata_path, memory_bank_path, index_path):
        path.touch()
    registry_path.write_text(
        json.dumps(
            {
                "shared_backbone": {"backbone_version": "shared_resnet18_imagenet1k_v1"},
                "fabrics": {
                    "fabric_001": {
                        "display_name": "Fabric A",
                        "status": "ACTIVE",
                        "active_reference": "fabric_001_ref_v1",
                        "references": [
                            {
                                "reference_id": "fabric_001_ref_v1",
                                "reference_version": 1,
                                "status": "ACTIVE",
                                "backbone_version": "shared_resnet18_imagenet1k_v1",
                                "artifact_dir": str(artifact_dir),
                                "metadata_path": str(metadata_path),
                                "memory_bank_path": str(memory_bank_path),
                                "faiss_index_path": str(index_path),
                                "thresholds": {
                                    "image_review_threshold": 0.1,
                                    "image_bad_threshold": 0.2,
                                    "pixel_threshold": 0.1,
                                    "minimum_defect_area": 1,
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(get_settings(), "max_upload_size_bytes", 10)
    response = isolated_client.post(
        "/api/inspect",
        data={"fabric_id": "fabric_001"},
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 413


def test_fabric_list_exposes_stable_identity(isolated_client: TestClient) -> None:
    response = isolated_client.get("/api/fabrics")

    assert response.status_code == 200
    assert response.json()["fabrics"] == [
        {
            "fabric_id": "fabric_001",
            "display_name": "Fabric A",
            "status": "INACTIVE",
            "active_reference": None,
            "inspectable": False,
        }
    ]
