"""Temporary Phase 1 fixtures: fake encoder, real FAISS/SQLite/HTTP."""

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock

from backend.app.ai import current_anomaly
from backend.app.ai.dense_feature_extractor import DenseFeatureExtractorInfo, DenseFeatureResult
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.patching import PatchBox
from backend.app.core.config import get_settings
from backend.app.db.database import Base, get_db
from backend.main import create_app


def baseline() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "phase_1_dense_baseline.json").read_text(encoding="utf-8"))


def dense_features() -> DenseFeatureResult:
    return DenseFeatureResult(
        embeddings=np.array([[1, 0]] * 5 + [[0, 1]], dtype=np.float32),
        boxes=tuple(PatchBox(c * 8, r * 8, 8, 8) for r in range(2) for c in range(3)),
        grid_rows=2, grid_cols=3, processed_width=24, processed_height=16,
        original_width=48, original_height=32, scale=0.5,
    )


def fake_extractor() -> Mock:
    extractor = Mock()
    extractor.info = DenseFeatureExtractorInfo(
        "shared_resnet18_imagenet1k_v1", "resnet18_layer2_layer3_dense", "cpu", False, 2, 8,
    )
    extractor.extract_path.side_effect = lambda *args, **kwargs: dense_features()
    return extractor


def normalize_response(value: dict) -> dict:
    value = dict(value)
    for key in baseline()["provenance"]["normalized_fields"]:
        if value[key] is not None:
            value[key] = f"<{key}>"
    return value


@dataclass
class Phase1Runtime:
    root: Path
    registry_path: Path
    registry: dict
    reference: dict
    client: TestClient
    sessions: sessionmaker
    extractor: Mock
    factory: Mock
    png: bytes

    def save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self.registry), encoding="utf-8")

    def use_case(self, name: str) -> None:
        self.reference["thresholds"] = dict(baseline()["cases"][name]["thresholds"])
        self.save_registry()

    def upload(self, fabric_id: str = "fixture_fabric", *, content: bytes | None = None):
        return self.client.post(
            "/api/inspect", data={"fabric_id": fabric_id},
            files={"image": ("fixture.png", self.png if content is None else content, "image/png")},
        )


@pytest.fixture
def phase1_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = get_settings()
    root = tmp_path / "storage"
    for name in ("uploads", "annotated", "indexes", "metadata"):
        (root / name).mkdir(parents=True)
        monkeypatch.setattr(settings, f"{name}_dir", str(root / name))
    monkeypatch.setattr(settings, "storage_dir", str(root))
    registry_path = root / "metadata" / "fabric_registry.json"
    monkeypatch.setattr(settings, "fabric_registry_path", str(registry_path))
    for key, value in {"device": "auto", "image_max_size": 768, "model_input_size": 224,
                       "patch_size": 96, "patch_stride": 48, "batch_size": 16,
                       "max_upload_size_bytes": 25 * 1024 * 1024}.items():
        monkeypatch.setattr(settings, key, value)
    artifact = root / "references" / "fixture_fabric" / "v1"
    artifact.mkdir(parents=True)
    bank = np.array([[1, 0]], dtype=np.float32)
    np.save(artifact / "memory_bank.npy", bank)
    FaissIndexStore().save(bank, artifact / "memory_bank.faiss")
    (artifact / "reference.json").write_text(json.dumps({"faiss_index_path": "memory_bank.faiss"}), encoding="utf-8")
    reference = {
        "reference_id": "fixture_ref_v1", "reference_version": 1, "status": "ACTIVE",
        "backbone_version": "shared_resnet18_imagenet1k_v1", "artifact_dir": str(artifact),
        "metadata_path": str(artifact / "reference.json"), "memory_bank_path": str(artifact / "memory_bank.npy"),
        "faiss_index_path": str(artifact / "memory_bank.faiss"), "thresholds": {},
    }
    registry = {
        "schema_version": 1,
        "shared_backbone": {"backbone_version": "shared_resnet18_imagenet1k_v1", "use_pretrained_weights": False, "image_max_size": 24},
        "fabrics": {"fixture_fabric": {"display_name": "Fixture Fabric", "status": "ACTIVE",
                                      "active_reference": reference["reference_id"], "references": [reference]}},
    }
    extractor = fake_extractor()
    factory = Mock(return_value=extractor)
    monkeypatch.setattr(current_anomaly, "get_shared_dense_extractor", factory)
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)

    def get_test_db():
        with sessions() as db:
            yield db

    buffer = BytesIO()
    Image.new("RGB", (48, 32), (200, 200, 200)).save(buffer, format="PNG")
    # Static mounts bind their directories when the app is constructed. Build
    # the fixture app after settings overrides so image routes use this sandbox.
    app = create_app()
    app.dependency_overrides[get_db] = get_test_db
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            runtime = Phase1Runtime(root, registry_path, registry, reference, client, sessions, extractor, factory, buffer.getvalue())
            runtime.use_case("good")
            yield runtime
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
