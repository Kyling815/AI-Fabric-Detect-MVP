from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import get_settings
from backend.main import app


def _png_bytes(width: int = 64, height: int = 32) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_inspect_upload_returns_image_metadata() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/inspect",
        data={"reference_id": "dme_plain_v1"},
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["final_result"] in {"Good", "Acceptable", "Bad", "ReferenceMissing"}
    assert body["reference_id"] == "dme_plain_v1"
    assert body["image_metadata"]["width"] == 64
    assert body["image_metadata"]["height"] == 32
    assert body["preprocess_plan"]["patch_size"] == 96
    assert body["patch_grid"]["patch_count"] == 1
    assert body["patch_grid"]["sample_patches"][0] == {"x": 0, "y": 0, "w": 64, "h": 32}
    assert body["embedding_summary"]["model_name"] == "resnet18"
    settings = get_settings()
    expected_embedding_dim = settings.projection_dim if settings.use_projection_head else 512
    assert body["embedding_summary"]["embedding_dim"] == expected_embedding_dim
    assert body["embedding_summary"]["patch_count"] == 1
    assert body["embedding_summary"]["device"] in {"cpu", "cuda"}
    assert body["anomaly_summary"]["status"] in {"scored", "reference_missing"}
    assert body["anomaly_summary"]["patch_count"] == 1
    if body["anomaly_summary"]["status"] == "scored":
        assert body["annotated_image_url"].startswith("/storage/annotated/")
        assert body["model_version"] == body["anomaly_summary"]["reference_version"]
    assert body["original_image_url"].startswith("/storage/uploads/")


def test_inspect_upload_rejects_unknown_reference_id() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/inspect",
        data={"reference_id": "missing_reference"},
        files={"image": ("fabric.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
