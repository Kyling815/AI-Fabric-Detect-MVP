"""Phase 0 diagnostics: read existing artifacts; write smoke fixtures only in temp.

No training, calibration, reference activation, or production runtime changes.
Run from the repository root with its existing virtual environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}", flush=True)


def isolated_environment(root: Path) -> dict[str, str]:
    return {
        "FABRIC_DATABASE_URL": f"sqlite:///{(root / 'app.db').as_posix()}",
        "FABRIC_STORAGE_DIR": str(root),
        **{
            f"FABRIC_{name.upper()}_DIR": str(root / name)
            for name in ("uploads", "annotated", "indexes", "metadata", "references")
        },
    }


def smoke() -> None:
    import numpy as np
    import torch
    from PIL import Image

    from backend.app.ai.dense_anomaly import DenseReferenceDetector
    from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
    from backend.app.ai.faiss_index import FaissIndexStore
    from backend.app.core.config import get_settings
    from backend.app.core.paths import resolve_project_path
    from backend.app.services.fabric_registry import FabricRegistry, FabricRegistryError
    from backend.app.services.reference_registry import ReferenceRegistry

    settings = get_settings()
    existing_registry_payload = json.loads(resolve_project_path(settings.fabric_registry_path).read_text(encoding="utf-8"))
    registry = FabricRegistry()
    backbone = registry.shared_backbone
    fabrics = registry.list_fabrics()
    require(bool(fabrics), "existing fabric registry loads")
    for fabric in fabrics:
        print(f"REGISTRY: {fabric['fabric_id']} {fabric['status']} active={fabric.get('active_reference')}")
        if fabric["status"] == "INACTIVE":
            try:
                registry.get_active_reference(fabric["fabric_id"])
            except FabricRegistryError:
                pass
            else:
                raise AssertionError("Inactive fabric unexpectedly accepted")

    legacy = ReferenceRegistry()
    require(bool(legacy.list_entries()), "existing legacy reference registry loads")
    print(f"LEGACY: active_reference_version={legacy.active_reference_version}")

    # A missing cache must fail diagnostics rather than silently download weights.
    if backbone.get("use_pretrained_weights", True):
        from torchvision.models import ResNet18_Weights

        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet18_Weights.DEFAULT.url).name
        require(cached.is_file(), "registered ResNet18 pretrained weights are cached")
    with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Audit forbids weight downloads")):
        extractor = get_shared_dense_extractor(
            backbone["backbone_version"],
            "cpu",
            bool(backbone.get("use_pretrained_weights", True)),
            str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
        )
    print("MODEL: " + json.dumps(asdict(extractor.info), sort_keys=True))
    yy, xx = np.indices((96, 128))
    pixels = np.repeat((((xx // 8 + yy // 8) % 2) * 120 + 80).astype(np.uint8)[..., None], 3, axis=2)
    image = Image.fromarray(pixels)
    features = extractor.extract_image(image, image_max_size=int(backbone["image_max_size"]))
    repeated = extractor.extract_image(image, image_max_size=int(backbone["image_max_size"]))
    require(features.embeddings.shape == (features.grid_rows * features.grid_cols, extractor.info.feature_dim),
            "registered model forward shape matches dense grid")
    require(np.isfinite(features.embeddings).all(), "registered model produces finite embeddings")
    require(np.allclose(np.linalg.norm(features.embeddings, axis=1), 1, atol=1e-5), "embeddings are L2 normalized")
    require(np.allclose(features.embeddings, repeated.embeddings, atol=1e-6, rtol=0), "CPU repeat forward is deterministic")

    for fabric in fabrics:
        for record in fabric.get("references", []):
            entry = registry.get_reference(fabric["fabric_id"], record["reference_id"])
            bank = np.load(entry.memory_bank_path, mmap_mode="r", allow_pickle=False)
            detector = DenseReferenceDetector(entry.metadata_path)
            require((detector.index.ntotal, detector.index.d) == bank.shape,
                    f"{entry.reference_id}: existing FAISS and memory dimensions match")
            sample = np.asarray(bank[:16]).copy()
            vectors = np.vstack([detector.index.reconstruct(i) for i in range(len(sample))])
            require(np.allclose(sample, vectors, atol=1e-7, rtol=0),
                    f"{entry.reference_id}: FAISS stored vectors match memory")
            scores = detector.store.search(detector.index, sample, k=1).scores[:, 0]
            require(np.allclose(scores, 1, atol=1e-5), f"{entry.reference_id}: normalized self-query succeeds")
            if entry.backbone_version == backbone["backbone_version"]:
                result = detector.score(
                    features.embeddings, features.boxes,
                    grid_rows=features.grid_rows, grid_cols=features.grid_cols,
                    image_width=features.processed_width, image_height=features.processed_height,
                    thresholds=entry.thresholds,
                )
                require(np.isfinite(result.patch_scores).all(), f"{entry.reference_id}: registered encoder scores existing index")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    with TemporaryDirectory(prefix="fabric-phase0-smoke-") as directory:
        root = Path(directory)
        env = isolated_environment(root)
        env["FABRIC_FABRIC_REGISTRY_PATH"] = str(root / "metadata" / "fixture_registry.json")
        with patch.dict(os.environ, env):
            get_settings.cache_clear()
            # These imports must occur after DB/storage overrides, before app initialization.
            from fastapi.testclient import TestClient
            from backend.app.db.database import engine
            from backend.main import app

            fixture_dir = root / "references" / "phase0_fixture"
            fixture_dir.mkdir(parents=True)
            bank_path = fixture_dir / "memory_bank.npy"
            index_path = fixture_dir / "memory_bank.faiss"
            metadata_path = fixture_dir / "reference.json"
            np.save(bank_path, features.embeddings)
            store = FaissIndexStore(metric="cosine")
            store.save(features.embeddings, index_path)
            metadata = {"faiss_index_path": index_path.name, "faiss_metric": "cosine"}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            reference = {
                "reference_id": "phase0_fixture_ref", "reference_version": 0, "status": "ACTIVE",
                "backbone_version": backbone["backbone_version"], "artifact_dir": str(fixture_dir),
                "metadata_path": str(metadata_path), "memory_bank_path": str(bank_path),
                "faiss_index_path": str(index_path), "thresholds": {},
            }
            payload = {
                "schema_version": 1, "shared_backbone": backbone,
                "fabrics": {"phase0_fixture": {
                    "display_name": "Phase 0 synthetic fixture", "status": "ACTIVE",
                    "active_reference": reference["reference_id"], "references": [reference],
                }},
            }
            registry_path = Path(env["FABRIC_FABRIC_REGISTRY_PATH"])

            def fixture_thresholds(review: float, bad: float, pixel: float) -> None:
                reference["thresholds"] = {
                    "image_score_method": "max", "image_review_threshold": review,
                    "image_bad_threshold": bad, "pixel_threshold": pixel, "minimum_defect_area": 1,
                }
                registry_path.write_text(json.dumps(payload), encoding="utf-8")

            fixture_thresholds(2, 2.1, 2)
            try:
                with TestClient(app) as client:
                    registry_path.write_text(json.dumps(existing_registry_payload), encoding="utf-8")
                    actual_fabrics = client.get("/api/fabrics")
                    require(actual_fabrics.status_code == 200 and len(actual_fabrics.json()["fabrics"]) == len(fabrics),
                            "API loads unchanged existing fabric registry")
                    for fabric in fabrics:
                        if fabric["status"] == "INACTIVE":
                            response = client.post("/api/inspect", data={"fabric_id": fabric["fabric_id"]},
                                                   files={"image": ("fixture.png", png, "image/png")})
                            require(response.status_code == 409, f"existing inactive {fabric['fabric_id']} API -> 409")
                    fixture_thresholds(2, 2.1, 2)
                    for route in ("/api/health", "/api/model-info", "/api/fabrics", "/api/inspections", "/api/reference/list", "/openapi.json"):
                        require(client.get(route).status_code == 200, f"API GET {route}")

                    def upload(data: dict | None, content: bytes = png, content_type: str = "image/png"):
                        return client.post("/api/inspect", data=data, files={"image": ("fixture.png", content, content_type)})

                    require(upload(None).status_code == 422, "inspect missing fabric_id -> 422")
                    require(upload({"fabric_id": "unknown"}).status_code == 400, "inspect unknown fabric -> 400")
                    require(upload({"fabric_id": "phase0_fixture"}, content_type="text/plain").status_code == 400,
                            "inspect non-image MIME -> 400")
                    for label, thresholds in (("GOOD", (2, 2.1, 2)), ("NEEDS_REVIEW", (0, 1, 2)), ("BAD", (0, 0, 0))):
                        fixture_thresholds(*thresholds)
                        response = upload({"fabric_id": "phase0_fixture"})
                        require(response.status_code == 201, f"synthetic inspect {label}: HTTP 201")
                        result = response.json()
                        require(result["final_result"] == label and result["confidence"] is None, f"synthetic decision {label}")
                        require(result["fabric_id"] == "phase0_fixture" and result["reference_id"] == reference["reference_id"],
                                f"{label}: identity preserved in response")
                        saved = client.get(f"/api/inspections/{result['inspection_id']}")
                        require(saved.status_code == 200 and saved.json()["final_result"] == label, f"{label}: SQLite persistence")
                        for field in ("original_image_url", "annotated_image_url"):
                            require(client.get(result[field]).status_code == 200, f"{label}: {field} served")

                    fixture_thresholds(2, 2.1, 2)
                    before = set((root / "uploads").iterdir())
                    require(upload({"fabric_id": "phase0_fixture"}, b"corrupted image").status_code == 400,
                            "inspect corrupt image -> 400")
                    require(set((root / "uploads").iterdir()) == before, "failed decode cleans temporary upload")
                    metadata_path.write_text("{invalid json", encoding="utf-8")
                    require(upload({"fabric_id": "phase0_fixture"}).status_code == 409, "inspect corrupt metadata -> 409")
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    store.save(features.embeddings[:, :-1], index_path)
                    require(upload({"fabric_id": "phase0_fixture"}).status_code == 409, "inspect wrong embedding dimension -> 409")
                    reference["faiss_index_path"] = str(fixture_dir / "missing.faiss")
                    registry_path.write_text(json.dumps(payload), encoding="utf-8")
                    require(upload({"fabric_id": "phase0_fixture"}).status_code == 409, "inspect missing artifact -> 409")
                    payload["fabrics"]["phase0_fixture"]["status"] = "INACTIVE"
                    registry_path.write_text(json.dumps(payload), encoding="utf-8")
                    require(upload({"fabric_id": "phase0_fixture"}).status_code == 409, "inspect inactive fabric -> 409")
            finally:
                engine.dispose()
                get_settings.cache_clear()
    print("SMOKE COMPLETE: synthetic fixtures verify software only; no real-data accuracy claim.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", action="store_true", help="Run existing pytest suite with temporary DB/storage first")
    parser.add_argument("--frontend", action="store_true", help="Run existing frontend build (writes ignored dist)")
    args = parser.parse_args()
    if args.regression:
        with TemporaryDirectory(prefix="fabric-phase0-regression-") as directory:
            env = {**os.environ, **isolated_environment(Path(directory))}
            subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, env=env, check=True)
    if args.frontend:
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm is required for frontend build")
        subprocess.run([npm, "run", "build"], cwd=PROJECT_ROOT / "frontend", check=True)
    smoke()


if __name__ == "__main__":
    main()
