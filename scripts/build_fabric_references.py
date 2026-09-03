import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build isolated dense memory banks for registered fabrics.")
    parser.add_argument("--config", default=settings.fabric_demo_config_path)
    parser.add_argument("--manifest-dir", default="storage/metadata/company_demo_manifests")
    parser.add_argument("--output-root", default=settings.references_dir)
    parser.add_argument("--fabric-id", action="append", default=None)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--include-development-synthetic-good",
        action="store_true",
        help=(
            "Add only synthetic Good descendants of reference sources. "
            "Validation and locked-test sources remain excluded."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backbone = config["backbone"]
    selected = set(args.fabric_id or config["fabrics"].keys())
    unknown = selected - set(config["fabrics"])
    if unknown:
        raise SystemExit(f"Unknown fabric ids: {sorted(unknown)}")

    start = time.perf_counter()
    extractor = get_shared_dense_extractor(
        backbone["backbone_version"],
        settings.device,
        bool(backbone["use_pretrained_weights"]),
        str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "",
    )
    backbone_load_seconds = time.perf_counter() - start
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_shared_backbone(output_root, extractor.info, backbone_load_seconds)

    results = []
    for fabric_id in sorted(selected):
        result = build_one(
            fabric_id=fabric_id,
            fabric=config["fabrics"][fabric_id],
            dataset_version=config["dataset_version"],
            backbone=backbone,
            extractor=extractor,
            manifest_dir=resolve_project_path(args.manifest_dir),
            output_root=output_root,
            version=args.version,
            force=args.force,
            include_development_synthetic_good=args.include_development_synthetic_good,
        )
        results.append(result)
    update_registry(config, output_root, results)
    print(json.dumps({"backbone_load_seconds": backbone_load_seconds, "references": results}, indent=2))


def build_one(**kwargs) -> dict:
    fabric_id = kwargs["fabric_id"]
    version = kwargs["version"]
    artifact_dir = kwargs["output_root"] / fabric_id / f"v{version}"
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not kwargs["force"]:
        raise FileExistsError(f"Immutable reference already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = kwargs["manifest_dir"] / f"{fabric_id}_all.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if row["label"] == "normal"
        and (
            (row["split"] == "reference" and row["origin"] == "real")
            or (
                kwargs["include_development_synthetic_good"]
                and row["split"] == "development_only"
                and row["origin"] == "synthetic"
            )
        )
    ]
    if not rows:
        raise RuntimeError(f"No approved Real Good rows for {fabric_id}")

    all_embeddings = []
    image_records = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        print(f"[{fabric_id}] extract {index}/{len(rows)} {Path(row['image_path']).name}", flush=True)
        dense = kwargs["extractor"].extract_path(
            Path(row["image_path"]),
            image_max_size=int(kwargs["backbone"]["image_max_size"]),
        )
        all_embeddings.append(dense.embeddings)
        image_records.append(
            {
                "image_id": row["image_id"],
                "path": row["image_path"],
                "source_original": row["source_original"],
                "dense_feature_count": int(dense.embeddings.shape[0]),
            }
        )
    full_bank = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    bank = deterministic_coreset(
        full_bank,
        maximum=int(kwargs["backbone"]["coreset_size"]),
        seed=int(kwargs["backbone"]["random_seed"]),
    )
    memory_bank_path = artifact_dir / "memory_bank.npy"
    faiss_path = artifact_dir / "memory_bank.faiss"
    np.save(memory_bank_path, bank)
    faiss_info = FaissIndexStore(metric="cosine").save(bank, faiss_path)
    build_seconds = time.perf_counter() - started
    reference_id = f"{fabric_id}_ref_v{version}"
    metadata = {
        "schema_version": 1,
        "fabric_id": fabric_id,
        "display_name": kwargs["fabric"]["display_name"],
        "reference_id": reference_id,
        "reference_version": version,
        "status": "CANDIDATE",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "backbone_version": kwargs["backbone"]["backbone_version"],
        "architecture": kwargs["backbone"]["architecture"],
        "base_backbone_version": kwargs["backbone"].get("base_backbone_version"),
        "adapter_version": kwargs["backbone"].get("adapter_version"),
        "adapter_path": kwargs["backbone"].get("adapter_path"),
        "dataset_version": kwargs["dataset_version"],
        "preprocessing": {"image_max_size": int(kwargs["backbone"]["image_max_size"])},
        "feature_dim": int(bank.shape[1]),
        "full_feature_count": int(full_bank.shape[0]),
        "memory_bank_count": int(bank.shape[0]),
        "good_source_count": len(rows),
        "reference_policy": (
            "real_good_plus_reference_source_synthetic_good"
            if kwargs["include_development_synthetic_good"]
            else "real_good_only"
        ),
        "real_good_count": sum(row["origin"] == "real" for row in rows),
        "synthetic_good_count": sum(row["origin"] == "synthetic" for row in rows),
        "memory_bank_path": memory_bank_path.name,
        "faiss_index_path": faiss_path.name,
        "faiss_metric": faiss_info.metric,
        "thresholds": {},
        "images": image_records,
        "performance": {
            "build_seconds": build_seconds,
            "memory_bank_mb": memory_bank_path.stat().st_size / (1024 * 1024),
        },
    }
    atomic_json(artifact_dir / "reference.json", metadata)
    atomic_json(artifact_dir / "manifest.json", {"rows": rows})
    atomic_json(artifact_dir / "thresholds.json", {"status": "NOT_CALIBRATED", "thresholds": {}})
    atomic_json(artifact_dir / "metrics.json", {"status": "NOT_EVALUATED"})
    return {
        "fabric_id": fabric_id,
        "display_name": kwargs["fabric"]["display_name"],
        "reference_id": reference_id,
        "reference_version": version,
        "status": "CANDIDATE",
        "backbone_version": kwargs["backbone"]["backbone_version"],
        "artifact_dir": str(artifact_dir),
        "metadata_path": str(artifact_dir / "reference.json"),
        "memory_bank_path": str(memory_bank_path),
        "faiss_index_path": str(faiss_path),
        "thresholds": {},
        "good_source_count": len(rows),
        "memory_bank_count": int(bank.shape[0]),
        "build_seconds": build_seconds,
    }


def deterministic_coreset(matrix: np.ndarray, *, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("coreset size must be positive")
    if matrix.shape[0] <= maximum:
        return np.ascontiguousarray(matrix, dtype=np.float32)
    generator = np.random.default_rng(seed)
    indexes = np.sort(generator.choice(matrix.shape[0], size=maximum, replace=False))
    return np.ascontiguousarray(matrix[indexes], dtype=np.float32)


def update_registry(config: dict, output_root: Path, results: list[dict]) -> None:
    registry_path = resolve_project_path(get_settings().fabric_registry_path)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema_version": 1,
            "shared_backbone": config["backbone"],
            "fabrics": {},
        }
    registry["shared_backbone"] = config["backbone"]
    for fabric_id, config_record in config["fabrics"].items():
        fabric = registry["fabrics"].setdefault(
            fabric_id,
            {
                "display_name": config_record["display_name"],
                "status": "INACTIVE",
                "active_reference": None,
                "references": [],
            },
        )
        fabric["display_name"] = config_record["display_name"]
    for result in results:
        fabric = registry["fabrics"][result["fabric_id"]]
        references = [
            item for item in fabric.get("references", []) if item.get("reference_id") != result["reference_id"]
        ]
        references.append({key: value for key, value in result.items() if key not in {"display_name", "build_seconds", "good_source_count", "memory_bank_count"}})
        fabric["references"] = sorted(references, key=lambda item: item["reference_version"])
    registry["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    atomic_json(registry_path, registry)


def _write_shared_backbone(output_root: Path, info, load_seconds: float) -> None:
    directory = output_root / "shared" / info.backbone_version
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        directory / "backbone.json",
        {
            "backbone_version": info.backbone_version,
            "architecture": info.architecture,
            "device": info.device,
            "use_pretrained_weights": info.use_pretrained_weights,
            "feature_dim": info.feature_dim,
            "feature_stride": info.feature_stride,
            "load_seconds": load_seconds,
        },
    )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
