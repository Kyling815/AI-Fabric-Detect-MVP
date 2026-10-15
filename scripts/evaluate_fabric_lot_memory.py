"""Read-only research comparison: fabric-only vs fabric+lot, default explicit opt-in.

--demo uses synthetic vectors/images and temporary artifacts, never real accuracy.
Prepared query embeddings must match encoder/adapter versions and a separate lot
evaluation manifest. No production thresholds, labels, registry writes or training.
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, file_sha256
from backend.app.ai.lot_evaluation import evaluate_fabric_lot
from backend.app.core.lot_evaluation import LotEvaluationManifest, LotEvaluationSample
from backend.app.core.lot_memory import FabricLotMemoryManifest, LotMemoryConfig, MemoryArtifactSpec
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage


def create_synthetic_evaluation_fixture(root: Path) -> tuple[FabricLotMemoryManifest, LotEvaluationManifest]:
    """Toy drift scenario for software checks; does not infer Good from model output."""
    config = MemoryExperimentConfig(enabled=True)
    specs = {}
    for scope, lot_id, vector, intensity in (("global", None, [1, 0], 80), ("lot", "demo_lot", [0.8, 0.6], 140)):
        source_path = root / f"{scope}_reference_source.png"
        Image.new("RGB", (16, 16), (intensity, 100, 100)).save(source_path)
        source = SourceGoodImage(image_id=f"demo_{scope}_reference", path=str(source_path),
                                 sha256=file_sha256(source_path), fabric_id="fabric_001", lot_id=lot_id)
        identity = MemoryIdentity(reference_version=f"demo_{scope}_v1", encoder_version="synthetic_unit_encoder_v1",
                                  adapter_version=None, fabric_id="fabric_001", lot_id=lot_id,
                                  threshold_profile_version=None, source_good_images=(source,))
        memory = ExperimentalAnomalyMemory.build(np.array([vector], dtype=np.float32), identity=identity, config=config)
        path = root / scope
        memory.save_new(path)
        specs[scope] = MemoryArtifactSpec(path=str(path), identity=identity, config=config)
    hierarchy = FabricLotMemoryManifest(fabric_id="fabric_001", lot_id="demo_lot",
                                       config=LotMemoryConfig(enabled=True, mode="global_plus_lot"),
                                       global_memory=specs["global"], lot_memory=specs["lot"])
    samples = []
    for index, (label, vector) in enumerate((("normal", [0.8, 0.6]), ("normal", [0.9, 0.435889894]),
                                            ("defect", [0, 1]), ("defect", [-1, 0]))):
        image_path = root / f"query_{index}.png"
        Image.new("RGB", (16, 16), (20 + index * 20, 30, 50)).save(image_path)
        embeddings_path = root / f"query_{index}.npy"
        np.save(embeddings_path, np.array([vector], dtype=np.float32), allow_pickle=False)
        samples.append(LotEvaluationSample(image_id=f"demo_query_{index}", fabric_id="fabric_001", lot_id="demo_lot",
                                           image_path=str(image_path), image_sha256=file_sha256(image_path),
                                           embeddings_path=str(embeddings_path), embeddings_sha256=file_sha256(embeddings_path),
                                           label=label, origin="synthetic"))
    dataset = LotEvaluationManifest(dataset_id="phase5_synthetic_drift_demo", fabric_id="fabric_001", lot_id="demo_lot",
                                    encoder_version="synthetic_unit_encoder_v1", adapter_version=None,
                                    independence_basis="Distinct toy query image files; vectors constructed for software checks only.",
                                    samples=tuple(samples))
    return hierarchy, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Explicit synthetic-only demo in temp, no real metric improvement claim")
    parser.add_argument("--hierarchy", type=Path, help="FabricLotMemoryManifest JSON with expected child identities/configs")
    parser.add_argument("--dataset", type=Path, help="LotEvaluationManifest JSON with independent queries/labels/hashes")
    parser.add_argument("--enable-lot-memory", action="store_true", help="Explicit research comparison opt-in, even if manifest enabled=true")
    args = parser.parse_args()
    if args.demo:
        if args.hierarchy or args.dataset:
            parser.error("--demo must not be mixed with real artifact/dataset arguments")
        with TemporaryDirectory(prefix="fabric-phase5-demo-") as directory:
            hierarchy, dataset = create_synthetic_evaluation_fixture(Path(directory))
            result = evaluate_fabric_lot(hierarchy, dataset)
        print(json.dumps(result, indent=2))
        return
    if not args.hierarchy or not args.dataset:
        parser.error("Use --demo, or provide both --hierarchy and --dataset")
    hierarchy = FabricLotMemoryManifest.model_validate_json(args.hierarchy.read_text(encoding="utf-8"))
    dataset = LotEvaluationManifest.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    if not args.enable_lot_memory:
        print(json.dumps({"status": "PLAN_ONLY", "fabric_id": hierarchy.fabric_id, "lot_id": hierarchy.lot_id,
                          "dataset_id": dataset.dataset_id, "lot_memory_used": False, "writes_performed": False}, indent=2))
        return
    enabled = hierarchy.config.model_copy(update={"enabled": True, "mode": "global_plus_lot"})
    hierarchy = hierarchy.model_copy(update={"config": enabled})
    result = evaluate_fabric_lot(hierarchy, dataset, artifact_base_dir=args.hierarchy.resolve().parent,
                                 dataset_base_dir=args.dataset.resolve().parent)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
