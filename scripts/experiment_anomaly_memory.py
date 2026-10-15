"""Explicit offline memory experiment; never touches registries or production thresholds."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, file_sha256, validate_research_output
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="Approved Good embeddings .npy; not an image dataset")
    parser.add_argument("--provenance", type=Path, required=True, help="MemoryIdentity JSON, including source hashes/splits")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "memory_experiment_example.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; never overwrite")
    parser.add_argument("--enable-experiment", action="store_true", help="Explicit CLI write gate, required even if JSON enabled=true")
    parser.add_argument("--query-embeddings", type=Path)
    args = parser.parse_args()
    config = MemoryExperimentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    identity = MemoryIdentity.model_validate_json(args.provenance.read_text(encoding="utf-8"))
    output = validate_research_output(args.output_dir)
    if not args.enable_experiment:
        print(json.dumps({"status": "PLAN_ONLY", "output_dir": str(output), "config": config.model_dump(),
                          "identity": identity.model_dump(mode="json"), "writes_performed": False}, indent=2))
        return
    # The explicit flag is the opt-in. No source images are relabeled or generated.
    config = config.model_copy(update={"enabled": True})
    for source in identity.source_good_images:
        if file_sha256(Path(source.path)) != source.sha256:
            raise ValueError(f"Source Good hash mismatch: {source.image_id}")
    memory = ExperimentalAnomalyMemory.build(np.load(args.embeddings, allow_pickle=False), identity=identity, config=config)
    query_result = None
    if args.query_embeddings:
        result = memory.search(np.load(args.query_embeddings, allow_pickle=False))
        query_result = {"distances": result.anomaly_distances.tolist(), "source_indices": result.source_indices.tolist()}
    metadata = memory.save_new(output)
    print(json.dumps({"status": "EXPERIMENT_ONLY", "metadata": metadata.model_dump(mode="json"),
                      "query_result": query_result, "qc_performed": False, "registry_modified": False}, indent=2))


if __name__ == "__main__":
    main()
