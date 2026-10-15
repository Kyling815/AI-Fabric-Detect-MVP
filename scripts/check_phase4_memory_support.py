"""Read cached real model/v5 banks, verify offline metrics/subsets in temp only.

No training, activation, calibration, production DB access or accuracy claims.
Ratios are relative to the INPUT bank (v5 is already a historical subset).
"""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.current_anomaly import CurrentAnomalyDetector
from backend.app.ai.dense_anomaly import DenseReferenceDetector
from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, file_sha256
from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionContext, InspectionInput
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage
from backend.app.services.fabric_registry import FabricRegistry


def main() -> None:
    import torch
    from torchvision.models import ResNet18_Weights

    registry = FabricRegistry()
    backbone = registry.shared_backbone
    if backbone.get("use_pretrained_weights", True):
        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet18_Weights.DEFAULT.url).name
        if not cached.is_file():
            raise RuntimeError("Registered pretrained weights must be cached; audit forbids downloads")
    records, parity = [], []
    yy, xx = np.indices((96, 128))
    checker = np.repeat((80 + 120 * ((xx // 8 + yy // 8) % 2)).astype(np.uint8)[..., None], 3, axis=2)
    scratch = checker.copy()
    scratch[:, 58:62] = 10
    with TemporaryDirectory(prefix="fabric-phase4-real-memory-") as directory:
        root = Path(directory)
        paths = []
        for name, values in (("synthetic_checker", checker), ("synthetic_scratch", scratch)):
            path = root / (name + ".png")
            Image.fromarray(values).save(path)
            paths.append((name, path))
        with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Audit forbids downloads")):
            for fabric in registry.list_fabrics():
                fabric_id = fabric["fabric_id"]
                reference = registry.get_reference(fabric_id, f"{fabric_id}_ref_v5")
                context = InspectionContext(reference, backbone, "cpu", get_settings().image_max_size)
                detector = DenseReferenceDetector(reference.metadata_path)
                metadata = detector.metadata
                good_path = Path(metadata["images"][0]["path"])
                sample_queries = None
                for name, path in [*paths, ("existing_reference_good", good_path)]:
                    # Explicit read-only component call; original registry status is never changed.
                    evidence = CurrentAnomalyDetector().inspect(InspectionInput(path, fabric_id), context)
                    features = evidence.features
                    baseline = detector.score(features.embeddings, features.boxes,
                                              grid_rows=features.grid_rows, grid_cols=features.grid_cols,
                                              image_width=features.processed_width, image_height=features.processed_height,
                                              thresholds=reference.thresholds)
                    np.testing.assert_array_equal(evidence.anomaly.patch_scores, baseline.patch_scores)
                    np.testing.assert_array_equal(evidence.anomaly.anomaly_map, baseline.anomaly_map)
                    assert evidence.anomaly.image_score == baseline.image_score
                    assert evidence.anomaly.localized_score == baseline.localized_score
                    assert evidence.anomaly.boxes == baseline.boxes
                    parity.append({"fabric_id": fabric_id, "input": name, "image_score": baseline.image_score,
                                   "patch_scores_equal": True, "anomaly_map_equal": True,
                                   "localized_score_equal": True, "boxes_equal": True})
                    if sample_queries is None:
                        sample_queries = features.embeddings[:16].copy()
                values = np.load(reference.memory_bank_path, allow_pickle=False)
                sources = tuple(SourceGoodImage(image_id=entry["image_id"], path=entry["path"],
                                                 sha256=file_sha256(Path(entry["path"])), fabric_id=fabric_id,
                                                 lot_id=None, split="reference") for entry in metadata["images"])
                provenance = MemoryIdentity(
                    reference_version=reference.reference_id + "_phase4_diagnostic",
                    encoder_version=reference.backbone_version, adapter_version=backbone.get("adapter_version"),
                    fabric_id=fabric_id, lot_id=None, source_good_images=sources,
                    threshold_profile_version=f"source_{reference.reference_id}_schema_{metadata['thresholds'].get('threshold_schema_version', 'legacy')}",
                )
                baseline_distances = np.maximum(0, 1 - detector.store.search(detector.index, sample_queries).scores)
                for metric in ("current", "cosine", "l2"):
                    for ratio in (1.0, 0.25, 0.10, 0.05):
                        config = MemoryExperimentConfig(enabled=True, distance_metric=metric, coreset_ratio=ratio)
                        memory = ExperimentalAnomalyMemory.build(values, identity=provenance, config=config)
                        result = memory.search(sample_queries)
                        artifact_dir = root / fabric_id / f"{metric}_{ratio}"
                        saved = memory.save_new(artifact_dir)
                        loaded = ExperimentalAnomalyMemory.load(artifact_dir, expected_identity=provenance, config=config)
                        reproduced = loaded.search(sample_queries)
                        np.testing.assert_array_equal(result.indices, reproduced.indices)
                        np.testing.assert_array_equal(result.anomaly_distances, reproduced.anomaly_distances)
                        preserved = metric == "current" and ratio == 1
                        if preserved:
                            np.testing.assert_array_equal(memory.bank, values)
                            np.testing.assert_array_equal(result.anomaly_distances, baseline_distances)
                        records.append({"fabric_id": fabric_id, "requested_metric": metric,
                                        "distance_metric": saved.distance_metric, "score_definition": saved.score_definition,
                                        "coreset_ratio": ratio, "input_bank_count": len(values),
                                        "memory_bank_count": len(memory.bank), "reload_scores_equal": True,
                                        "baseline_exactly_preserved": True if preserved else None,
                                        "calibration": saved.threshold_calibration})
    print(json.dumps({"phase": 4, "scope": "cached CPU CNN/adapter and existing v5 banks; all new artifacts in temporary directories",
                      "runtime_parity": parity, "memory_experiments": records,
                      "limitations": "Software checks only; no unseen/locked evaluation. Good input is a reference source. Ratios apply to existing 12000-vector banks, not original full training features. No metric/ratio is approved for QC."}, indent=2))


if __name__ == "__main__":
    main()
