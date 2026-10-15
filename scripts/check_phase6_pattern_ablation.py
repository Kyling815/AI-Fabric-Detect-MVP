"""Cached real-model fixed/adaptive pooling checks. Writes new memory in temp only.

Synthetic reference/query retrieval is not independent real evaluation. Changed
geometry must never query historical v5 banks; baseline scores are checked separately.
"""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_anomaly import DenseReferenceDetector
from backend.app.ai.experimental_memory import file_sha256
from backend.app.ai.pattern_memory import PatternAwareAnomalyMemory
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage
from backend.app.core.pattern_pooling import PatternPoolingConfig
from backend.app.services.fabric_registry import FabricRegistry
from scripts.inspect_pattern_pooling import cached_research_extractor
from tests.pattern_period_fixtures import pattern


def main() -> None:
    records, parity, memories = [], [], []
    extractors = {
        "baseline_off": cached_research_extractor(PatternPoolingConfig()),
        "fixed_multiscale": cached_research_extractor(PatternPoolingConfig(enabled=True, mode="fixed_multiscale")),
        "period_aware": cached_research_extractor(PatternPoolingConfig(enabled=True)),
    }
    reference_results = {}
    for kind in ("checker", "dots", "stripes", "uniform", "noise", "stretch", "lighting", "rotate"):
        image = Image.fromarray(pattern(kind))
        baseline = extractors["baseline_off"].baseline.extract_image(image, image_max_size=256)
        for branch, extractor in extractors.items():
            first = extractor.extract_image(image, image_max_size=256)
            second = extractor.extract_image(image, image_max_size=256)
            np.testing.assert_array_equal(first.features.embeddings, second.features.embeddings)
            assert first.features.embeddings.shape == baseline.embeddings.shape and first.features.embeddings.shape[1] == 128
            assert first.features.boxes == baseline.boxes
            assert np.isfinite(first.features.embeddings).all()
            np.testing.assert_allclose(np.linalg.norm(first.features.embeddings, axis=1), 1, atol=1e-5)
            if branch == "baseline_off":
                np.testing.assert_array_equal(first.features.embeddings, baseline.embeddings)
            if kind == "checker":
                reference_results[branch] = first
            records.append({"input": kind, "branch": branch, "embedding_shape": list(first.features.embeddings.shape),
                            "pooling_plan": asdict(first.pooling_plan), "observed_period": asdict(first.observed_period) if first.observed_period else None,
                            "repeat_embeddings_equal": True, "baseline_embedding_equal": True if branch == "baseline_off" else None,
                            "geometry_signature": first.geometry.signature,
                            "mean_abs_embedding_difference_from_baseline": float(np.abs(first.features.embeddings - baseline.embeddings).mean())})
    registry = FabricRegistry()
    baseline = reference_results["baseline_off"].features
    for fabric in registry.list_fabrics():
        reference = registry.get_reference(fabric["fabric_id"], f"{fabric['fabric_id']}_ref_v5")
        detector = DenseReferenceDetector(reference.metadata_path)
        expected = detector.score(baseline.embeddings, baseline.boxes, grid_rows=baseline.grid_rows, grid_cols=baseline.grid_cols,
                                  image_width=baseline.processed_width, image_height=baseline.processed_height, thresholds=reference.thresholds)
        unwrapped = extractors["baseline_off"].baseline.extract_image(Image.fromarray(pattern("checker")), image_max_size=256)
        actual = detector.score(unwrapped.embeddings, unwrapped.boxes, grid_rows=unwrapped.grid_rows, grid_cols=unwrapped.grid_cols,
                                image_width=unwrapped.processed_width, image_height=unwrapped.processed_height, thresholds=reference.thresholds)
        np.testing.assert_array_equal(expected.patch_scores, actual.patch_scores)
        assert expected.image_score == actual.image_score and expected.boxes == actual.boxes
        parity.append({"fabric_id": fabric["fabric_id"], "baseline_OFF_scores_equal": True, "baseline_OFF_boxes_equal": True,
                       "old_bank_used_with_experimental_geometry": False})
    with TemporaryDirectory(prefix="fabric-phase6-memory-") as directory:
        root = Path(directory)
        path = root / "synthetic_reference.png"
        Image.fromarray(pattern("checker")).save(path)
        source = SourceGoodImage(image_id="synthetic_checker_reference", path=str(path), sha256=file_sha256(path),
                                 fabric_id="fabric_001", lot_id=None)
        for branch, value in reference_results.items():
            identity = MemoryIdentity(reference_version=f"phase6_{branch}_synthetic_v1", encoder_version=value.geometry.encoder_version,
                                      adapter_version=value.geometry.adapter_version, fabric_id="fabric_001", lot_id=None,
                                      threshold_profile_version=None, source_good_images=(source,))
            config = MemoryExperimentConfig(enabled=True)
            memory = PatternAwareAnomalyMemory.build([value], identity=identity, config=config)
            memory.save_new(root / branch)
            restored = PatternAwareAnomalyMemory.load(root / branch, expected_geometry=value.geometry, expected_identity=identity, config=config)
            np.testing.assert_array_equal(memory.search(value).anomaly_distances, restored.search(value).anomaly_distances)
            wrong = reference_results["baseline_off"] if branch != "baseline_off" else reference_results["period_aware"]
            try:
                restored.search(wrong)
            except ValueError:
                pass
            else:
                raise AssertionError("Different feature geometry was silently accepted")
            memories.append({"branch": branch, "feature_dim": value.geometry.feature_dim, "reload_scores_equal": True,
                             "same_dim_wrong_geometry_rejected": True, "status": "NOT_CALIBRATED"})
    print(json.dumps({"phase": 6, "scope": "cached CPU CNN/adapter, synthetic pooling ablations, v5 baseline-only queries; new memories in temp",
                      "feature_cases": records, "baseline_bank_parity": parity, "new_geometry_memories": memories,
                      "real_evaluation_performed": False, "improvement_status": "UNVERIFIED_NO_INDEPENDENT_EVALUATION",
                      "limitations": "Embedding differences/retrieval verify software only. No accuracy or throughput improvement claim. Adapter was not retrained; no old bank queried with changed geometry."}, indent=2))


if __name__ == "__main__":
    main()
