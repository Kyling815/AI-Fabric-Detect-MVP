"""Read-only real-model/global-bank checks with a SYNTHETIC diagnostic lot.

Not an independent lot evaluation: reference-source inputs and synthetic checker
retrieval exercise infrastructure only. No lot-id inference, activation or QC.
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
from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, file_sha256
from backend.app.ai.lot_memory import FabricLotAnomalyMemory
from backend.app.core.config import get_settings
from backend.app.core.inspection_contracts import InspectionContext, InspectionInput
from backend.app.core.lot_memory import FabricLotMemoryManifest, LotMemoryConfig, MemoryArtifactSpec
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage
from backend.app.services.fabric_registry import FabricRegistry


def main() -> None:
    import torch
    from torchvision.models import ResNet18_Weights

    registry = FabricRegistry()
    backbone = registry.shared_backbone
    if backbone.get("use_pretrained_weights", True):
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet18_Weights.DEFAULT.url).name
        if not checkpoint.is_file():
            raise RuntimeError("Audit requires cached pretrained weights and forbids downloads")
    records = []
    with TemporaryDirectory(prefix="fabric-phase5-real-hierarchy-") as directory:
        root = Path(directory)
        yy, xx = np.indices((96, 128))
        checker = np.repeat((80 + 120 * ((xx // 8 + yy // 8) % 2)).astype(np.uint8)[..., None], 3, axis=2)
        scratch = checker.copy()
        scratch[:, 58:62] = 10
        checker_path, scratch_path = root / "synthetic_lot_reference.png", root / "synthetic_scratch.png"
        Image.fromarray(checker).save(checker_path)
        Image.fromarray(scratch).save(scratch_path)
        with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Audit forbids downloads")):
            for fabric in registry.list_fabrics():
                fabric_id = fabric["fabric_id"]
                reference = registry.get_reference(fabric_id, f"{fabric_id}_ref_v5")
                metadata = json.loads(reference.metadata_path.read_text(encoding="utf-8"))
                context = InspectionContext(reference, backbone, "cpu", get_settings().image_max_size)
                sources = tuple(SourceGoodImage(image_id=entry["image_id"], path=entry["path"],
                                                 sha256=file_sha256(Path(entry["path"])), fabric_id=fabric_id,
                                                 lot_id=None, split="reference") for entry in metadata["images"])
                identity = MemoryIdentity(reference_version=reference.reference_id + "_phase5_global_diagnostic",
                                          encoder_version=reference.backbone_version, adapter_version=backbone.get("adapter_version"),
                                          fabric_id=fabric_id, lot_id=None, threshold_profile_version=None, source_good_images=sources)
                config = MemoryExperimentConfig(enabled=True)
                global_memory = ExperimentalAnomalyMemory.build(np.load(reference.memory_bank_path, allow_pickle=False),
                                                                 identity=identity, config=config)
                lot_id = "synthetic_diagnostic_lot"
                lot_source = SourceGoodImage(image_id=f"synthetic_{fabric_id}_diagnostic_reference", path=str(checker_path),
                                             sha256=file_sha256(checker_path), fabric_id=fabric_id, lot_id=lot_id)
                lot_identity = MemoryIdentity(reference_version=reference.reference_id + "_phase5_synthetic_lot_diagnostic",
                                              encoder_version=reference.backbone_version, adapter_version=backbone.get("adapter_version"),
                                              fabric_id=fabric_id, lot_id=lot_id, threshold_profile_version=None,
                                              source_good_images=(lot_source,))
                reference_evidence = CurrentAnomalyDetector().inspect(InspectionInput(checker_path, fabric_id), context)
                lot_memory = ExperimentalAnomalyMemory.build(reference_evidence.features.embeddings[:32],
                                                              identity=lot_identity, config=config)
                global_path, lot_path = root / fabric_id / "global", root / fabric_id / "lot"
                global_memory.save_new(global_path)
                lot_memory.save_new(lot_path)
                manifest = FabricLotMemoryManifest(
                    fabric_id=fabric_id, lot_id=lot_id, config=LotMemoryConfig(enabled=True, mode="global_plus_lot"),
                    global_memory=MemoryArtifactSpec(path=str(global_path), identity=identity, config=config),
                    lot_memory=MemoryArtifactSpec(path=str(lot_path), identity=lot_identity, config=config),
                )
                combined = FabricLotAnomalyMemory.load(manifest)
                restored = FabricLotAnomalyMemory.load(FabricLotMemoryManifest.model_validate_json(manifest.model_dump_json()))
                disabled = FabricLotAnomalyMemory.load(manifest.model_copy(update={"config": LotMemoryConfig(mode="global_plus_lot")}))
                global_only = FabricLotAnomalyMemory(fabric_id=fabric_id, global_memory=global_memory, config=LotMemoryConfig())
                lot_only = FabricLotAnomalyMemory(fabric_id=fabric_id, lot_id=lot_id, lot_memory=lot_memory,
                                                config=LotMemoryConfig(enabled=True, mode="lot_only"))
                for name, path in (("synthetic_checker_retrieval", checker_path), ("synthetic_scratch", scratch_path),
                                   ("existing_reference_good", Path(metadata["images"][0]["path"]))):
                    evidence = CurrentAnomalyDetector().inspect(InspectionInput(path, fabric_id), context)
                    queries = evidence.features.embeddings
                    global_result, lot_result, union_result = global_only.search(queries), lot_only.search(queries), combined.search(queries)
                    np.testing.assert_array_equal(global_result.anomaly_distances[:, 0], evidence.anomaly.patch_scores)
                    np.testing.assert_array_equal(disabled.search(queries).anomaly_distances, global_result.anomaly_distances)
                    np.testing.assert_allclose(union_result.anomaly_distances,
                                               np.minimum(global_result.anomaly_distances, lot_result.anomaly_distances),
                                               atol=1e-6, rtol=0)
                    np.testing.assert_array_equal(restored.search(queries).anomaly_distances, union_result.anomaly_distances)
                    records.append({"fabric_id": fabric_id, "input": name, "global_vector_count": len(global_memory.bank),
                                    "synthetic_lot_vector_count": len(lot_memory.bank), "feature_dim": queries.shape[1],
                                    "global_only_patch_scores_equal": True, "OFF_exact_baseline": True,
                                    "union_min_rule_matches": True, "combined_reload_exact": True})
    print(json.dumps({"phase": 5, "scope": "cached CPU CNN/adapter and v5 global banks; all new artifacts in temp",
                      "cases": records, "real_lot_evaluation_performed": False,
                      "improvement_status": "UNVERIFIED_NO_INDEPENDENT_REAL_LOT_DATA",
                      "limitations": "Synthetic diagnostic lot, not a real approved lot. Checker retrieval and existing reference Good inputs are not independent evaluation. No accuracy/throughput, QC or activation claims."}, indent=2))


if __name__ == "__main__":
    main()
