"""Read-only cached-model feature diagnostic; explicit pattern flag, no QC/artifacts."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_feature_extractor import get_shared_dense_extractor
from backend.app.ai.pattern_pooling import PatternAwareDenseExtractor
from backend.app.core.paths import resolve_project_path
from backend.app.core.pattern_pooling import PatternPoolingConfig
from backend.app.services.fabric_registry import FabricRegistry


def cached_research_extractor(config: PatternPoolingConfig) -> PatternAwareDenseExtractor:
    import torch
    from torchvision.models import ResNet18_Weights

    backbone = FabricRegistry().shared_backbone
    if backbone.get("use_pretrained_weights", True):
        path = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet18_Weights.DEFAULT.url).name
        if not path.is_file():
            raise RuntimeError("Pretrained weights must be cached; diagnostic forbids downloads")
    with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Diagnostic forbids downloads")):
        baseline = get_shared_dense_extractor(backbone["backbone_version"], "cpu", bool(backbone.get("use_pretrained_weights", True)),
                                              str(resolve_project_path(backbone["adapter_path"])) if backbone.get("adapter_path") else "")
    return PatternAwareDenseExtractor(baseline, adapter_version=backbone.get("adapter_version"), config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/pattern_pooling_example.json")
    parser.add_argument("--enable-pattern-aware", action="store_true", help="Explicit research feature flag, required even if JSON enabled=true")
    parser.add_argument("--image-max-size", type=int, default=768)
    args = parser.parse_args()
    config = PatternPoolingConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    config = config.model_copy(update={"enabled": args.enable_pattern_aware})
    result = cached_research_extractor(config).extract_path(args.image, image_max_size=args.image_max_size)
    print(json.dumps({"status": "EXPERIMENT_ONLY", "pattern_pooling_enabled": config.enabled,
                      "geometry_signature": result.geometry.signature, "geometry": result.geometry.model_dump(mode="json"),
                      "embedding_shape": list(result.features.embeddings.shape), "pooling_plan": asdict(result.pooling_plan),
                      "observed_period": asdict(result.observed_period) if result.observed_period else None,
                      "qc_performed": False, "writes_performed": False}, indent=2))


if __name__ == "__main__":
    main()
