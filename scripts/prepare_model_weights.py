import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.app.ai.model_registry import get_model_spec, pretrained_weight_filename


def weight_path(model_name: str) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / pretrained_weight_filename(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or prepare torchvision pretrained weights.")
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Instantiate the model with pretrained weights, downloading them if missing.",
    )
    args = parser.parse_args()

    spec = get_model_spec(args.model_name)
    path = weight_path(args.model_name)

    if args.download and not path.exists():
        spec.build(use_pretrained_weights=True)

    result = {
        "model_name": spec.name,
        "embedding_dim": spec.embedding_dim,
        "weights_url": spec.weights_url,
        "weights_path": str(path),
        "cached": path.exists(),
        "next_step": (
            "Set FABRIC_USE_PRETRAINED_WEIGHTS=true and rebuild the reference library."
            if path.exists()
            else "Run this script with --download, then set FABRIC_USE_PRETRAINED_WEIGHTS=true."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
