import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


NORMAL_CLASS_NAMES = {"non defective", "non_defective", "good", "normal"}


def build_dme_split(
    dataset_root: Path,
    reference_normal_ratio: float,
    seed: int,
) -> dict:
    if not dataset_root.exists():
        raise FileNotFoundError(f"DME dataset root not found: {dataset_root}")
    if not 0 < reference_normal_ratio < 1:
        raise ValueError("reference-normal-ratio must be between 0 and 1.")

    rng = random.Random(seed)
    reference_normal: list[dict] = []
    evaluation: list[dict] = []
    class_counts: dict[str, dict[str, int]] = {}

    for class_dir in sorted((item for item in dataset_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        class_name = class_dir.name
        expected_defective = class_name.lower() not in NORMAL_CLASS_NAMES
        images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda item: str(item).lower(),
        )
        rng.shuffle(images)

        if expected_defective:
            eval_images = images
            reference_images: list[Path] = []
        else:
            reference_count = max(1, round(len(images) * reference_normal_ratio))
            reference_count = min(reference_count, max(1, len(images) - 1))
            reference_images = sorted(images[:reference_count], key=lambda item: str(item).lower())
            eval_images = sorted(images[reference_count:], key=lambda item: str(item).lower())

        reference_normal.extend(_records(reference_images, class_name, expected_defective=False))
        evaluation.extend(_records(eval_images, class_name, expected_defective=expected_defective))
        class_counts[class_name] = {
            "total": len(images),
            "reference": len(reference_images),
            "evaluation": len(eval_images),
            "expected_defective": int(expected_defective),
        }

    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_name": "dme_20250129",
        "dataset_root": str(dataset_root),
        "seed": seed,
        "reference_normal_ratio": reference_normal_ratio,
        "reference_normal": reference_normal,
        "evaluation": evaluation,
        "class_counts": class_counts,
    }


def _records(paths: list[Path], class_name: str, expected_defective: bool) -> list[dict]:
    return [
        {
            "path": str(path),
            "class_name": class_name,
            "expected_defective": expected_defective,
        }
        for path in paths
    ]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create a deterministic DME reference/evaluation split manifest.")
    parser.add_argument("--dataset-root", default=settings.dme_dataset_dir)
    parser.add_argument("--reference-normal-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    split = build_dme_split(
        dataset_root=resolve_project_path(args.dataset_root),
        reference_normal_ratio=args.reference_normal_ratio,
        seed=args.seed,
    )
    output_path = (
        Path(args.output)
        if args.output
        else resolve_project_path(settings.metadata_dir)
        / f"dme_split_seed{args.seed}_ref{int(args.reference_normal_ratio * 100)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    latest_path = resolve_project_path(settings.metadata_dir) / "latest_dme_split.json"
    latest_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "latest_path": str(latest_path),
                "reference_normal_count": len(split["reference_normal"]),
                "evaluation_count": len(split["evaluation"]),
                "class_counts": split["class_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
