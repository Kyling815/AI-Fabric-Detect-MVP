import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from backend.app.ai.feature_extractor import CNNFeatureExtractor
from backend.app.ai.preprocess import ImageReadError
from backend.app.ai.preprocessing.adapter import prepare_image
from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.ai.projection_head import ProjectionHead, save_projection_checkpoint
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path
from backend.app.services.dataset_service import IMAGE_EXTENSIONS


NORMAL_CLASS_NAMES = {"non defective", "non_defective", "good", "normal"}


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train a lightweight projection head on frozen CNN patch embeddings.")
    parser.add_argument("--dataset-root", default=settings.dme_dataset_dir)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--output", default="storage/models/projection_head_dme.pt")
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--max-normal-images", type=int, default=None)
    parser.add_argument("--max-defect-images-per-class", type=int, default=None)
    parser.add_argument("--max-patches-per-image", type=int, default=160)
    parser.add_argument("--projection-dim", type=int, default=settings.projection_dim)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--preprocessing-config", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.manifest_csv:
        image_records = collect_from_manifest_csv(Path(args.manifest_csv), args.max_normal_images, args.max_defect_images_per_class)
    elif args.split_manifest:
        image_records = collect_from_split_manifest(Path(args.split_manifest), args.max_normal_images, args.max_defect_images_per_class)
    else:
        image_records = collect_from_dme_root(
            resolve_project_path(args.dataset_root),
            args.max_normal_images,
            args.max_defect_images_per_class,
        )
    if not image_records:
        raise SystemExit("No training images found.")

    extractor = CNNFeatureExtractor(
        model_name=settings.model_name,
        device=settings.device,
        use_pretrained_weights=settings.use_pretrained_weights,
        model_input_size=settings.model_input_size,
    )
    preprocessing_config = (
        (PreprocessConfig.load(Path(args.preprocessing_config)) if args.preprocessing_config else PreprocessConfig()).to_dict()
        if args.preprocessing_version == "v2"
        else None
    )
    embeddings, labels = extract_training_embeddings(
        image_records=image_records,
        extractor=extractor,
        max_patches_per_image=args.max_patches_per_image,
        preprocessing_version=args.preprocessing_version,
        preprocessing_config=preprocessing_config,
    )
    if embeddings.shape[0] == 0:
        raise SystemExit("No patches were extracted for training.")

    dataset = TensorDataset(torch.from_numpy(embeddings), torch.from_numpy(labels.astype(np.float32)))
    val_count = max(1, round(len(dataset) * args.validation_ratio))
    train_count = max(1, len(dataset) - val_count)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device(extractor.info.device)
    model = ProjectionHead(
        input_dim=embeddings.shape[1],
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"early_stopping epoch={epoch} patience={args.early_stopping_patience}")
                break

    output_path = resolve_project_path(args.output)
    save_projection_checkpoint(
        model=model,
        path=output_path,
        base_model_name=settings.model_name,
        use_pretrained_weights=settings.use_pretrained_weights,
        extra={
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "image_count": len(image_records),
            "patch_count": int(embeddings.shape[0]),
            "normal_patch_count": int((labels == 0).sum()),
            "defect_patch_count": int((labels == 1).sum()),
            "history": history,
            "args": vars(args),
            "preprocessing_version": args.preprocessing_version,
            "preprocessing_config": preprocessing_config,
            "configuration_hash": (
                PreprocessConfig.from_mapping(preprocessing_config).configuration_hash
                if preprocessing_config
                else None
            ),
        },
    )
    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_path": str(output_path),
        "image_count": len(image_records),
        "patch_count": int(embeddings.shape[0]),
        "normal_patch_count": int((labels == 0).sum()),
        "defect_patch_count": int((labels == 1).sum()),
        "history": history,
        "final": history[-1],
        "args": vars(args),
        "preprocessing_version": args.preprocessing_version,
        "preprocessing_config": preprocessing_config,
        "configuration_hash": (
            PreprocessConfig.from_mapping(preprocessing_config).configuration_hash
            if preprocessing_config
            else None
        ),
        "model": {
            "base_model_name": settings.model_name,
            "use_pretrained_weights": settings.use_pretrained_weights,
            "projection_dim": args.projection_dim,
            "frozen_backbone": True,
        },
    }
    if args.report_output:
        report_path = resolve_project_path(args.report_output)
    else:
        report_path = resolve_project_path("storage/metadata/training_reports") / f"{output_path.stem}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "image_count": len(image_records),
                "patch_count": int(embeddings.shape[0]),
                "normal_patch_count": int((labels == 0).sum()),
                "defect_patch_count": int((labels == 1).sum()),
                "final": history[-1],
                "report_path": str(report_path),
                "next_step": "Set FABRIC_USE_PROJECTION_HEAD=true and FABRIC_PROJECTION_HEAD_PATH to this checkpoint, then rebuild references.",
            },
            indent=2,
        )
    )


def collect_from_dme_root(
    dataset_root: Path,
    max_normal_images: int | None,
    max_defect_images_per_class: int | None,
) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    for class_dir in sorted((item for item in dataset_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        is_normal = class_dir.name.lower() in NORMAL_CLASS_NAMES
        limit = max_normal_images if is_normal else max_defect_images_per_class
        label = 0 if is_normal else 1
        images = _list_images(class_dir, limit)
        records.extend((path, label) for path in images)
    return records


def collect_from_split_manifest(
    manifest_path: Path,
    max_normal_images: int | None,
    max_defect_images_per_class: int | None,
) -> list[tuple[Path, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normal_records = manifest["reference_normal"]
    if max_normal_images is not None:
        normal_records = normal_records[:max_normal_images]
    records = [(Path(item["path"]), 0) for item in normal_records]

    grouped_defects: dict[str, list[dict]] = {}
    for item in manifest["evaluation"]:
        if item["expected_defective"]:
            grouped_defects.setdefault(item["class_name"], []).append(item)
    for class_name in sorted(grouped_defects):
        defect_records = sorted(grouped_defects[class_name], key=lambda item: item["path"].lower())
        if max_defect_images_per_class is not None:
            defect_records = defect_records[:max_defect_images_per_class]
        records.extend((Path(item["path"]), 1) for item in defect_records)
    return records


def collect_from_manifest_csv(
    manifest_path: Path,
    max_normal_images: int | None,
    max_defect_images_per_class: int | None,
) -> list[tuple[Path, int]]:
    grouped_defects: dict[str, list[Path]] = {}
    normal_paths: list[Path] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label", "").lower()
            path = Path(row["image_path"])
            if label == "normal":
                normal_paths.append(path)
            elif label == "defective":
                key = row.get("fabric_group") or row.get("defect_type") or "defective"
                grouped_defects.setdefault(key, []).append(path)

    if max_normal_images is not None:
        normal_paths = normal_paths[:max_normal_images]
    records = [(path, 0) for path in normal_paths]
    for key in sorted(grouped_defects):
        paths = sorted(grouped_defects[key], key=lambda item: str(item).lower())
        if max_defect_images_per_class is not None:
            paths = paths[:max_defect_images_per_class]
        records.extend((path, 1) for path in paths)
    return records


def extract_training_embeddings(
    image_records: list[tuple[Path, int]],
    extractor: CNNFeatureExtractor,
    max_patches_per_image: int,
    preprocessing_version: str = "v1",
    preprocessing_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    settings = get_settings()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for image_path, label in image_records:
        try:
            prepared = prepare_image(
                image_path,
                preprocessing_version=preprocessing_version,
                image_max_size=settings.image_max_size,
                patch_size=settings.patch_size,
                patch_stride=settings.patch_stride,
                preprocessing_config=preprocessing_config,
                split="train",
            )
        except (ImageReadError, OSError, ValueError):
            continue
        patches = prepared.patches
        if max_patches_per_image and len(patches) > max_patches_per_image:
            patches = patches[:max_patches_per_image]
        embeddings = extractor.extract((patch.image for patch in patches), batch_size=settings.batch_size)
        all_embeddings.append(embeddings)
        all_labels.append(np.full((embeddings.shape[0],), label, dtype=np.float32))

    if not all_embeddings:
        return np.empty((0, extractor.info.embedding_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(all_embeddings, axis=0).astype(np.float32), np.concatenate(all_labels, axis=0).astype(np.float32)


def run_epoch(model: ProjectionHead, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def evaluate(model: ProjectionHead, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = total = 0
    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model.forward_logits(embeddings)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            losses.append(float(loss.item()))
    return (sum(losses) / len(losses) if losses else 0.0, correct / total if total else 0.0)


def _list_images(root: Path, limit: int | None) -> list[Path]:
    images = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
    return images[:limit] if limit is not None else images


if __name__ == "__main__":
    main()
