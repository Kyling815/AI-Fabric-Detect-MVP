import argparse
import csv
import json
import random
import sys
from collections import defaultdict
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
from backend.app.ai.patching import build_patch_grid, extract_image_patches
from backend.app.ai.preprocess import ImageReadError, build_preprocess_plan, inspect_image_file, load_resized_rgb_image
from backend.app.ai.projection_head import ProjectionHead, save_projection_checkpoint
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train an extended pairwise projection head on frozen CNN embeddings.")
    parser.add_argument("--manifest-csv", default="storage/metadata/extended_dataset_manifests/extended_all.csv")
    parser.add_argument("--output", default="storage/models/projection_head_extended_v1.pt")
    parser.add_argument("--report-output", default="storage/metadata/training_reports/projection_head_extended_v1_report.json")
    parser.add_argument("--max-images-per-group", type=int, default=80)
    parser.add_argument("--max-patches-per-image", type=int, default=80)
    parser.add_argument("--pairs-per-group", type=int, default=4000)
    parser.add_argument("--projection-dim", type=int, default=settings.projection_dim)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_records = collect_manifest_records(resolve_project_path(args.manifest_csv), args.max_images_per_group)
    if not image_records:
        raise SystemExit("No labeled image records found in manifest.")

    extractor = CNNFeatureExtractor(
        model_name=settings.model_name,
        device=settings.device,
        use_pretrained_weights=settings.use_pretrained_weights,
        model_input_size=settings.model_input_size,
    )
    embeddings, labels, groups = extract_embeddings(image_records, extractor, args.max_patches_per_image)
    if embeddings.shape[0] == 0:
        raise SystemExit("No patch embeddings extracted.")

    pair_left, pair_right, pair_labels = build_pairs(embeddings, labels, groups, args.pairs_per_group, args.seed)
    dataset = TensorDataset(
        torch.from_numpy(pair_left),
        torch.from_numpy(pair_right),
        torch.from_numpy(pair_labels.astype(np.float32)),
    )
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
        train_loss = run_pairwise_epoch(model, train_loader, optimizer, device, args.temperature)
        val_loss, val_accuracy = evaluate_pairwise(model, val_loader, device, args.temperature)
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
            "training_mode": "pairwise_projection",
            "image_count": len(image_records),
            "patch_count": int(embeddings.shape[0]),
            "pair_count": int(pair_labels.shape[0]),
            "positive_pair_count": int(pair_labels.sum()),
            "negative_pair_count": int(pair_labels.shape[0] - pair_labels.sum()),
            "history": history,
            "args": vars(args),
        },
    )
    report = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint_path": str(output_path),
        "training_mode": "pairwise_projection",
        "image_count": len(image_records),
        "patch_count": int(embeddings.shape[0]),
        "pair_count": int(pair_labels.shape[0]),
        "positive_pair_count": int(pair_labels.sum()),
        "negative_pair_count": int(pair_labels.shape[0] - pair_labels.sum()),
        "final": history[-1],
        "history": history,
        "args": vars(args),
    }
    report_path = resolve_project_path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_path": str(output_path), "report_path": str(report_path), "final": history[-1]}, indent=2))


def collect_manifest_records(manifest_path: Path, max_images_per_group: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in {"normal", "defective"}:
                continue
            grouped[(row["fabric_group"], row["label"])].append(row)

    records = []
    for key in sorted(grouped):
        records.extend(grouped[key][:max_images_per_group])
    return records


def extract_embeddings(records: list[dict], extractor: CNNFeatureExtractor, max_patches_per_image: int):
    settings = get_settings()
    all_embeddings = []
    all_labels = []
    all_groups = []
    for record in records:
        image_path = Path(record["image_path"])
        try:
            image_info = inspect_image_file(image_path)
        except ImageReadError:
            continue
        plan = build_preprocess_plan(
            width=image_info.width,
            height=image_info.height,
            target_max_size=settings.image_max_size,
            patch_size=settings.patch_size,
            patch_stride=settings.patch_stride,
        )
        image = load_resized_rgb_image(image_path, plan)
        grid = build_patch_grid(image.width, image.height, plan.patch_size, plan.patch_stride)
        patches = extract_image_patches(image, grid)
        if max_patches_per_image and len(patches) > max_patches_per_image:
            patches = patches[:max_patches_per_image]
        embeddings = extractor.extract((patch.image for patch in patches), batch_size=settings.batch_size)
        all_embeddings.append(embeddings)
        label = 0 if record["label"] == "normal" else 1
        all_labels.extend([label] * embeddings.shape[0])
        all_groups.extend([record["fabric_group"]] * embeddings.shape[0])

    if not all_embeddings:
        return (
            np.empty((0, extractor.info.embedding_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )
    return (
        np.concatenate(all_embeddings, axis=0).astype(np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_groups, dtype=object),
    )


def build_pairs(embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray, pairs_per_group: int, seed: int):
    rng = random.Random(seed)
    group_names = sorted(set(groups.tolist()))
    positive_pairs = []
    negative_pairs = []
    for group in group_names:
        normal_indices = np.where((groups == group) & (labels == 0))[0].tolist()
        defect_indices = np.where((groups == group) & (labels == 1))[0].tolist()
        other_normal_indices = np.where((groups != group) & (labels == 0))[0].tolist()
        if len(normal_indices) >= 2:
            for _ in range(pairs_per_group):
                left, right = rng.sample(normal_indices, 2)
                positive_pairs.append((left, right, 1.0))
        if normal_indices and defect_indices:
            for _ in range(pairs_per_group):
                negative_pairs.append((rng.choice(normal_indices), rng.choice(defect_indices), 0.0))
        if normal_indices and other_normal_indices:
            for _ in range(max(1, pairs_per_group // 2)):
                negative_pairs.append((rng.choice(normal_indices), rng.choice(other_normal_indices), 0.0))

    pairs = positive_pairs + negative_pairs
    if not pairs:
        raise SystemExit("No pairwise training pairs could be sampled from the manifest.")
    rng.shuffle(pairs)
    left = embeddings[[item[0] for item in pairs]]
    right = embeddings[[item[1] for item in pairs]]
    pair_labels = np.asarray([item[2] for item in pairs], dtype=np.float32)
    return left.astype(np.float32), right.astype(np.float32), pair_labels


def run_pairwise_epoch(model: ProjectionHead, loader: DataLoader, optimizer, device: torch.device, temperature: float) -> float:
    model.train()
    losses = []
    for left, right, labels in loader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = pair_logits(model, left, right, temperature)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def evaluate_pairwise(model: ProjectionHead, loader: DataLoader, device: torch.device, temperature: float) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = total = 0
    with torch.inference_mode():
        for left, right, labels in loader:
            left = left.to(device)
            right = right.to(device)
            labels = labels.to(device)
            logits = pair_logits(model, left, right, temperature)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            losses.append(float(loss.item()))
    return (sum(losses) / len(losses) if losses else 0.0, correct / total if total else 0.0)


def pair_logits(model: ProjectionHead, left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    left_features = model.forward_features(left)
    right_features = model.forward_features(right)
    return F.cosine_similarity(left_features, right_features) * temperature


if __name__ == "__main__":
    main()
