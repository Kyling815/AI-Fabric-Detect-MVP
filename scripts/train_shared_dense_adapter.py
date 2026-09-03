import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.dense_feature_extractor import BASE_BACKBONE_VERSION, get_shared_dense_extractor
from backend.app.ai.projection_head import ProjectionHead, save_projection_checkpoint
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train one shared supervised adapter over frozen dense ResNet18 features."
    )
    parser.add_argument("--manifest-dir", default="storage/metadata/company_demo_manifests")
    parser.add_argument("--output", default="storage/models/shared_dense_adapter_company_v1.pt")
    parser.add_argument(
        "--report-output",
        default="storage/metadata/training_reports/shared_dense_adapter_company_v1.json",
    )
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-normal-patches-per-image", type=int, default=128)
    parser.add_argument("--max-defect-patches-per-image", type=int, default=512)
    parser.add_argument("--pairs-per-fabric", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_dir = resolve_project_path(args.manifest_dir)
    rows = []
    for path in sorted(manifest_dir.glob("fabric_*_all.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    training_rows = [
        row
        for row in rows
        if (
            row["label"] == "normal"
            and row["split"] in {"reference", "development_only", "adapter_train"}
        )
        or (row["label"] == "defective" and row["split"] == "adapter_train")
    ]
    if not training_rows:
        raise RuntimeError("No adapter training rows found")

    extractor = get_shared_dense_extractor(
        BASE_BACKBONE_VERSION,
        settings.device,
        True,
    )
    features, labels, fabrics, extraction = extract_training_features(
        training_rows,
        extractor,
        image_max_size=settings.image_max_size,
        max_normal=args.max_normal_patches_per_image,
        max_defect=args.max_defect_patches_per_image,
    )
    left, right, pair_labels, pair_report = build_pairs(
        features,
        labels,
        fabrics,
        pairs_per_fabric=args.pairs_per_fabric,
        seed=args.seed,
    )
    dataset = TensorDataset(
        torch.from_numpy(left),
        torch.from_numpy(right),
        torch.from_numpy(pair_labels),
    )
    val_count = max(1, round(len(dataset) * args.validation_ratio))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    device = torch.device(extractor.info.device)
    model = ProjectionHead(
        input_dim=features.shape[1],
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.temperature)
        val_loss, val_accuracy = evaluate(model, val_loader, device, args.temperature)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pair_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_pair_accuracy={val_accuracy:.4f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Adapter training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output = resolve_project_path(args.output)
    best = next(item for item in history if item["epoch"] == best_epoch)
    extra = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "training_mode": "shared_dense_mask_supervised_pairwise_adapter",
        "base_backbone_version": BASE_BACKBONE_VERSION,
        "source_splits": ["reference", "development_only", "adapter_train"],
        "locked_test_used": False,
        "extraction": extraction,
        "pair_report": pair_report,
        "best": best,
        "history": history,
        "args": vars(args),
    }
    save_projection_checkpoint(
        model=model,
        path=output,
        base_model_name="resnet18_layer2_layer3_dense",
        use_pretrained_weights=True,
        extra=extra,
    )
    report = {"checkpoint_path": str(output), **extra}
    report_path = resolve_project_path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_path": str(output), "report_path": str(report_path), "best": best}, indent=2))


def extract_training_features(rows, extractor, *, image_max_size: int, max_normal: int, max_defect: int):
    matrices = []
    labels = []
    fabrics = []
    counts = defaultdict(lambda: {"normal_images": 0, "defect_images": 0, "normal_patches": 0, "defect_patches": 0})
    for index, row in enumerate(rows, start=1):
        print(f"extract {index}/{len(rows)} {row['fabric_id']} {Path(row['image_path']).name}", flush=True)
        dense = extractor.extract_path(Path(row["image_path"]), image_max_size=image_max_size)
        if row["label"] == "defective":
            selected = masked_feature_indexes(row["mask_path"], dense)
            maximum = max_defect
        else:
            selected = np.arange(dense.embeddings.shape[0])
            maximum = max_normal
        if selected.size == 0:
            continue
        if maximum > 0 and selected.size > maximum:
            positions = np.linspace(0, selected.size - 1, num=maximum, dtype=int)
            selected = selected[positions]
        matrix = dense.embeddings[selected].astype(np.float32)
        label = 1 if row["label"] == "defective" else 0
        matrices.append(matrix)
        labels.extend([label] * len(matrix))
        fabrics.extend([row["fabric_id"]] * len(matrix))
        key = "defect" if label else "normal"
        counts[row["fabric_id"]][f"{key}_images"] += 1
        counts[row["fabric_id"]][f"{key}_patches"] += len(matrix)
    if not matrices:
        raise RuntimeError("Dense extraction produced no adapter features")
    return (
        np.concatenate(matrices).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(fabrics, dtype=object),
        dict(counts),
    )


def masked_feature_indexes(mask_path: str, dense) -> np.ndarray:
    if not mask_path or not Path(mask_path).exists():
        raise RuntimeError(f"Adapter defect row requires a mask: {mask_path}")
    with Image.open(mask_path) as source:
        mask = ImageOps.exif_transpose(source).convert("L")
        mask = mask.resize((dense.processed_width, dense.processed_height), Image.Resampling.NEAREST)
        values = np.asarray(mask) > 0
    selected = []
    for index, box in enumerate(dense.boxes):
        region = values[box.y : box.y + box.h, box.x : box.x + box.w]
        if region.any():
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def build_pairs(features, labels, fabrics, *, pairs_per_fabric: int, seed: int):
    rng = np.random.default_rng(seed)
    left = []
    right = []
    targets = []
    report = {}
    for fabric_id in sorted(set(fabrics.tolist())):
        normal = np.where((fabrics == fabric_id) & (labels == 0))[0]
        defect = np.where((fabrics == fabric_id) & (labels == 1))[0]
        if len(normal) < 2 or len(defect) < 1:
            raise RuntimeError(f"{fabric_id} requires normal and masked-defect adapter features")
        normalized_normal = features[normal]
        normalized_defect = features[defect]
        # Hard negatives are the base dense normal patches most similar to each
        # masked defect patch. Pushing these apart targets the former false-negative mode.
        similarities = normalized_defect @ normalized_normal.T
        hard_normal = normal[np.argmax(similarities, axis=1)]
        for pair_index in range(pairs_per_fabric):
            first, second = rng.choice(normal, size=2, replace=False)
            left.append(features[first])
            right.append(features[second])
            targets.append(1.0)
            defect_offset = pair_index % len(defect)
            left.append(features[hard_normal[defect_offset]])
            right.append(features[defect[defect_offset]])
            targets.append(0.0)
        report[fabric_id] = {
            "normal_feature_count": int(len(normal)),
            "defect_feature_count": int(len(defect)),
            "positive_pairs": pairs_per_fabric,
            "hard_negative_pairs": pairs_per_fabric,
        }
    order = rng.permutation(len(targets))
    return (
        np.asarray(left, dtype=np.float32)[order],
        np.asarray(right, dtype=np.float32)[order],
        np.asarray(targets, dtype=np.float32)[order],
        report,
    )


def run_epoch(model, loader, optimizer, device, temperature: float) -> float:
    model.train()
    losses = []
    for left, right, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = pair_logits(model, left.to(device), right.to(device), temperature)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device, temperature: float):
    model.eval()
    losses = []
    correct = total = 0
    with torch.inference_mode():
        for left, right, labels in loader:
            logits = pair_logits(model, left.to(device), right.to(device), temperature)
            target = labels.to(device)
            losses.append(float(F.binary_cross_entropy_with_logits(logits, target).item()))
            correct += int(((torch.sigmoid(logits) >= 0.5) == target.bool()).sum().item())
            total += int(target.numel())
    return (float(np.mean(losses)) if losses else 0.0, correct / max(1, total))


def pair_logits(model, left, right, temperature: float):
    return F.cosine_similarity(model.forward_features(left), model.forward_features(right)) * temperature


if __name__ == "__main__":
    main()
