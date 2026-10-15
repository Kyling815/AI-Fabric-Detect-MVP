"""Offline supervised trainer. No API, anomaly model, registry or locked-test use."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.app.ai.segmentation_dataset import SegmentationDataset, load_segmentation_sample
from backend.app.ai.segmentation_loss import CrossEntropyDiceLoss
from backend.app.ai.segmentation_metrics import SegmentationMetrics
from backend.app.ai.segmentation_model import ARCHITECTURE_VERSION, PREPROCESSING, ResNet18UNet
from backend.app.ai.segmentation_split import grouped_segmentation_split
from backend.app.core.segmentation_data import SegmentationManifest, SegmentationTaxonomy
from backend.app.core.segmentation_training import SegmentationBaselineConfig


def resolve_training_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported training device")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


@contextmanager
def training_reproducibility(seed: int, cpu_threads: int):
    """Restore process RNG/thread/backend settings; deterministic single-worker loaders."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, cudnn_deterministic


def baseline_loss(config: SegmentationBaselineConfig) -> CrossEntropyDiceLoss:
    return CrossEntropyDiceLoss(ignore_index=config.pipeline.taxonomy.ignore_index,
                                ce_weight=config.ce_weight, dice_weight=config.dice_weight,
                                focal_gamma=config.focal_gamma if config.pipeline.training.focal_optional else None)


def build_optimizer(model: ResNet18UNet, config: SegmentationBaselineConfig) -> torch.optim.AdamW:
    training = config.pipeline.training
    return torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": training.encoder_lr, "name": "supervised_encoder"},
        {"params": list(model.decoder.parameters()) + list(model.head.parameters()), "lr": training.decoder_lr, "name": "decoder_head"},
    ], weight_decay=training.weight_decay)


@torch.inference_mode()
def evaluate_segmentation(model: ResNet18UNet, loader: DataLoader, criterion: CrossEntropyDiceLoss,
                          *, device: torch.device) -> dict:
    if len(loader.dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    previous = model.training
    metrics = SegmentationMetrics(model.taxonomy.classes, ignore_index=model.taxonomy.ignore_index)
    total_loss, samples = 0.0, 0
    try:
        model.eval()
        for batch in loader:
            images, targets = batch["image"].to(device), batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation loss")
            total_loss += float(loss) * len(images)
            samples += len(images)
            metrics.update(logits.argmax(1).cpu().numpy(), targets.cpu().numpy())
        if samples == 0:
            raise ValueError("Evaluation loader produced no samples")
        return {"loss": total_loss / samples, "samples": samples, "metrics": metrics.compute()}
    finally:
        model.train(previous)


def _json_exclusive(path: Path, value: dict):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _validate_output_dir(output_dir: Path):
    root = Path(__file__).resolve().parents[3]
    protected = [root / name for name in ("Data", "test_image", ".git", "backend", "frontend", "tests", "docs", "config")]
    protected.append(Path(torch.hub.get_dir()))
    storage, candidates = root / "storage", root / "storage/models/segmentation_candidates"
    if output_dir.is_relative_to(storage.resolve()) and not output_dir.is_relative_to(candidates.resolve()):
        raise ValueError("Production storage is protected; use a new segmentation_candidates version directory")
    if any(output_dir.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Output directory is inside protected data/source/cache scope")
    if output_dir.exists():
        raise FileExistsError("Output version directory already exists; refusing to overwrite any artifact")


def save_segmentation_checkpoint(path: Path, model: ResNet18UNet, *, config: SegmentationBaselineConfig,
                                 epoch: int, validation_loss: float, dataset_fingerprint: str, synthetic: bool,
                                 optimizer=None, scaler=None):
    if (model.taxonomy != config.pipeline.taxonomy or model.decoder_channels != config.decoder_channels
            or model.freeze_encoder_batchnorm != config.freeze_encoder_batchnorm
            or config.pretrained_encoder != (model.encoder_weights_sha256 is not None)):
        raise ValueError("Checkpoint model/config mismatch")
    if type(epoch) is not int or epoch < 1 or not np.isfinite(validation_loss) or validation_loss < 0:
        raise ValueError("Invalid checkpoint epoch/loss")
    if type(synthetic) is not bool:
        raise ValueError("Synthetic provenance must be explicit")
    if re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint) is None:
        raise ValueError("Dataset fingerprint must be SHA-256")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Refusing to save nonfinite model weights")
    metadata = {"schema_version": 1, "model_version": path.parent.name,
                "architecture_version": ARCHITECTURE_VERSION, "preprocessing": PREPROCESSING,
                "taxonomy": model.taxonomy.model_dump(mode="json"), "config": config.model_dump(mode="json"),
                "epoch": epoch, "validation_loss": float(validation_loss), "dataset_fingerprint": dataset_fingerprint,
                "encoder_weights_sha256": model.encoder_weights_sha256,
                "initialization": "imagenet1k_v1" if model.encoder_weights_sha256 else "random_explicit",
                "synthetic": synthetic, "production_approved": False, "torch_version": str(torch.__version__)}
    payload = {"metadata": metadata, "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
               "scaler_state": scaler.state_dict() if scaler is not None else None}
    # Exclusive create: each improvement has a new filename, never replace best/current checkpoints.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_segmentation_checkpoint(path: Path, *, trusted_local: bool = False,
                                 expected_taxonomy: SegmentationTaxonomy | None = None, device: str = "cpu"):
    # weights_only limits deserialization, but is not a security boundary on this legacy Torch build.
    if type(trusted_local) is not bool or not trusted_local:
        raise ValueError("Only explicitly trusted local checkpoints may be loaded")
    resolved_device = resolve_training_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"metadata", "model_state"}.issubset(payload):
        raise ValueError("Invalid segmentation checkpoint structure")
    metadata = payload["metadata"]
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata.get("schema_version") != 1
            or metadata.get("architecture_version") != ARCHITECTURE_VERSION or metadata.get("preprocessing") != PREPROCESSING
            or metadata.get("production_approved") is not False or type(metadata.get("synthetic")) is not bool
            or not isinstance(metadata.get("model_version"), str) or not metadata["model_version"].strip()):
        raise ValueError("Unsupported checkpoint metadata or unsafe approval state")
    config = SegmentationBaselineConfig.model_validate(metadata["config"])
    taxonomy = SegmentationTaxonomy.model_validate(metadata["taxonomy"])
    if taxonomy != config.pipeline.taxonomy or (expected_taxonomy is not None and taxonomy != expected_taxonomy):
        raise ValueError("Checkpoint taxonomy/order mismatch")
    initialization = "imagenet1k_v1" if config.pretrained_encoder else "random_explicit"
    source_hash = metadata.get("encoder_weights_sha256")
    if (metadata.get("initialization") != initialization
            or (config.pretrained_encoder and (not isinstance(source_hash, str) or re.fullmatch(r"f37072fd[0-9a-f]{56}", source_hash) is None))
            or (not config.pretrained_encoder and source_hash is not None)
            or type(metadata.get("epoch")) is not int or metadata["epoch"] < 1
            or type(metadata.get("validation_loss")) not in {int, float}
            or not np.isfinite(metadata["validation_loss"]) or metadata["validation_loss"] < 0
            or not isinstance(metadata.get("dataset_fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata["dataset_fingerprint"]) is None):
        raise ValueError("Invalid checkpoint provenance/selection metadata")
    state = payload["model_state"]
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Invalid/nonfinite checkpoint weights")
    model = ResNet18UNet(taxonomy, pretrained=False, decoder_channels=config.decoder_channels,
                         freeze_encoder_batchnorm=config.freeze_encoder_batchnorm)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint model-state shape/key mismatch") from exc
    if not torch.equal(model.rgb_mean, torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)) or not torch.equal(model.rgb_std, torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)):
        raise ValueError("Checkpoint normalization buffers disagree with preprocessing contract")
    model.encoder_weights_sha256 = metadata.get("encoder_weights_sha256")
    return model.to(resolved_device).eval(), metadata


def train_segmentation_baseline(manifest: SegmentationManifest, config: SegmentationBaselineConfig, *,
                                output_dir: Path, base_dir: Path = Path("."), allow_synthetic: bool = False) -> dict:
    training = config.pipeline.training
    if not training.enabled:
        raise ValueError("Training is disabled; explicit training.enabled=true required")
    if manifest.taxonomy != config.pipeline.taxonomy:
        raise ValueError("Manifest/config taxonomy mismatch")
    if not config.pipeline.split.require_roll_id:
        raise ValueError("Baseline training requires explicit roll-disjoint grouping")
    if type(allow_synthetic) is not bool:
        raise ValueError("Explicit synthetic boolean required")
    device = resolve_training_device(config.device)
    output_dir = output_dir.resolve()
    _validate_output_dir(output_dir)
    split = grouped_segmentation_split(manifest, config.pipeline.split)
    if split.sample_counts["validation"] == 0:
        raise ValueError("Early stopping requires an independent validation partition")
    eligible = [sample for sample in manifest.samples if split.assignments[sample.sample_id] in {"train", "validation"}]
    synthetic = any(sample.origin == "synthetic" for sample in eligible)
    if synthetic and not (allow_synthetic and training.allow_synthetic):
        raise ValueError("Synthetic training requires BOTH config allow_synthetic=true and explicit CLI/function opt-in")
    if any(sample.origin == "real" and sample.annotation_source != "human_verified" for sample in eligible):
        raise ValueError("Real training/validation samples require human_verified annotations")
    train_classes, validation_classes = set(), set()
    for sample in eligible:
        if any(_path_inside_output(value, base_dir, output_dir) for value in (sample.image_path, sample.mask_path) if value):
            raise ValueError("Output must not contain input dataset paths")
        load_segmentation_sample(sample, manifest.taxonomy, base_dir=base_dir, allow_synthetic=synthetic)
        (train_classes if split.assignments[sample.sample_id] == "train" else validation_classes).update(sample.defect_classes)
    if train_classes != set(manifest.taxonomy.classes[1:]):
        raise ValueError("Every active foreground class must have annotated training support")
    if not validation_classes:
        raise ValueError("Validation has no annotated foreground; cannot select a segmentation baseline")
    warnings = list(split.warnings)
    if not config.pretrained_encoder:
        warnings.append("Random encoder initialization was explicitly configured; not the preferred pretrained baseline.")
    if validation_classes != set(manifest.taxonomy.classes[1:]):
        warnings.append("Some active foreground classes have no validation support; real per-class quality is unverified.")
    fingerprint = hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()
    with training_reproducibility(config.seed, config.cpu_threads):
        model = ResNet18UNet(manifest.taxonomy, pretrained=config.pretrained_encoder, decoder_channels=config.decoder_channels,
                             freeze_encoder_batchnorm=config.freeze_encoder_batchnorm).to(device)
        criterion, optimizer = baseline_loss(config), build_optimizer(model, config)
        amp = device.type == "cuda" and training.amp_if_cuda
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        datasets = {partition: SegmentationDataset(manifest, partition=partition, split_config=config.pipeline.split,
                    augmentation=config.pipeline.augmentation, output_size=training.image_size, base_dir=base_dir,
                    allow_synthetic=synthetic, seed=config.seed) for partition in ("train", "validation")}
        generator = torch.Generator().manual_seed(config.seed)
        loaders = {partition: DataLoader(dataset, batch_size=training.batch_size, shuffle=partition == "train",
                    generator=generator if partition == "train" else None, num_workers=0, drop_last=False)
                   for partition, dataset in datasets.items()}
        # All gates, source files and cached pretrained weights are verified before making a new version directory.
        output_dir.mkdir(parents=True, exist_ok=False)
        _json_exclusive(output_dir / "config.json", config.model_dump(mode="json"))
        _json_exclusive(output_dir / "manifest.json", manifest.model_dump(mode="json"))
        _json_exclusive(output_dir / "split.json", asdict(split))
        history, best, stale, best_path = [], float("inf"), 0, None
        for epoch in range(1, training.epochs + 1):
            model.train()
            datasets["train"].set_epoch(epoch)
            summed_loss, samples = 0.0, 0
            for batch in loaders["train"]:
                images, targets = batch["image"].to(device), batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; no candidate approval")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                summed_loss += float(loss.detach()) * len(images)
                samples += len(images)
            validation = evaluate_segmentation(model, loaders["validation"], criterion, device=device)
            improved = validation["loss"] < best - config.min_delta
            history.append({"epoch": epoch, "train_loss": summed_loss / samples, "validation": validation, "improved": improved})
            if improved:
                best, stale = validation["loss"], 0
                best_path = output_dir / f"best_epoch_{epoch:04d}.pt"
                save_segmentation_checkpoint(best_path, model, config=config, epoch=epoch, validation_loss=best,
                                              dataset_fingerprint=fingerprint, synthetic=synthetic, optimizer=optimizer, scaler=scaler)
            else:
                stale += 1
            if stale >= training.early_stopping_patience:
                break
        best_model, metadata = load_segmentation_checkpoint(best_path, trusted_local=True, expected_taxonomy=manifest.taxonomy, device=str(device))
        best_validation = evaluate_segmentation(best_model, loaders["validation"], criterion, device=device)
        report = {"phase": 8, "status": "SYNTHETIC_SOFTWARE_CHECK_ONLY" if synthetic else "RESEARCH_CANDIDATE_UNAPPROVED",
                  "model_version": metadata["model_version"],
                  "architecture": ARCHITECTURE_VERSION, "preprocessing": PREPROCESSING, "device": str(device), "amp_enabled": amp,
                  "dataset_version": manifest.dataset_version, "dataset_fingerprint": fingerprint,
                  "encoder_initialization": metadata["initialization"], "encoder_weights_sha256": model.encoder_weights_sha256,
                  "epochs_completed": len(history), "early_stopped": len(history) < training.epochs,
                  "best_epoch": metadata["epoch"], "best_checkpoint": best_path.name, "history": history,
                  "validation": best_validation, "split_counts": split.sample_counts, "warnings": warnings,
                  "trained_model": True, "contains_synthetic_train_or_validation": synthetic,
                  "test_partition_evaluated": False, "production_accuracy_evaluated": False, "production_approved": False,
                  "registry_modified": False, "note": "Validation selected the checkpoint, not a fresh acceptance test. Confidence is uncalibrated. No QC integration."}
        _json_exclusive(output_dir / "report.json", report)
        return report


def _path_inside_output(value: str, base_dir: Path, output_dir: Path) -> bool:
    path = Path(value)
    return (path if path.is_absolute() else base_dir / path).resolve().is_relative_to(output_dir)
