"""One disposable CPU worker, stdin/stdout JSON only. Never API/DB/training imports."""

import hashlib
import json
from pathlib import Path
import sys

from backend.app.core.segmentation_shadow import ShadowClassSummary, SupervisedShadowConfig, SupervisedShadowObservation


def observe_checkpoint(image_path: Path, fabric_id: str, config: SupervisedShadowConfig) -> SupervisedShadowObservation:
    def failure(reason):
        return SupervisedShadowObservation(status="unavailable", reason=reason, requested_model_version=config.expected_model_version)
    if not config.trusted_local:
        return failure("checkpoint_untrusted")
    if fabric_id not in config.allowed_fabric_ids:
        return failure("fabric_not_allowed")
    path = Path(config.checkpoint_path)
    if not config.checkpoint_path or not path.is_file():
        return failure("checkpoint_missing")
    if path.stat().st_size > config.max_checkpoint_bytes:
        return failure("checkpoint_too_large")
    # Verify the exact SAME open file that is deserialized, not a second path open susceptible to substitution.
    try:
        import torch
        from PIL import Image, ImageOps
        import numpy as np
        from backend.app.ai.segmentation_trainer import load_segmentation_checkpoint
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != config.checkpoint_sha256:
                return failure("checkpoint_hash_mismatch")
            stream.seek(0)
            model, metadata = load_segmentation_checkpoint(stream, trusted_local=True, device="cpu")
        if metadata["model_version"] != config.expected_model_version:
            return failure("version_mismatch")
        if metadata["synthetic"] and not config.allow_synthetic:
            return failure("synthetic_not_allowed")
        size = tuple(metadata["config"]["pipeline"]["training"]["image_size"])
        if max(size) > 512:
            return failure("unsupported_image_size")
    except Exception:
        return failure("checkpoint_invalid")
    try:
        with Image.open(image_path) as source:
            source.verify()
        with Image.open(image_path) as source:
            orientation = source.getexif().get(274, 1)
            if type(orientation) is not int or orientation not in range(1, 9):
                raise ValueError("Unsupported image EXIF orientation")
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            original_shape = (oriented.height, oriented.width)
            resized = oriented.resize((size[1], size[0]), Image.Resampling.BILINEAR)
            values = np.asarray(resized).copy()
        image = torch.from_numpy(values.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
    except Exception:
        return SupervisedShadowObservation(status="error", reason="image_decode_failed", requested_model_version=config.expected_model_version)
    try:
        prediction = model.predict(image)
        probability = prediction.probabilities[0].cpu().numpy()
        mask = prediction.mask[0].cpu().numpy().astype("<i4")
        confidence = prediction.confidence[0].cpu().numpy()
        classes = tuple(ShadowClassSummary(name=name, predicted_pixels=int((mask == index).sum()),
                            mean_class_probability=float(probability[index].mean()),
                            mean_winning_confidence=float(confidence[mask == index].mean()) if (mask == index).any() else None)
                        for index, name in enumerate(model.taxonomy.classes))
        flat = mask.reshape(-1)
        boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [flat.size]))
        runs = tuple((int(flat[start]), int(end - start)) for start, end in zip(starts, ends)) if len(starts) <= config.max_mask_runs else None
        return SupervisedShadowObservation(status="observed", reason="shadow_only", requested_model_version=config.expected_model_version,
                    model_version=metadata["model_version"], checkpoint_sha256=digest, synthetic_checkpoint=metadata["synthetic"],
                    original_shape=original_shape, mask_shape=tuple(mask.shape), taxonomy=model.taxonomy.classes, classes=classes,
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), mask_rle=runs, mask_rle_omitted=runs is None)
    except Exception:
        return SupervisedShadowObservation(status="error", reason="prediction_failed", requested_model_version=config.expected_model_version)


def main():
    try:
        request = json.loads(sys.stdin.read(64 * 1024))
        config = SupervisedShadowConfig.model_validate(request["config"])
        result = observe_checkpoint(Path(request["image_path"]), request["fabric_id"], config)
    except Exception:
        result = SupervisedShadowObservation(status="error", reason="worker_request_invalid")
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
