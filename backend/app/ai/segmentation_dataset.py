"""Verified image/mask pairs and deterministic paired transforms; no trainer/model."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from backend.app.ai.segmentation_split import grouped_segmentation_split
from backend.app.core.segmentation_data import (GroupedSplitConfig, SegmentationAugmentationConfig, SegmentationManifest,
                                                SegmentationSample, SegmentationTaxonomy)


@dataclass(frozen=True)
class SegmentationPair:
    image: np.ndarray  # Oriented uint8 HWC RGB, original resolution.
    mask: np.ndarray  # int64 HW class ids, optional explicit ignore_index.
    sample_id: str


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_segmentation_sample(sample: SegmentationSample, taxonomy: SegmentationTaxonomy, *,
                              base_dir: Path = Path("."), allow_synthetic: bool = False) -> SegmentationPair:
    if type(allow_synthetic) is not bool:
        raise ValueError("Synthetic opt-in must be an explicit boolean")
    if sample.origin == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic samples require explicit opt-in; never treat them as real annotation")
    if not set(sample.defect_classes).issubset(set(taxonomy.classes[1:])):
        raise ValueError("Inactive/unsupported sample labels")
    if sample.binary_class_name is not None and sample.binary_class_name not in taxonomy.classes[1:]:
        raise ValueError("Inactive binary class mapping")
    image_path = _path(sample.image_path, base_dir)
    if sha256_file(image_path) != sample.image_sha256:
        raise ValueError("Image checksum mismatch")
    try:
        with Image.open(image_path) as source:
            source.verify()
        with Image.open(image_path) as source:
            raw_size = source.size
            orientation = source.getexif().get(274, 1)
            if type(orientation) is not int or orientation not in range(1, 9):
                raise ValueError("Unsupported image EXIF orientation")
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
        if sample.mask_path is None:
            mask = np.zeros(image.shape[:2], dtype=np.int64)  # Only Good permits no annotation mask.
        else:
            mask_path = _path(sample.mask_path, base_dir)
            if sha256_file(mask_path) != sample.mask_sha256:
                raise ValueError("Mask checksum mismatch")
            with Image.open(mask_path) as source:
                source.verify()
            with Image.open(mask_path) as source:
                if source.mode not in {"1", "L", "P", "I", "I;16"} or source.getexif().get(274, 1) != 1:
                    raise ValueError("Mask must be explicit single-channel/palette integer ids with no independent EXIF transform")
                expected = raw_size if sample.mask_coordinates == "raw_image" else (image.shape[1], image.shape[0])
                if source.size != expected:
                    raise ValueError("Image/mask geometry mismatch; do not silently resize an invalid annotation")
                mask_image = source.copy()
            if sample.mask_coordinates == "raw_image" and orientation != 1:
                method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                          4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                          6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}[orientation]
                mask_image = mask_image.transpose(method)
            mask = np.asarray(mask_image).astype(np.int64)
            if sample.mask_encoding == "binary_255":
                if sample.binary_class_name not in taxonomy.classes[1:] or not set(np.unique(mask)).issubset({0, 255}):
                    raise ValueError("binary_255 requires exactly 0/255 pixels and an active class mapping")
                mask = np.where(mask == 255, taxonomy.classes.index(sample.binary_class_name), 0).astype(np.int64)
        valid = mask != taxonomy.ignore_index if taxonomy.ignore_index is not None else np.ones(mask.shape, dtype=bool)
        if not valid.any() or np.any(mask[valid] < 0) or np.any(mask[valid] >= len(taxonomy.classes)):
            raise ValueError("Mask contains invalid/inactive labels or no valid annotated pixels")
        present = {taxonomy.classes[index] for index in np.unique(mask[valid]) if index != 0}
        if present != set(sample.defect_classes) or (sample.label == "normal" and present):
            raise ValueError("Declared image labels/classes disagree with mask foreground")
        if sample.label == "defect" and not present:
            raise ValueError("Defect sample must not have an empty mask")
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Unreadable segmentation image/mask") from exc
    return SegmentationPair(image, np.array(mask, dtype=np.int64, copy=True), sample.sample_id)


def paired_segmentation_transform(pair: SegmentationPair, config: SegmentationAugmentationConfig, *,
                                   seed: int, output_size: tuple[int, int] | None = None) -> SegmentationPair:
    image, mask = pair.image.copy(), pair.mask.copy()
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or mask.ndim != 2 or image.shape[:2] != mask.shape
            or image.size == 0 or mask.dtype.kind not in "iu" or np.any(mask < 0) or np.any(mask > np.iinfo(np.int32).max)):
        raise ValueError("Aligned HWC uint8 RGB image and HW class mask required")
    rng = np.random.default_rng(seed)
    if config.enabled:
        if rng.random() < config.horizontal_flip_probability:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if rng.random() < config.vertical_flip_probability:
            image, mask = image[::-1], mask[::-1]
        if config.rotate_quarter_turns:
            turns = int(rng.integers(0, 4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    if output_size is not None:
        if len(output_size) != 2 or any(type(size) is not int or not 1 <= size <= 4096 for size in output_size):
            raise ValueError("Output size must be positive bounded integer (height,width)")
        dimensions = (output_size[1], output_size[0])
        image = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize(dimensions, Image.Resampling.BILINEAR)).copy()
        mask = np.asarray(Image.fromarray(np.ascontiguousarray(mask).astype(np.int32), mode="I").resize(dimensions, Image.Resampling.NEAREST)).astype(np.int64)
    if config.enabled:
        values = image.astype(np.float64)
        contrast = rng.uniform(*config.contrast_range)
        center = values.mean(axis=(0, 1), keepdims=True)
        values = (values - center) * contrast + center
        values *= rng.uniform(1 - config.brightness_delta, 1 + config.brightness_delta)
        image = np.clip(np.round(values), 0, 255).astype(np.uint8)
    return SegmentationPair(np.ascontiguousarray(image), np.ascontiguousarray(mask, dtype=np.int64), pair.sample_id)


class SegmentationDataset(Dataset):
    def __init__(self, manifest: SegmentationManifest, *, partition: str,
                 split_config: GroupedSplitConfig | None = None, augmentation: SegmentationAugmentationConfig | None = None,
                 output_size: tuple[int, int] = (256, 256), base_dir: Path = Path("."), allow_synthetic: bool = False, seed: int = 42):
        if partition not in {"train", "validation", "test"}:
            raise ValueError("Historical/excluded data must not be loaded as fresh training/evaluation")
        if type(allow_synthetic) is not bool or type(seed) is not int or not 0 <= seed <= 2**32-1:
            raise ValueError("Explicit synthetic boolean and nonnegative integer seed required")
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2 or any(type(value) is not int or not 1 <= value <= 4096 for value in output_size):
            raise ValueError("Dataset output size must be a bounded integer height/width pair")
        self.split = grouped_segmentation_split(manifest, split_config or GroupedSplitConfig())
        self.samples = tuple(sample for sample in manifest.samples if self.split.assignments[sample.sample_id] == partition)
        if not self.samples:
            raise ValueError("Requested partition is empty")
        if any(sample.origin == "synthetic" for sample in self.samples) and not allow_synthetic:
            raise ValueError("Synthetic dataset rows require explicit opt-in")
        self.manifest, self.partition, self.base_dir = manifest, partition, base_dir
        self.augmentation = augmentation or SegmentationAugmentationConfig()
        self.output_size, self.allow_synthetic, self.seed, self.epoch = output_size, allow_synthetic, seed, 0

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        sample = self.samples[index]
        pair = load_segmentation_sample(sample, self.manifest.taxonomy, base_dir=self.base_dir, allow_synthetic=self.allow_synthetic)
        seed_bytes = hashlib.sha256(f"{self.seed}|{self.epoch if self.partition == 'train' else 0}|{sample.sample_id}".encode()).digest()
        config = self.augmentation if self.partition == "train" else SegmentationAugmentationConfig()
        transformed = paired_segmentation_transform(pair, config, seed=int.from_bytes(seed_bytes[:8], "big"), output_size=self.output_size)
        image = torch.from_numpy(transformed.image.transpose(2, 0, 1).copy()).float() / 255
        return {"image": image, "mask": torch.from_numpy(transformed.mask.copy()).long(), "sample_id": sample.sample_id}
