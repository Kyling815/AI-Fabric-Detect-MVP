"""Supervised dataset contracts only. No model, training or production integration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFECT_CLASSES = ("background", "stripe", "scratch", "missing_fabric", "wrong_color", "wrinkle", "spot", "stain", "defect")


class SegmentationTaxonomy(BaseModel):
    classes: tuple[str, ...] = ("background", "defect")
    ignore_index: int | None = Field(default=None, ge=0, le=65535, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def valid_subset(self):
        if (len(self.classes) < 2 or self.classes[0] != "background" or len(set(self.classes)) != len(self.classes)
                or any(name not in DEFECT_CLASSES for name in self.classes)):
            raise ValueError("Taxonomy must be a unique supported subset beginning with background, plus a defect class")
        if self.ignore_index is not None and self.ignore_index < len(self.classes):
            raise ValueError("ignore_index must not collide with active class ids")
        return self


class SegmentationSample(BaseModel):
    sample_id: str = Field(min_length=1)
    fabric_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_path: str | None = Field(default=None, min_length=1)
    mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    label: Literal["normal", "defect"]
    defect_classes: tuple[str, ...] = ()
    mask_encoding: Literal["class_ids", "binary_255"] = "class_ids"
    binary_class_name: str | None = None
    mask_coordinates: Literal["raw_image", "oriented_image"] = "oriented_image"
    origin: Literal["real", "synthetic"]
    annotation_source: Literal["human_verified", "existing_binary", "synthetic_fixture"]
    source_id: str = Field(min_length=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ancestor_source_ids: tuple[str, ...] = ()
    ancestor_source_sha256s: tuple[str, ...] = ()
    roll_id: str | None = None
    lot_id: str | None = None
    camera_id: str | None = None
    split: Literal["unassigned", "train", "validation", "test", "historical_locked", "excluded"] = "unassigned"
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("sample_id", "fabric_id", "source_id", "roll_id", "lot_id", "camera_id")
    @classmethod
    def clean_identifier(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("Identifiers must be nonblank with no surrounding whitespace")
        return value

    @model_validator(mode="after")
    def consistent_annotation(self):
        if self.annotation_source == "synthetic_fixture" and self.origin != "synthetic":
            raise ValueError("Synthetic fixture annotations must never be declared real")
        if (self.mask_path is None) != (self.mask_sha256 is None):
            raise ValueError("Mask path and SHA-256 must be supplied together")
        if self.label == "normal" and self.defect_classes:
            raise ValueError("Good image cannot declare defect classes")
        if self.label == "defect" and (self.mask_path is None or not self.defect_classes):
            raise ValueError("Defect image requires an explicit mask and class declarations")
        if len(set(self.defect_classes)) != len(self.defect_classes):
            raise ValueError("Duplicate defect classes")
        if self.mask_encoding == "binary_255":
            if self.binary_class_name is None or self.binary_class_name == "background":
                raise ValueError("binary_255 needs an explicit foreground class mapping")
        elif self.binary_class_name is not None:
            raise ValueError("Class-id mask must not declare a binary mapping")
        if any(not value.strip() or value != value.strip() for value in self.ancestor_source_ids):
            raise ValueError("Ancestor source ids must be nonblank")
        import re
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.ancestor_source_sha256s):
            raise ValueError("Invalid ancestor source hash")
        return self


class SegmentationManifest(BaseModel):
    schema_version: Literal[1] = 1
    dataset_version: str = Field(min_length=1)
    taxonomy: SegmentationTaxonomy = Field(default_factory=SegmentationTaxonomy)
    samples: tuple[SegmentationSample, ...] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported segmentation manifest version")
        return value

    @model_validator(mode="after")
    def coherent_classes(self):
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample ids")
        active = set(self.taxonomy.classes[1:])
        for sample in self.samples:
            if not set(sample.defect_classes).issubset(active):
                raise ValueError("Sample declares unsupported/inactive defect label")
            if sample.binary_class_name is not None and sample.binary_class_name not in active:
                raise ValueError("Binary mask mapping must name an active foreground class")
        return self


class GroupedSplitConfig(BaseModel):
    train_ratio: float = Field(default=0.7, ge=0, le=1)
    validation_ratio: float = Field(default=0.15, ge=0, le=1)
    test_ratio: float = Field(default=0.15, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    require_roll_id: bool = Field(default=True, strict=True)
    group_by_lot: bool = Field(default=False, strict=True)
    group_by_camera: bool = Field(default=False, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("train_ratio", "validation_ratio", "test_ratio", mode="before")
    @classmethod
    def numeric_ratios(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("Split ratios must be numeric")
        return value

    @model_validator(mode="after")
    def valid_ratios(self):
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1) > 1e-9:
            raise ValueError("Split ratios must sum to 1")
        return self


class SegmentationAugmentationConfig(BaseModel):
    enabled: bool = Field(default=False, strict=True)
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0, ge=0, le=1)
    rotate_quarter_turns: bool = Field(default=True, strict=True)
    brightness_delta: float = Field(default=0.1, ge=0, le=1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_contrast(self):
        if not 0 < self.contrast_range[0] <= self.contrast_range[1] <= 3:
            raise ValueError("Contrast range must be positive, ordered, <=3")
        return self


class SegmentationTrainingConfig(BaseModel):
    enabled: bool = Field(default=False, strict=True)
    architecture: Literal["unet_resnet18"] = "unet_resnet18"  # Specification for a later approved phase, not implemented here.
    share_anomaly_encoder: Literal[False] = False
    image_size: tuple[int, int] = (256, 256)  # height,width
    batch_size: int = Field(default=4, ge=1, strict=True)
    epochs: int = Field(default=30, ge=1, strict=True)
    encoder_lr: float = Field(default=3e-5, gt=0)
    decoder_lr: float = Field(default=3e-4, gt=0)
    optimizer: Literal["adamw"] = "adamw"
    weight_decay: float = Field(default=0.01, ge=0)
    loss: Literal["cross_entropy_dice"] = "cross_entropy_dice"
    focal_optional: bool = Field(default=False, strict=True)
    early_stopping_patience: int = Field(default=8, ge=1, strict=True)
    amp_if_cuda: bool = Field(default=True, strict=True)
    allow_synthetic: bool = Field(default=False, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("image_size", mode="before")
    @classmethod
    def strict_dimensions(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
            raise ValueError("Image size must be an integer pair")
        return value

    @field_validator("share_anomaly_encoder", mode="before")
    @classmethod
    def no_encoder_sharing(cls, value):
        if type(value) is not bool or value:
            raise ValueError("Supervised and anomaly encoders must remain independent")
        return value

    @model_validator(mode="after")
    def valid_size(self):
        if any(type(value) is not int or not 1 <= value <= 4096 for value in self.image_size):
            raise ValueError("image_size must be positive integer height/width <=4096")
        return self


class SegmentationPipelineConfig(BaseModel):
    taxonomy: SegmentationTaxonomy = Field(default_factory=SegmentationTaxonomy)
    split: GroupedSplitConfig = Field(default_factory=GroupedSplitConfig)
    augmentation: SegmentationAugmentationConfig = Field(default_factory=SegmentationAugmentationConfig)
    training: SegmentationTrainingConfig = Field(default_factory=SegmentationTrainingConfig)
    model_config = ConfigDict(extra="forbid", frozen=True)
