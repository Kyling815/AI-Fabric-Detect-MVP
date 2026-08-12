"""Typed, reproducible configuration for preprocessing v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PreprocessConfig:
    preprocessing_version: str = "v2"
    min_width: int = 512
    min_height: int = 512
    blur_threshold: float = 70.0
    max_dark_ratio: float = 0.20
    max_bright_ratio: float = 0.20
    dark_pixel_threshold: int = 20
    bright_pixel_threshold: int = 235

    roi_mode: str = "none"
    fixed_roi: Mapping[str, float | int] | None = None
    automatic_roi_min_area_ratio: float = 0.50
    automatic_roi_margin_ratio: float = 0.02

    apply_orientation_normalization: bool = False
    orientation_confidence_threshold: float = 0.60
    maximum_rotation_degrees: float = 15.0
    expected_orientation_degrees: float = 0.0

    apply_illumination_correction: bool = True
    illumination_sigma_ratio: float = 0.08

    apply_color_normalization: bool = False
    luminance_strength: float = 1.0
    chromatic_strength: float = 0.25
    lab_reference_mean: tuple[float, float, float] | None = None
    lab_reference_std: tuple[float, float, float] | None = None

    apply_denoising: bool = True
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 20.0
    bilateral_sigma_space: float = 5.0

    apply_clahe: bool = True
    clahe_clip_limit: float = 1.5
    clahe_grid_size: tuple[int, int] = (8, 8)

    apply_sharpening: bool = False
    sharpen_alpha: float = 0.25
    sharpen_sigma: float = 1.0

    target_max_size: int | None = 1024
    letterbox_size: tuple[int, int] | None = None
    patch_sizes: tuple[int, ...] = (224,)
    patch_overlap_ratio: float = 0.50
    minimum_fabric_coverage: float = 0.0

    generate_texture_view: bool = False
    calculate_fft_features: bool = False
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.preprocessing_version != "v2":
            raise ValueError("PreprocessConfig.preprocessing_version must be 'v2'.")
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("Minimum image dimensions must be positive.")
        if self.blur_threshold < 0:
            raise ValueError("blur_threshold cannot be negative.")
        for name in (
            "max_dark_ratio",
            "max_bright_ratio",
            "automatic_roi_min_area_ratio",
            "automatic_roi_margin_ratio",
            "orientation_confidence_threshold",
            "luminance_strength",
            "chromatic_strength",
            "patch_overlap_ratio",
            "minimum_fabric_coverage",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.roi_mode not in {"none", "fixed", "automatic"}:
            raise ValueError("roi_mode must be 'none', 'fixed', or 'automatic'.")
        if self.roi_mode == "fixed" and not self.fixed_roi:
            raise ValueError("fixed_roi is required when roi_mode is 'fixed'.")
        if not 0 < self.illumination_sigma_ratio <= 0.5:
            raise ValueError("illumination_sigma_ratio must be in (0, 0.5].")
        if self.maximum_rotation_degrees < 0 or self.maximum_rotation_degrees > 45:
            raise ValueError("maximum_rotation_degrees must be between 0 and 45.")
        if self.bilateral_diameter < 1 or self.bilateral_diameter % 2 == 0:
            raise ValueError("bilateral_diameter must be a positive odd integer.")
        if self.bilateral_sigma_color <= 0 or self.bilateral_sigma_space <= 0:
            raise ValueError("Bilateral sigmas must be positive.")
        if self.clahe_clip_limit <= 0 or any(value <= 0 for value in self.clahe_grid_size):
            raise ValueError("CLAHE settings must be positive.")
        if self.sharpen_alpha < 0 or self.sharpen_sigma <= 0:
            raise ValueError("Sharpen settings must be non-negative/positive.")
        if self.target_max_size is not None and self.target_max_size <= 0:
            raise ValueError("target_max_size must be positive when provided.")
        if self.letterbox_size is not None and any(value <= 0 for value in self.letterbox_size):
            raise ValueError("letterbox_size dimensions must be positive.")
        if not self.patch_sizes or any(size <= 0 for size in self.patch_sizes):
            raise ValueError("patch_sizes must contain positive integers.")
        if self.lab_reference_mean is not None and len(self.lab_reference_mean) != 3:
            raise ValueError("lab_reference_mean must contain three values.")
        if self.lab_reference_std is not None:
            if len(self.lab_reference_std) != 3 or any(value <= 0 for value in self.lab_reference_std):
                raise ValueError("lab_reference_std must contain three positive values.")
        if self.apply_color_normalization and (
            self.lab_reference_mean is None or self.lab_reference_std is None
        ):
            raise ValueError("Color normalization requires Lab reference mean and std.")

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PreprocessConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown preprocessing configuration fields: {', '.join(unknown)}")
        normalized = dict(values)
        for key in ("patch_sizes", "clahe_grid_size", "letterbox_size", "lab_reference_mean", "lab_reference_std"):
            if normalized.get(key) is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)

    @classmethod
    def load(cls, path: Path) -> "PreprocessConfig":
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing config not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML configuration; use JSON otherwise.") from exc
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Configuration must be JSON or YAML.")
        if not isinstance(payload, dict):
            raise ValueError("Preprocessing configuration must be an object.")
        return cls.from_mapping(payload)

    def with_overrides(self, values: Mapping[str, Any]) -> "PreprocessConfig":
        merged = self.to_dict()
        merged.update(values)
        return self.from_mapping(merged)


def load_registry_overrides(
    registry_path: Path | None,
    fabric_type: str | None,
) -> dict[str, Any]:
    """Read fabric preprocessing settings without rewriting the registry.

    Both a top-level ``fabric_preprocessing`` object and legacy direct
    fabric-name objects are supported. Existing ``references`` records remain
    untouched and are ignored for configuration purposes.
    """

    if registry_path is None or not registry_path.exists() or not fabric_type:
        return {}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed reference registry JSON: {registry_path}") from exc
    if not isinstance(registry, dict):
        raise ValueError("Reference registry root must be an object.")
    settings = registry.get("fabric_preprocessing", {})
    record = settings.get(fabric_type) if isinstance(settings, dict) else None
    if record is None:
        candidate = registry.get(fabric_type)
        record = candidate if isinstance(candidate, dict) else None
    if record is None:
        return {}
    return normalize_registry_overrides(record)


def normalize_registry_overrides(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(PreprocessConfig)}
    aliases = {
        "orientation_normalization": "apply_orientation_normalization",
    }
    overrides: dict[str, Any] = {}
    for key, value in record.items():
        normalized_key = aliases.get(key, key)
        if normalized_key in allowed:
            overrides[normalized_key] = value
    color = record.get("color_normalization")
    if color is not None:
        if not isinstance(color, Mapping):
            raise ValueError("color_normalization must be an object.")
        color_aliases = {
            "enabled": "apply_color_normalization",
            "luminance_strength": "luminance_strength",
            "chromatic_strength": "chromatic_strength",
            "lab_reference_mean": "lab_reference_mean",
            "lab_reference_std": "lab_reference_std",
        }
        for key, target in color_aliases.items():
            if key in color:
                overrides[target] = color[key]
    if overrides.get("preprocessing_version", "v2") != "v2":
        raise ValueError("Fabric registry override is not compatible with preprocessing v2.")
    overrides["preprocessing_version"] = "v2"
    return overrides


ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "P0": {
        "apply_illumination_correction": False,
        "apply_denoising": False,
        "apply_clahe": False,
    },
    "P1": {"apply_denoising": False, "apply_clahe": False},
    "P2": {"apply_denoising": True, "apply_clahe": False},
    "P3": {},
    "P4": {"apply_orientation_normalization": True},
    "P5": {"patch_sizes": (128, 224, 384)},
    "P6": {"patch_sizes": (128, 224, 384), "generate_texture_view": True},
    "P7": {
        "patch_sizes": (128, 224, 384),
        "generate_texture_view": True,
        "calculate_fft_features": True,
    },
}


def config_for_ablation(variant: str, base: PreprocessConfig | None = None) -> PreprocessConfig:
    name = variant.upper()
    if name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return (base or PreprocessConfig()).with_overrides(ABLATION_VARIANTS[name])


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value
