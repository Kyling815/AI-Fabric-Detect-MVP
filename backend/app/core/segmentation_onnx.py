"""Fixed-shape, offline supervised export contracts; never registry activation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.segmentation_data import SegmentationTaxonomy


INPUT_NAME = "images"
OUTPUT_NAMES = ("logits", "probabilities", "mask", "confidence")


class OnnxExportConfig(BaseModel):
    input_shape: tuple[int, int, int, int] = (1, 3, 64, 64)
    opset_version: Literal[17] = 17
    atol: float = Field(default=1e-4, gt=0, le=1e-3)
    rtol: float = Field(default=1e-4, ge=0, le=1e-3)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("input_shape", mode="before")
    @classmethod
    def bounded_nchw(cls, value):
        if (not isinstance(value, (list, tuple)) or len(value) != 4
                or any(type(item) is not int for item in value)
                or not 1 <= value[0] <= 4 or value[1] != 3
                or not all(32 <= item <= 512 for item in value[2:])):
            raise ValueError("Fixed NCHW RGB shape required: batch 1..4, H/W 32..512")
        return tuple(value)

    @field_validator("opset_version", mode="before")
    @classmethod
    def strict_opset(cls, value):
        if type(value) is not int or value != 17:
            raise ValueError("Only explicitly tested opset 17 is supported")
        return value

    @field_validator("atol", "rtol", mode="before")
    @classmethod
    def numeric_tolerance(cls, value):
        if isinstance(value, (str, bool)):
            raise ValueError("Numerical tolerance required")
        return value


class OnnxManifest(BaseModel):
    schema_version: Literal[1] = 1
    artifact_type: Literal["offline_supervised_onnx"] = "offline_supervised_onnx"
    model_filename: Literal["model.onnx"] = "model.onnx"
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_model_version: str = Field(min_length=1, max_length=256)
    synthetic: bool = Field(strict=True)
    production_approved: bool = Field(default=False, strict=True)
    architecture: Literal["independent_resnet18_unet_v1"] = "independent_resnet18_unet_v1"
    preprocessing: Literal["rgb_float_0_1_imagenet_mean_std_in_model_v1"] = "rgb_float_0_1_imagenet_mean_std_in_model_v1"
    taxonomy: SegmentationTaxonomy
    config: OnnxExportConfig
    input_name: Literal["images"] = INPUT_NAME
    output_names: tuple[Literal["logits", "probabilities", "mask", "confidence"], ...] = OUTPUT_NAMES
    input_dtype: Literal["float32"] = "float32"
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    exporter: Literal["torch.onnx.export_dynamo_false"] = "torch.onnx.export_dynamo_false"
    torch_version: str
    onnx_version: str
    onnxruntime_version: str
    parity_passed: bool = Field(strict=True)
    parity_case_count: int = Field(ge=7, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Only integer manifest schema 1 is supported")
        return value

    @model_validator(mode="after")
    def offline_only(self):
        if self.production_approved or not self.parity_passed or self.output_names != OUTPUT_NAMES:
            raise ValueError("Only parity-verified offline artifacts with the exact output contract are accepted")
        if self.source_model_version != self.source_model_version.strip():
            raise ValueError("Clean source version required")
        return self
