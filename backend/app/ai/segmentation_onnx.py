"""Opt-in CPU ONNX export/reload/parity of the independent supervised CNN only.

No HTTP, registry, FAISS, fusion, QC, training, dynamic axes or GPU conversion.
Pinned trusted local checkpoints/exports ONLY: legacy torch.load is not a
security boundary. Exported graphs lack Python input guards; use this runner.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import warnings

import numpy as np
import torch
from torch import nn

from backend.app.ai.segmentation_trainer import (_validate_output_dir as validate_candidate_output_dir,
                                                load_segmentation_checkpoint, training_reproducibility)
from backend.app.core.segmentation_onnx import INPUT_NAME, OUTPUT_NAMES, OnnxExportConfig, OnnxManifest


MAX_MODEL_BYTES = 512 * 1024 * 1024


class OnnxDependencyError(RuntimeError):
    pass


class OnnxParityError(ValueError):
    pass


def onnx_dependencies():
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise OnnxDependencyError("Install optional offline tools: python -m pip install -r requirements-onnx.txt") from exc
    return onnx, onnxruntime


def _check_pin(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Explicit lowercase SHA-256 pin required")


def _read_pinned(path, expected, *, limit):
    _check_pin(expected)
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Artifact exceeds size bound")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Artifact SHA-256 mismatch")
    return data


def validate_onnx_input(images, config):
    if (not isinstance(images, np.ndarray) or images.dtype != np.float32
            or images.shape != config.input_shape or not np.isfinite(images).all()
            or images.min() < 0 or images.max() > 1):
        raise ValueError("Exact fixed-shape NCHW float32 RGB input in [0,1] required; no resizing or dtype coercion")
    return np.ascontiguousarray(images)


class ExportPrediction(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        logits = self.model(images).float()
        probabilities = logits.softmax(dim=1)
        confidence, mask = probabilities.max(dim=1)
        return logits, probabilities, mask, confidence


def parity_inputs(shape):
    """Deterministic SOFTWARE probes, never real-data accuracy or a locked test."""
    rows, cols = np.indices(shape[-2:])
    def rgb(value):
        return np.broadcast_to(value, shape).astype(np.float32).copy()
    checker = ((rows // 5 + cols // 7) % 2).astype(np.float32)
    scratch = np.full(shape[-2:], 0.2, dtype=np.float32)
    scratch[:, shape[-1] // 2:shape[-1] // 2 + 2] = 0.9
    color = np.empty(shape, dtype=np.float32)
    color[:, 0], color[:, 1], color[:, 2] = 0.1, 0.5, 0.9
    return {"black": np.zeros(shape, dtype=np.float32), "white": np.ones(shape, dtype=np.float32),
            "uniform": np.full(shape, 0.5, dtype=np.float32), "checker": rgb(checker),
            "scratch": rgb(scratch), "color_channels": color,
            "seeded_noise": np.random.default_rng(16).random(shape, dtype=np.float32)}


def _output_shapes(config, classes):
    batch, _, height, width = config.input_shape
    return ((batch, classes, height, width), (batch, classes, height, width),
            (batch, height, width), (batch, height, width))


def _check_outputs(values, config, classes):
    if len(values) != 4:
        raise ValueError("Wrong ONNX output count")
    for index, (value, shape) in enumerate(zip(values, _output_shapes(config, classes))):
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != (np.int64 if index == 2 else np.float32) or not np.isfinite(value).all()):
            raise ValueError("Invalid ONNX output shape, dtype or finite values")
    logits, probabilities, mask, confidence = values
    if (probabilities.min() < 0 or probabilities.max() > 1
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)
            or not np.array_equal(mask, probabilities.argmax(1))
            or not np.allclose(confidence, probabilities.max(1), atol=1e-6, rtol=0)):
        raise ValueError("ONNX prediction output contract violated")


def compare_predictions(expected, actual, config):
    classes = expected[0].shape[1]
    _check_outputs(expected, config, classes)
    _check_outputs(actual, config, classes)
    metrics = {}
    for name, left, right in zip(OUTPUT_NAMES, expected, actual):
        if name == "mask":
            mismatches = int(np.count_nonzero(left != right))
            metrics[name] = {"mismatched_pixels": mismatches, "exact": mismatches == 0}
            if mismatches:
                raise OnnxParityError("Mask parity failed; argmax flips are not silently accepted")
        else:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            passed = bool(np.allclose(right, left, atol=config.atol, rtol=config.rtol))
            metrics[name] = {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()), "allclose": passed}
            if not passed:
                raise OnnxParityError("Numerical parity failed for " + name)
    return metrics


def _session(model_bytes, config, classes):
    onnx, ort = onnx_dependencies()
    graph = onnx.load_model_from_string(model_bytes)
    if any(item.data_location == onnx.TensorProto.EXTERNAL or item.external_data for item in graph.graph.initializer):
        raise ValueError("External tensor data is not supported")
    if ([(item.domain, item.version) for item in graph.opset_import] != [("", config.opset_version)]
            or any(node.domain not in {"", "ai.onnx"} for node in graph.graph.node)):
        raise ValueError("Only standard-domain opset 17 graphs are supported")
    onnx.checker.check_model(graph, full_check=True)
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != INPUT_NAME:
        raise ValueError("Unexpected ONNX input contract")
    if tuple(item.name for item in graph.graph.output) != OUTPUT_NAMES:
        raise ValueError("Unexpected ONNX output names")
    shape = graph.graph.input[0].type.tensor_type
    if (shape.elem_type != onnx.TensorProto.FLOAT
            or tuple(item.dim_value for item in shape.shape.dim) != config.input_shape
            or any(item.dim_param for item in shape.shape.dim)):
        raise ValueError("Graph must have the declared fixed float32 input shape")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Unexpected ONNX execution provider")
    for index, (output, expected_shape) in enumerate(zip(session.get_outputs(), _output_shapes(config, classes))):
        if output.type != ("tensor(int64)" if index == 2 else "tensor(float)"):
            raise ValueError("Unexpected ONNX output dtype")
        # Some exporters leave symbolic output metadata despite fixed inputs;
        # every actual output is checked against the exact dimensions at predict.
        if len(output.shape) != len(expected_shape) or any(type(x) is int and x != y for x, y in zip(output.shape, expected_shape)):
            raise ValueError("Unexpected ONNX output shape")
    return session


@dataclass(frozen=True)
class OnnxPrediction:
    logits: np.ndarray
    probabilities: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray


class OnnxSegmentationRunner:
    """Trusted, hash-pinned bundle only. Manifest pin binds class order + shape."""

    def __init__(self, bundle_dir, *, expected_model_sha256, expected_manifest_sha256, trusted_local=False):
        if type(trusted_local) is not bool or not trusted_local:
            raise ValueError("Only explicitly trusted local ONNX bundles may be loaded")
        root = Path(bundle_dir)
        data = _read_pinned(root / "manifest.json", expected_manifest_sha256, limit=64 * 1024)
        self.manifest = OnnxManifest.model_validate_json(data)
        if self.manifest.model_sha256 != expected_model_sha256:
            raise ValueError("Manifest/model hash mismatch")
        model_bytes = _read_pinned(root / "model.onnx", expected_model_sha256, limit=MAX_MODEL_BYTES)
        self._session = _session(model_bytes, self.manifest.config, len(self.manifest.taxonomy.classes))

    def predict(self, images):
        images = validate_onnx_input(images, self.manifest.config)
        values = self._session.run(list(OUTPUT_NAMES), {INPUT_NAME: images})
        _check_outputs(values, self.manifest.config, len(self.manifest.taxonomy.classes))
        return OnnxPrediction(*values)


def _publish_bundle(output_dir, files):
    """Exclusive directory + files. Clean only files created by this call on error."""
    output_dir.mkdir(parents=True, exist_ok=False)
    owned = []
    try:
        for name, source in files.items():
            target = output_dir / name
            with target.open("xb") as stream:
                owned.append(target)
                if isinstance(source, bytes):
                    stream.write(source)
                else:
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, stream)
    except BaseException:
        for target in owned:
            target.unlink(missing_ok=True)
        # Never remove unrelated/external files if someone populated this directory.
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def export_segmentation_onnx(checkpoint_path, *, output_dir, expected_sha256, expected_model_version,
                             trusted_local=False, allow_synthetic=False, config=None):
    if type(trusted_local) is not bool or not trusted_local:
        raise ValueError("Explicit trusted_local acknowledgement required")
    if type(allow_synthetic) is not bool:
        raise ValueError("Explicit synthetic boolean required")
    _check_pin(expected_sha256)
    if (not isinstance(expected_model_version, str) or not expected_model_version.strip()
            or expected_model_version != expected_model_version.strip() or len(expected_model_version) > 256):
        raise ValueError("Expected source model version required")
    config = OnnxExportConfig.model_validate(config.model_dump() if isinstance(config, OnnxExportConfig) else ({} if config is None else config))
    output_dir = Path(output_dir).resolve()
    # Reuse Phase 8's canonical data/source/cache/production-storage safeguards.
    validate_candidate_output_dir(output_dir)
    onnx, ort = onnx_dependencies()
    # Loading/export may consume RNG or backend flags; restore process state afterward.
    with training_reproducibility(16, 1):
        with Path(checkpoint_path).open("rb") as stream:
            if stream.seek(0, 2) > MAX_MODEL_BYTES:
                raise ValueError("Checkpoint exceeds size bound")
            stream.seek(0)
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Checkpoint SHA-256 mismatch")
            stream.seek(0)
            model, metadata = load_segmentation_checkpoint(stream, trusted_local=True, device="cpu")
        if metadata["model_version"] != expected_model_version:
            raise ValueError("Source model version mismatch")
        if metadata["synthetic"] and not allow_synthetic:
            raise ValueError("Synthetic checkpoint needs explicit allow_synthetic")
        wrapper = ExportPrediction(model).eval()
        cases = parity_inputs(config.input_shape)
        expected = {}
        with torch.inference_mode():
            for name, values in cases.items():
                expected[name] = tuple(item.cpu().numpy() for item in wrapper(torch.from_numpy(values)))
        with tempfile.TemporaryDirectory(prefix="fabric_onnx_export_") as directory:
            model_path = Path(directory) / "model.onnx"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with torch.inference_mode():
                    torch.onnx.export(wrapper, (torch.from_numpy(cases["seeded_noise"]),), str(model_path),
                        input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES), opset_version=config.opset_version,
                        dynamic_axes=None, dynamo=False, external_data=False,
                        training=torch.onnx.TrainingMode.EVAL, export_params=True)
            if model_path.stat().st_size > MAX_MODEL_BYTES:
                raise ValueError("Exported model exceeds size bound")
            model_bytes = model_path.read_bytes()
            session = _session(model_bytes, config, len(model.taxonomy.classes))
            measured = {}
            for name, values in cases.items():
                actual = session.run(list(OUTPUT_NAMES), {INPUT_NAME: validate_onnx_input(values, config)})
                measured[name] = compare_predictions(expected[name], actual, config)
            manifest = OnnxManifest(model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                source_checkpoint_sha256=expected_sha256, source_model_version=metadata["model_version"],
                synthetic=metadata["synthetic"], taxonomy=model.taxonomy, config=config,
                torch_version=str(torch.__version__), onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                parity_passed=True, parity_case_count=len(cases))
            parity = {"scope": "NUMERICAL_SOFTWARE_PARITY_NOT_ACCURACY_OR_RELEASE_ACCEPTANCE",
                "passed": True, "atol": config.atol, "rtol": config.rtol, "mask_requires_exact_equality": True,
                "cases": measured, "export_warnings": sorted(set(type(item.message).__name__ + ": " + str(item.message) for item in captured)),
                "note": "Python tensor-value/shape checks are traced as constants, NOT ONNX guards. Use the fixed-shape validating host runner.",
                "production_approved": False}
            manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
            report_bytes = json.dumps(parity, indent=2, allow_nan=False).encode("utf-8")
            _publish_bundle(output_dir, {"model.onnx": model_path, "manifest.json": manifest_bytes,
                                         "parity_report.json": report_bytes})
            return {"model_sha256": manifest.model_sha256, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "manifest": manifest.model_dump(mode="json"), "parity": parity,
                    "bundle_bytes": len(model_bytes), "output_dir": str(output_dir)}
