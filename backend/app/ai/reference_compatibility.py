"""Reject declared mismatches without inventing provenance for legacy metadata."""

from backend.app.ai.faiss_index import faiss
from backend.app.core.inspection_contracts import InspectionContext, InspectionInput


def validate_current_reference(detector, inspection: InspectionInput, context: InspectionContext) -> None:
    reference = context.reference
    if inspection.fabric_id != reference.fabric_id:
        raise ValueError("Inspection fabric does not match selected memory")
    if reference.backbone_version != context.backbone["backbone_version"]:
        raise ValueError("Selected reference encoder version does not match runtime")
    metadata = detector.metadata
    if metadata.get("artifact_type") == "memory_experiment" or metadata.get("status") == "EXPERIMENT_ONLY":
        raise ValueError("Research memory is not calibrated or approved for production inspection")
    if "schema_version" in metadata and (type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1):
        raise ValueError("Unsupported reference schema version")
    expected = {
        "fabric_id": reference.fabric_id, "reference_id": reference.reference_id,
        "reference_version": reference.reference_version, "backbone_version": reference.backbone_version,
        "adapter_version": context.backbone.get("adapter_version"),
    }
    for name, value in expected.items():
        if name in metadata and (metadata[name] != value or (type(value) is int and type(metadata[name]) is not int)):
            raise ValueError(f"Reference {name} mismatch")
    if metadata.get("faiss_metric", "cosine") != "cosine" or detector.index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Current dense reference requires cosine-compatible inner-product FAISS")
    index_path = detector.metadata_path.parent / metadata["faiss_index_path"]
    if index_path.resolve() != reference.faiss_index_path.resolve():
        raise ValueError("Selected registry index path disagrees with reference metadata")
    if "memory_bank_path" in metadata:
        bank_path = detector.metadata_path.parent / metadata["memory_bank_path"]
        if bank_path.resolve() != reference.memory_bank_path.resolve():
            raise ValueError("Selected registry memory path disagrees with reference metadata")
    if "feature_dim" in metadata and metadata["feature_dim"] != int(detector.index.d):
        raise ValueError("Reference feature dimension disagrees with FAISS index")
