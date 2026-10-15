"""Read-only hypothetical Phase 10 truth table; no CNN inference, QC activation or writes."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.fusion import FusionEngine
from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, EvidenceIdentity, FusionBox, FusionPolicy,
                                     KnownClassEvidence, MaskDescriptor, SupervisedEvidence)


def demo():
    identity = EvidenceIdentity(fabric_id="hypothetical_fabric", capture_id="hypothetical_capture")
    valid = CaptureEvidence(status="valid", source="caller_declared")
    def sup(confidence):
        return SupervisedEvidence(status="available", identity=identity, model_version="hypothetical_model_v1", checkpoint_sha256="a" * 64,
                synthetic=True, taxonomy=("background", "defect"), candidates=(KnownClassEvidence(class_name="defect", class_id=1,
                confidence=confidence, predicted_pixels=20),), mask=MaskDescriptor(sha256="b" * 64, shape=(32, 32), original_shape=(32, 48), payload_available=True))
    def ano(score):
        return AnomalyEvidence(status="available", identity=identity, reference_version="hypothetical_ref_v1", backbone_version="hypothetical_encoder_v1",
                score=score, image_shape=(32, 48), score_method="top_k_mean", boxes=(FusionBox(x=2, y=2, w=4, h=5, score=score, label="dense_anomaly"),))
    engine = FusionEngine(FusionPolicy(allow_synthetic_supervised=True))  # Explicitly demo-only.
    cases = [
        ("high_supervised_low_anomaly", sup(0.95), ano(0.1), valid, "KNOWN_DEFECT"),
        ("high_supervised_high_anomaly", sup(0.95), ano(0.9), valid, "KNOWN_DEFECT"),
        ("low_supervised_high_anomaly", sup(0.2), ano(0.9), valid, "UNKNOWN_DEFECT"),
        ("low_supervised_review_anomaly", sup(0.2), ano(0.4), valid, "NEEDS_REVIEW"),
        ("low_supervised_low_anomaly", sup(0.2), ano(0.1), valid, "NORMAL"),
        ("missing_supervised", None, ano(0.1), valid, "NEEDS_REVIEW"),
        ("missing_anomaly", sup(0.2), None, valid, "NEEDS_REVIEW"),
        ("both_missing", None, None, valid, "NEEDS_REVIEW"),
        ("supervised_failure", SupervisedEvidence(status="failed"), ano(0.1), valid, "NEEDS_REVIEW"),
        ("anomaly_failure", sup(0.2), AnomalyEvidence(status="failed"), valid, "NEEDS_REVIEW"),
        ("supervised_timeout", SupervisedEvidence(status="timeout"), ano(0.1), valid, "NEEDS_REVIEW"),
        ("invalid_capture", sup(0.95), ano(0.9), CaptureEvidence(status="invalid", source="caller_declared", reasons=("invalid_fixture",)), "INVALID_CAPTURE"),
        ("capture_unchecked", sup(0.2), ano(0.1), CaptureEvidence(), "NEEDS_REVIEW"),
    ]
    outputs = []
    for name, supervised, anomaly, capture, expected in cases:
        result = engine.fuse(supervised, anomaly, capture=capture)
        if result.kind != expected:
            raise AssertionError("Fusion truth table mismatch: " + name)
        outputs.append({"case": name, "expected": expected, "result": result.model_dump(mode="json")})
    default_block = FusionEngine().fuse(sup(0.95), ano(0.1), capture=valid)
    if default_block.kind != "NEEDS_REVIEW":
        raise AssertionError("Default synthetic evidence gate failed")
    return {"phase": 10, "status": "HYPOTHETICAL_SOFTWARE_CHECK_ONLY", "cases": outputs,
            "default_synthetic_policy_result": default_block.model_dump(mode="json"), "case_count": len(outputs),
            "models_trained_or_loaded": False, "real_accuracy_evaluated": False, "artifacts_created": False,
            "hybrid_qc_active": False, "note": "Declared example confidence, scores, hashes and capture validity only; not real predictions or accepted thresholds. NORMAL is evidence, never a GOOD business decision."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    if parser.parse_args().demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
