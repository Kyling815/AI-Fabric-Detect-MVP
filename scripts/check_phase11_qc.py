"""Read-only hypothetical QC rules demo. No production policy, model or registry activation."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.fusion import DefectEvidence
from backend.app.core.qc import KnownSeverityRule, QCPolicy
from backend.app.services.qc_decision import QCDecisionEngine, qc_input_from_evidence
from scripts.check_phase10_fusion import demo as fusion_demo


def demo():
    examples = {item["case"]: DefectEvidence.model_validate(item["result"]) for item in fusion_demo()["cases"]}
    policy = QCPolicy(policy_version="hypothetical_qc_demo_v1", allowed_fabric_ids=("hypothetical_fabric",),
        known_rules=(KnownSeverityRule(class_name="defect", allow_max_fraction=0.05, reject_min_fraction=0.2),),
        unknown_high_action="BAD", allow_synthetic_evidence=True, allow_uncalibrated_supervised=True)
    known = examples["high_supervised_low_anomaly"]
    def severity(pixels):
        fields = known.model_dump(mode="json")
        fields["known_defects"][0]["predicted_pixels"] = pixels
        return DefectEvidence.model_validate(fields)  # Declared hypothetical counts, not a real mask digest claim.
    cases = [("known_allowable", known, "GOOD"), ("known_rejectable", severity(512), "BAD"),
             ("known_borderline", severity(128), "NEEDS_REVIEW"), ("unknown_high", examples["low_supervised_high_anomaly"], "BAD"),
             ("anomaly_borderline", examples["low_supervised_review_anomaly"], "NEEDS_REVIEW"),
             ("complete_normal", examples["low_supervised_low_anomaly"], "GOOD"), ("no_evidence", None, "NEEDS_REVIEW"),
             ("missing_branch", examples["missing_supervised"], "NEEDS_REVIEW"), ("invalid_capture", examples["invalid_capture"], "NEEDS_REVIEW")]
    engine, outputs = QCDecisionEngine(policy), []
    for name, evidence, expected in cases:
        result = engine.evaluate(qc_input_from_evidence("hypothetical_fabric", evidence))
        if result.label != expected:
            raise AssertionError("QC demo mismatch: " + name)
        outputs.append({"case": name, "expected": expected, "decision": result.model_dump(mode="json")})
    return {"phase": 11, "status": "HYPOTHETICAL_SOFTWARE_CHECK_ONLY", "case_count": len(outputs), "cases": outputs,
            "business_thresholds_calibrated": False, "production_policy_activated": False, "models_trained_or_loaded": False,
            "writes_performed": False, "note": "Illustrative model-grid class area rules, NOT customer SLA, physical severity or real-data accuracy. Default runtime remains OFF/observe-only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    if parser.parse_args().demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
