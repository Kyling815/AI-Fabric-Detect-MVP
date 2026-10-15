from pathlib import Path

import pytest

from backend.app.core.qc import ClassSeverity, KnownSeverityRule, QCDecision, QCPolicy, QCRuleInput
from backend.app.services.qc_decision import QCDecisionEngine
from tests.qc_fixtures import qc_input, qc_policy


def test_default_example_has_no_invented_customer_class_allowance():
    root = Path(__file__).resolve().parents[1]
    example = QCPolicy.model_validate_json((root / "config/qc_policy_example.json").read_text())
    assert example == QCPolicy() and not example.known_rules and not example.allow_synthetic_evidence
    assert not example.allow_uncalibrated_supervised and example.unknown_high_action == "NEEDS_REVIEW"


@pytest.mark.parametrize("changes", [{"class_name": "background"}, {"class_name": "made_up"}, {"unit": "mm2"},
        {"allow_max_fraction": 0.4}, {"reject_min_fraction": 0.1}, {"allow_max_fraction": True},
        {"allow_max_fraction": "0.1"}, {"reject_min_fraction": float("nan")}, {"allow_max_fraction": -0.1}])
def test_invalid_severity_policy_or_fake_physical_units_rejected(changes):
    fields = dict(class_name="defect", allow_max_fraction=0.1, reject_min_fraction=0.4)
    fields.update(changes)
    with pytest.raises(ValueError):
        KnownSeverityRule(**fields)


@pytest.mark.parametrize("changes", [{"policy_version": " "}, {"allowed_fabric_ids": ["a", "a"]},
        {"unknown_high_action": "GOOD"}, {"allow_synthetic_evidence": "true"}, {"allow_uncalibrated_supervised": 1}, {"learned_weights": [1]}])
def test_invalid_policy_cannot_waive_unknown_or_strict_gates(changes):
    with pytest.raises(ValueError):
        QCPolicy(**changes)


def test_duplicate_class_rules_rejected():
    rule = KnownSeverityRule(class_name="defect", allow_max_fraction=0.1, reject_min_fraction=0.4)
    with pytest.raises(ValueError):
        QCPolicy(known_rules=(rule, rule))


@pytest.mark.parametrize("changes", [{"class_id": True}, {"area_px": 0}, {"area_px": 1025}, {"frame_pixels": 0}, {"coordinate_frame": "source_mm"}])
def test_severity_must_be_bounded_integer_model_grid_count(changes):
    fields = dict(class_name="defect", class_id=1, area_px=128, frame_pixels=1024)
    fields.update(changes)
    with pytest.raises(ValueError):
        ClassSeverity(**fields)


@pytest.mark.parametrize("failure", ["wrong_area", "wrong_grid", "wrong_class", "duplicate", "no_evidence"])
def test_rule_inputs_cannot_fabricate_severity(failure):
    fields = qc_input().model_dump(mode="json")
    if failure == "wrong_area":
        fields["severities"][0]["area_px"] = 1
    elif failure == "wrong_grid":
        fields["severities"][0]["frame_pixels"] = 2048
    elif failure == "wrong_class":
        fields["severities"][0]["class_name"] = "scratch"
    elif failure == "duplicate":
        fields["severities"].append(fields["severities"][0])
    else:
        fields["evidence"] = None
    with pytest.raises(ValueError):
        QCRuleInput.model_validate(fields)


@pytest.mark.parametrize("failure", ["no_policy", "uncertainty", "no_trace", "unknown_good", "unapproved_allowance", "observer_applied"])
def test_qc_output_cannot_claim_good_without_traceable_rules(failure):
    fields = QCDecisionEngine(qc_policy()).evaluate(qc_input()).model_dump(mode="json")
    if failure == "no_policy":
        fields["policy"] = None
    elif failure == "uncertainty":
        fields["issues"] = ["missing_branch"]
    elif failure == "no_trace":
        fields["rule_trace"] = []
    elif failure == "unknown_good":
        fields["evidence_kind"] = "UNKNOWN_DEFECT"
    elif failure == "unapproved_allowance":
        fields["assessments"][0]["allow_max_fraction"] = 0.9
    else:
        fields["applied"] = True
    with pytest.raises(ValueError):
        QCDecision.model_validate(fields)


@pytest.mark.parametrize("failure", ["duplicate_id", "duplicate_name", "overlapping_counts"])
def test_disjoint_model_grid_counts_cannot_be_fabricated(failure):
    from copy import deepcopy
    fields = qc_input().model_dump(mode="json")
    extra = deepcopy(fields["evidence"]["known_defects"][0])
    if failure == "duplicate_id":
        extra["class_name"] = "scratch"
    elif failure == "duplicate_name":
        extra["class_id"] = 2
    else:
        extra.update(class_name="scratch", class_id=2, predicted_pixels=1024)
    fields["evidence"]["known_defects"].append(extra)
    with pytest.raises(ValueError):
        QCRuleInput.model_validate(fields)
