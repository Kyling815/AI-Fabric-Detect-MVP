import pytest

from backend.app.core.qc import KnownSeverityRule, QCPolicy
from backend.app.services.qc_decision import QCDecisionEngine, qc_input_from_evidence
from tests.qc_fixtures import evidence, qc_input, qc_policy


@pytest.mark.parametrize("pixels,expected", [(1, "GOOD"), (128, "GOOD"), (204, "GOOD"), (205, "NEEDS_REVIEW"),
                                           (409, "NEEDS_REVIEW"), (410, "BAD"), (512, "BAD"), (1024, "BAD")])
def test_known_allow_reject_and_borderline_model_grid_severity(pixels, expected):
    decision = QCDecisionEngine(qc_policy()).evaluate(qc_input(pixels=pixels))
    assert decision.label == expected and not decision.applied
    assert decision.assessments[0].area_fraction == pixels / 1024
    assert decision.policy.policy_version and decision.rule_trace


def test_inclusive_exact_allow_reject_boundaries():
    policy = qc_policy(known_rules=(KnownSeverityRule(class_name="defect", allow_max_fraction=0.125, reject_min_fraction=0.5),))
    engine = QCDecisionEngine(policy)
    assert engine.evaluate(qc_input(pixels=128)).label == "GOOD"
    assert engine.evaluate(qc_input(pixels=129)).label == "NEEDS_REVIEW"
    assert engine.evaluate(qc_input(pixels=512)).label == "BAD"


@pytest.mark.parametrize("action", ["BAD", "NEEDS_REVIEW"])
def test_unknown_high_action_is_explicit_and_cannot_be_good(action):
    decision = QCDecisionEngine(qc_policy(unknown_high_action=action)).evaluate(qc_input(confidence=0.2, score=0.9))
    assert decision.label == action and decision.reason == "UNKNOWN_HIGH_ANOMALY"


def test_known_allowed_must_not_hide_simultaneous_high_unknown_anomaly():
    decision = QCDecisionEngine(qc_policy()).evaluate(qc_input(pixels=128, score=0.9))
    assert decision.label == "NEEDS_REVIEW" and decision.reason == "UNKNOWN_HIGH_ANOMALY"


def test_rejectable_known_is_bad_even_when_unknown_policy_only_requires_review():
    assert QCDecisionEngine(qc_policy()).evaluate(qc_input(pixels=512, score=0.9)).label == "BAD"


@pytest.mark.parametrize("score", [0.3, 0.4, 0.699999])
def test_borderline_anomaly_review_never_becomes_good(score):
    decision = QCDecisionEngine(qc_policy()).evaluate(qc_input(confidence=0.2, score=score))
    assert decision.label == "NEEDS_REVIEW"


@pytest.mark.parametrize("missing", ["supervised", "anomaly", "both"])
def test_missing_evidence_no_automatic_good(missing):
    assert QCDecisionEngine(qc_policy()).evaluate(qc_input(missing=missing)).label == "NEEDS_REVIEW"


@pytest.mark.parametrize("cap", ["invalid", "unchecked", "error"])
def test_invalid_capture_is_review_not_bad_fabric(cap):
    decision = QCDecisionEngine(qc_policy()).evaluate(qc_input(cap=cap, pixels=512))
    assert decision.label == "NEEDS_REVIEW"
    if cap == "invalid":
        assert decision.reason == "INVALID_CAPTURE"


def test_complete_normal_can_be_good_but_no_input_is_review():
    engine = QCDecisionEngine(qc_policy())
    assert engine.evaluate(qc_input(confidence=0.2)).label == "GOOD"
    assert engine.evaluate(None).label == "NEEDS_REVIEW"
    assert engine.evaluate(qc_input_from_evidence("fixture_fabric", None)).label == "NEEDS_REVIEW"


def test_unmapped_known_class_no_invented_allowance():
    decision = QCDecisionEngine(qc_policy(known_rules=())).evaluate(qc_input())
    assert decision.label == "NEEDS_REVIEW" and decision.assessments[0].outcome == "unmapped"


def test_missing_severity_no_allowance_even_if_ai_class_is_high_confidence():
    inputs = qc_input().model_copy(update={"severities": ()})
    result = QCDecisionEngine(qc_policy()).evaluate(inputs)
    assert result.label == "NEEDS_REVIEW" and result.reason == "KNOWN_SEVERITY_UNAVAILABLE"


@pytest.mark.parametrize("failure", ["synthetic", "uncalibrated", "scope", "identity", "severity_nan", "forged_count"])
def test_provenance_scope_and_invalid_input_gates(failure):
    inputs, policy = qc_input(), qc_policy()
    if failure == "synthetic":
        policy = qc_policy(allow_synthetic_evidence=False)
    elif failure == "uncalibrated":
        policy = qc_policy(allow_uncalibrated_supervised=False)
    elif failure == "scope":
        policy = qc_policy(allowed_fabric_ids=("other_fabric",))
    elif failure == "identity":
        inputs = inputs.model_copy(update={"fabric_id": "other_fabric"})
    elif failure == "severity_nan":
        inputs = inputs.model_copy(update={"evidence": inputs.evidence.model_copy(update={"known_confidence": float("nan")})})
    else:
        inputs = inputs.model_copy(update={"severities": (inputs.severities[0].model_copy(update={"area_px": 1}),)})
    result = QCDecisionEngine(policy).evaluate(inputs)
    assert result.label == "NEEDS_REVIEW" and result.issues


def test_default_policy_blocks_unapproved_synthetic_uncalibrated_normal():
    result = QCDecisionEngine(QCPolicy()).evaluate(qc_input(confidence=0.2))
    assert result.label == "NEEDS_REVIEW" and result.reason == "SYNTHETIC_OR_UNKNOWN_PROVENANCE"


def test_qc_policy_change_does_not_change_ai_evidence_or_fusion():
    inputs = qc_input(pixels=128)
    before = inputs.model_dump_json()
    allowed = QCDecisionEngine(qc_policy()).evaluate(inputs)
    rejected = QCDecisionEngine(qc_policy(known_rules=(KnownSeverityRule(class_name="defect", allow_max_fraction=0, reject_min_fraction=0.1),))).evaluate(inputs)
    assert allowed.label == "GOOD" and rejected.label == "BAD" and inputs.model_dump_json() == before


@pytest.mark.parametrize("failure", ["missing_trace", "duplicate_trace", "failed_method", "failed_geometry", "mask_payload_missing"])
def test_known_confidence_cannot_bypass_complete_evidence_checks(failure):
    inputs = qc_input()
    fields = inputs.evidence.model_dump(mode="json")
    if failure == "missing_trace":
        fields["trace"] = []
    elif failure == "duplicate_trace":
        fields["trace"].append(fields["trace"][0])
    elif failure == "mask_payload_missing":
        fields["supervised_mask"]["payload_available"] = False
    else:
        name = "anomaly_score_method_matches" if failure == "failed_method" else "geometry_compatible"
        next(item for item in fields["trace"] if item["rule"] == name)["passed"] = False
    from backend.app.core.fusion import DefectEvidence
    inputs = inputs.model_copy(update={"evidence": DefectEvidence.model_validate(fields)})
    decision = QCDecisionEngine(qc_policy()).evaluate(inputs)
    assert decision.label == "NEEDS_REVIEW" and decision.issues
