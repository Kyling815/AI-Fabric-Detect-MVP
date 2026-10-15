from itertools import product

import pytest

from backend.app.ai.fusion import FusionEngine
from backend.app.core.fusion import AnomalyEvidence, SupervisedEvidence
from tests.fusion_fixtures import anomaly, capture, identity, policy, supervised


@pytest.mark.parametrize("confidence,score", list(product((0.0, 0.2, 0.899999, 0.9, 0.99, 1.0), (0.0, 0.299999, 0.3, 0.699999, 0.7, 1.2))))
def test_exhaustive_threshold_grid_known_unknown_review_normal(confidence, score):
    engine = FusionEngine(policy())
    value = engine.fuse(supervised(confidence), anomaly(score), capture=capture())
    expected = "KNOWN_DEFECT" if confidence >= 0.9 else "UNKNOWN_DEFECT" if score >= 0.7 else "NEEDS_REVIEW" if score >= 0.3 else "NORMAL"
    assert value.kind == expected and not value.affects_qc and value.anomaly_score == score
    assert value.review_required == (expected == "NEEDS_REVIEW")
    assert value.rule_id and value.policy.policy_version and len(value.trace) == 10
    assert engine.fuse(supervised(confidence), anomaly(score), capture=capture()) == value


@pytest.mark.parametrize("status", ["missing", "disabled", "failed", "timeout", "busy", "invalid"])
@pytest.mark.parametrize("branch", ["supervised", "anomaly"])
def test_missing_failed_branch_with_low_other_never_becomes_normal(status, branch):
    sup = SupervisedEvidence(status=status) if branch == "supervised" else supervised()
    ano = AnomalyEvidence(status=status) if branch == "anomaly" else anomaly()
    result = FusionEngine(policy()).fuse(sup, ano, capture=capture())
    assert result.kind == "NEEDS_REVIEW" and result.review_required and branch + "_" + status in result.uncertainties


@pytest.mark.parametrize("branch", ["supervised", "anomaly", "both"])
def test_none_is_missing_not_zero_score(branch):
    result = FusionEngine(policy()).fuse(None if branch in {"supervised", "both"} else supervised(),
                                        None if branch in {"anomaly", "both"} else anomaly(), capture=capture())
    assert result.kind == "NEEDS_REVIEW" and result.review_required
    if branch in {"anomaly", "both"}:
        assert result.anomaly_score is None


@pytest.mark.parametrize("kind", ["known", "unknown"])
def test_positive_evidence_is_preserved_when_other_branch_unavailable_but_review_required(kind):
    result = FusionEngine(policy()).fuse(supervised(0.99) if kind == "known" else None,
                                         anomaly(0.8) if kind == "unknown" else None, capture=capture())
    assert result.kind == ("KNOWN_DEFECT" if kind == "known" else "UNKNOWN_DEFECT") and result.review_required
    assert result.uncertainties


@pytest.mark.parametrize("state", ["invalid", "unchecked", "error"])
@pytest.mark.parametrize("positive", [False, True])
def test_capture_failure_dominates_positive_and_negative_branches(state, positive):
    result = FusionEngine(policy()).fuse(supervised(0.99 if positive else 0.2), anomaly(0.9 if positive else 0.1), capture=capture(state))
    assert result.kind == ("INVALID_CAPTURE" if state == "invalid" else "NEEDS_REVIEW") and result.review_required


def test_absent_capture_is_unchecked_not_valid():
    result = FusionEngine(policy()).fuse(supervised(), anomaly())
    assert result.kind == "NEEDS_REVIEW" and "capture_unchecked" in result.uncertainties


@pytest.mark.parametrize("mismatch", ["fabric", "capture", "geometry"])
def test_incompatible_capture_identity_or_geometry_cannot_fuse_known_or_normal(mismatch):
    sup = supervised(0.99)
    sup = sup.model_copy(update={"mask": sup.mask.model_copy(update={"original_shape": (33, 48)})}) if mismatch == "geometry" else sup.model_copy(
            update={"identity": identity("other_fabric" if mismatch == "fabric" else "fixture_fabric", "other_capture" if mismatch == "capture" else "fixture_capture")})
    result = FusionEngine(policy()).fuse(sup, anomaly(0.9), capture=capture())
    assert result.kind == "NEEDS_REVIEW" and result.review_required and not result.known_defects


def test_background_probability_never_becomes_known_defect():
    result = FusionEngine(policy()).fuse(supervised(foreground=False), anomaly(), capture=capture())
    assert result.kind == "NORMAL" and result.known_confidence is None and not result.known_defects


def test_score_method_mismatch_does_not_apply_unrelated_thresholds():
    result = FusionEngine(policy()).fuse(supervised(), anomaly(1.2, method="max"), capture=capture())
    assert result.kind == "NEEDS_REVIEW" and "anomaly_score_method_mismatch" in result.uncertainties


def test_anomaly_distance_is_not_a_probability_or_clipped():
    result = FusionEngine(policy(anomaly_high_threshold=1.5, anomaly_review_threshold=1.0)).fuse(supervised(), anomaly(1.6), capture=capture())
    assert result.kind == "UNKNOWN_DEFECT" and result.anomaly_score == 1.6


def test_existing_strong_localized_signal_cannot_become_normal():
    result = FusionEngine(policy()).fuse(supervised(), anomaly(0.1, localized=True), capture=capture())
    assert result.kind == "NEEDS_REVIEW" and result.rule_id == "localized_anomaly_review"


def test_unknown_without_localization_preserves_positive_but_requires_review():
    result = FusionEngine(policy()).fuse(supervised(), anomaly(0.9, boxes=False), capture=capture())
    assert result.kind == "UNKNOWN_DEFECT" and result.review_required and "anomaly_localization_unavailable" in result.uncertainties


def test_synthetic_default_block_and_explicit_opt_in():
    result = FusionEngine().fuse(supervised(0.99), anomaly(), capture=capture())
    assert result.kind == "NEEDS_REVIEW" and "synthetic_supervised_disallowed" in result.uncertainties
    assert FusionEngine(policy()).fuse(supervised(0.99), anomaly(), capture=capture()).kind == "KNOWN_DEFECT"


def test_mask_payload_omission_does_not_claim_complete_normal_or_localized_known():
    low = FusionEngine(policy()).fuse(supervised(payload=False), anomaly(), capture=capture())
    high = FusionEngine(policy()).fuse(supervised(0.99, payload=False), anomaly(), capture=capture())
    assert low.kind == "NEEDS_REVIEW" and low.review_required
    assert high.kind == "KNOWN_DEFECT" and high.review_required and not high.supervised_mask.payload_available


@pytest.mark.parametrize("corrupt", ["supervised_nan", "anomaly_nan", "negative_score", "capture_invalid_model"])
def test_model_copy_bypass_is_revalidated_fail_safe(corrupt):
    sup, ano, cap = supervised(), anomaly(), capture()
    if corrupt == "supervised_nan":
        sup = sup.model_copy(update={"candidates": (sup.candidates[0].model_copy(update={"confidence": float("nan")}),)})
    elif corrupt in {"anomaly_nan", "negative_score"}:
        ano = ano.model_copy(update={"score": float("nan") if corrupt == "anomaly_nan" else -1})
    else:
        cap = cap.model_copy(update={"status": "valid", "source": "not_checked"})
    result = FusionEngine(policy()).fuse(sup, ano, capture=cap)
    assert result.kind == "NEEDS_REVIEW" and result.review_required


def test_fusion_does_not_mutate_inputs_and_retains_both_positive_provenances():
    sup, ano = supervised(0.99), anomaly(0.9)
    before = (sup.model_dump_json(), ano.model_dump_json())
    result = FusionEngine(policy()).fuse(sup, ano, capture=capture())
    assert result.kind == "KNOWN_DEFECT" and result.anomaly_score == 0.9
    assert result.supervised_model_version == sup.model_version and result.anomaly_reference_version == ano.reference_version
    assert (sup.model_dump_json(), ano.model_dump_json()) == before
