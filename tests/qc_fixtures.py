"""Hypothetical QC rules/measurements only, not a customer SLA or real model accuracy."""

from backend.app.ai.fusion import FusionEngine
from backend.app.core.qc import KnownSeverityRule, QCPolicy
from backend.app.services.qc_decision import qc_input_from_evidence
from tests.fusion_fixtures import anomaly, capture, policy as fusion_policy, supervised


def qc_policy(**changes):
    values = dict(policy_version="qc_hypothetical_fixture_v1", allowed_fabric_ids=("fixture_fabric",),
                  known_rules=(KnownSeverityRule(class_name="defect", allow_max_fraction=0.2, reject_min_fraction=0.4),),
                  allow_synthetic_evidence=True, allow_uncalibrated_supervised=True)
    values.update(changes)
    return QCPolicy(**values)


def evidence(*, pixels=128, confidence=0.99, score=0.1, cap="valid", missing=None):
    sup = supervised(confidence)
    if pixels:
        sup = sup.model_copy(update={"candidates": (sup.candidates[0].model_copy(update={"predicted_pixels": pixels}),)})
    else:
        sup = sup.model_copy(update={"candidates": ()})
    ano = anomaly(score)
    return FusionEngine(fusion_policy()).fuse(None if missing in {"supervised", "both"} else sup,
                        None if missing in {"anomaly", "both"} else ano, capture=capture(cap))


def qc_input(**changes):
    return qc_input_from_evidence("fixture_fabric", evidence(**changes))
