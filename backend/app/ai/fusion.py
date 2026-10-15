"""Deterministic, auditable rule fusion. Pure evidence, never business QC decision."""

from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, DefectEvidence, FusionPolicy,
                                     RuleCheck, SupervisedEvidence)


class FusionEngine:
    def __init__(self, policy: FusionPolicy | None = None):
        self.policy = FusionPolicy.model_validate((policy or FusionPolicy()).model_dump())

    def fuse(self, supervised: SupervisedEvidence | None, anomaly: AnomalyEvidence | None, *,
             capture: CaptureEvidence | None = None) -> DefectEvidence:
        # Revalidate frozen models as callers can bypass validation via model_copy.
        try:
            supervised = SupervisedEvidence.model_validate(supervised.model_dump()) if supervised is not None else SupervisedEvidence(status="missing")
        except Exception:
            supervised = SupervisedEvidence(status="invalid", reason="invalid_supervised_payload")
        try:
            anomaly = AnomalyEvidence.model_validate(anomaly.model_dump()) if anomaly is not None else AnomalyEvidence(status="missing")
        except Exception:
            anomaly = AnomalyEvidence(status="invalid", reason="invalid_anomaly_payload")
        try:
            capture = CaptureEvidence.model_validate(capture.model_dump()) if capture is not None else CaptureEvidence()
        except Exception:
            capture = CaptureEvidence(status="error", reasons=("invalid_capture_payload",))
        policy = self.policy
        issues = []
        supervised_ok, anomaly_ok = supervised.status == "available", anomaly.status == "available"
        if not supervised_ok:
            issues.append("supervised_" + supervised.status)
        if not anomaly_ok:
            issues.append("anomaly_" + anomaly.status)
        if capture.status != "valid":
            issues.append("capture_" + capture.status)
        identity_match = not (supervised_ok and anomaly_ok) or supervised.identity == anomaly.identity
        geometry_match = not (supervised_ok and anomaly_ok) or supervised.mask.original_shape == anomaly.image_shape
        if not identity_match:
            issues.append("capture_identity_mismatch")
        if not geometry_match:
            issues.append("original_geometry_mismatch")
        synthetic_allowed = not supervised_ok or not supervised.synthetic or policy.allow_synthetic_supervised
        if not synthetic_allowed:
            issues.append("synthetic_supervised_disallowed")
        method_match = not anomaly_ok or anomaly.score_method == policy.anomaly_score_method
        if not method_match:
            issues.append("anomaly_score_method_mismatch")
        if supervised_ok and not supervised.mask.payload_available and supervised.candidates:
            issues.append("supervised_mask_payload_unavailable")
        candidates = tuple(sorted(supervised.candidates, key=lambda item: item.class_id)) if supervised_ok else ()
        confidence = max((item.confidence for item in candidates), default=None)
        comparable = identity_match and geometry_match
        known = tuple(item for item in candidates if item.confidence >= policy.known_threshold) if comparable and synthetic_allowed else ()
        score = anomaly.score if anomaly_ok else None
        high = bool(anomaly_ok and method_match and comparable and score >= policy.anomaly_high_threshold)
        review = bool(anomaly_ok and method_match and comparable and score >= policy.anomaly_review_threshold)
        localized = bool(anomaly_ok and comparable and anomaly.localized_review_trigger)
        if (high or review or localized) and not anomaly.boxes:
            issues.append("anomaly_localization_unavailable")
        if capture.status == "invalid":
            kind, rule = "INVALID_CAPTURE", "capture_invalid"
        elif capture.status != "valid":
            kind, rule = "NEEDS_REVIEW", "capture_not_verified"
        elif not comparable:
            kind, rule = "NEEDS_REVIEW", "incompatible_evidence"
        elif known:
            kind, rule = "KNOWN_DEFECT", "known_foreground_at_threshold"
        elif high:
            kind, rule = "UNKNOWN_DEFECT", "high_anomaly_at_threshold"
        elif review or localized:
            kind, rule = "NEEDS_REVIEW", "localized_anomaly_review" if localized and not review else "anomaly_review_at_threshold"
        elif issues:
            kind, rule = "NEEDS_REVIEW", "incomplete_or_untrusted_evidence"
        else:
            kind, rule = "NORMAL", "complete_below_thresholds"
        trace = tuple(RuleCheck(rule=name, passed=bool(passed)) for name, passed in (
            ("capture_valid", capture.status == "valid"), ("both_branches_available", supervised_ok and anomaly_ok),
            ("identity_compatible", identity_match), ("geometry_compatible", geometry_match),
            ("synthetic_supervised_allowed", synthetic_allowed), ("anomaly_score_method_matches", method_match),
            ("known_foreground_at_threshold", bool(known)), ("high_anomaly_at_threshold", high),
            ("anomaly_review_at_threshold", review), ("localized_anomaly_review", localized),
        ))
        return DefectEvidence(kind=kind, rule_id=rule, policy=policy, capture_status=capture.status, capture_source=capture.source,
            capture_reasons=capture.reasons,
            supervised_status=supervised.status, anomaly_status=anomaly.status,
            identity=anomaly.identity if anomaly_ok else supervised.identity,
            supervised_model_version=supervised.model_version, supervised_checkpoint_sha256=supervised.checkpoint_sha256,
            supervised_synthetic=supervised.synthetic, supervised_confidence_calibrated=supervised.confidence_calibrated if supervised_ok else None,
            anomaly_reference_version=anomaly.reference_version, known_confidence=confidence, known_defects=known,
            supervised_mask=supervised.mask, anomaly_score=score, anomaly_boxes=anomaly.boxes,
            uncertainties=tuple(issues), review_required=bool(issues or kind in {"NEEDS_REVIEW", "INVALID_CAPTURE"}), trace=trace)


def fusion_failure(reason: str) -> DefectEvidence:
    return DefectEvidence(kind="NEEDS_REVIEW", rule_id=reason, capture_status="unchecked",
                          supervised_status="invalid", anomaly_status="invalid", review_required=True, uncertainties=(reason,))
