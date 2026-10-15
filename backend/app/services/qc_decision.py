"""QC rules live here, never in neural networks. Legacy behavior retained exactly."""

from backend.app.core.qc import (ClassSeverity, LegacyDenseQCInput, LegacyPatchQCInput, QCClassAssessment,
                                 QCDecision, QCPolicy, QCRuleInput)


def decide_legacy_dense(value: LegacyDenseQCInput):
    if value.bad_threshold < value.review_threshold:
        raise ValueError("image_bad_threshold must be >= image_review_threshold")
    localized = (value.has_localization and value.localized_score is not None and value.localized_review_threshold is not None
                 and value.localized_score >= float(value.localized_review_threshold))
    if value.image_score < value.review_threshold and localized:
        return "NEEDS_REVIEW", "STRONG_LOCALIZED_ANOMALY"
    if value.image_score < value.review_threshold:
        return "GOOD", None
    if not value.has_localization:
        return "NEEDS_REVIEW", "ANOMALY_WITHOUT_LOCALIZATION"
    if value.image_score >= value.bad_threshold:
        return "BAD", None
    return "NEEDS_REVIEW", "ANOMALY_BELOW_BAD_THRESHOLD"


def legacy_confidence(base, max_score):
    return max(0.0, min(0.99, base + min(max_score, 0.2)))


def decide_legacy_patch(value: LegacyPatchQCInput):
    max_score = value.max_anomaly_score or 0.0
    if value.defect_count == 0:
        return "Good", legacy_confidence(0.9, max_score), "No localized suspicious defect was detected."
    if value.defect_area_ratio >= value.bad_area_threshold:
        return "Bad", legacy_confidence(0.75 + min(value.defect_area_ratio, 0.2), max_score), "Defect area exceeded the configured bad threshold."
    reason = ("A localized defect was detected below the configured minimum area threshold." if value.defect_area_ratio < value.good_area_threshold
              else "A small localized suspicious area was detected below the bad threshold.")
    return "Acceptable", legacy_confidence(0.7, max_score), reason


def qc_input_from_evidence(fabric_id, evidence):
    if evidence is None:
        return QCRuleInput(fabric_id=fabric_id)
    from backend.app.core.fusion import DefectEvidence
    evidence = DefectEvidence.model_validate(evidence.model_dump())
    mask = evidence.supervised_mask
    severities = tuple(ClassSeverity(class_name=item.class_name, class_id=item.class_id, area_px=item.predicted_pixels,
                                    frame_pixels=mask.shape[0] * mask.shape[1]) for item in evidence.known_defects) if mask is not None else ()
    return QCRuleInput(fabric_id=fabric_id, evidence=evidence, severities=severities)


def qc_failure(reason):
    return QCDecision(label="NEEDS_REVIEW", reason=reason, issues=(reason.lower(),), rule_trace=("fail_closed",))


class QCDecisionEngine:
    def __init__(self, policy: QCPolicy | None = None):
        self.policy = QCPolicy.model_validate((policy or QCPolicy()).model_dump())

    def evaluate(self, inputs: QCRuleInput | None) -> QCDecision:
        policy, trace, evidence = self.policy, [], None
        def result(label, reason, issues=(), assessments=()):
            return QCDecision(label=label, reason=reason, policy=policy, evidence_kind=evidence.kind if evidence is not None else None,
                              issues=tuple(issues), assessments=tuple(assessments), rule_trace=tuple(trace + [reason.lower()]))
        try:
            inputs = QCRuleInput.model_validate(inputs.model_dump()) if inputs is not None else None
        except Exception:
            return result("NEEDS_REVIEW", "INVALID_QC_INPUT", ("invalid_rule_input",))
        if inputs is None or inputs.evidence is None:
            return result("NEEDS_REVIEW", "NO_EVIDENCE", ("no_evidence",))
        evidence = inputs.evidence
        trace.append("rule_input_validated")
        if evidence.capture_status == "invalid" or evidence.kind == "INVALID_CAPTURE":
            return result("NEEDS_REVIEW", "INVALID_CAPTURE", ("invalid_capture",))
        if evidence.capture_status != "valid":
            return result("NEEDS_REVIEW", "CAPTURE_NOT_VERIFIED", ("capture_not_verified",))
        if evidence.identity is None or evidence.identity.fabric_id != inputs.fabric_id or (policy.allowed_fabric_ids and inputs.fabric_id not in policy.allowed_fabric_ids):
            return result("NEEDS_REVIEW", "QC_FABRIC_SCOPE_MISMATCH", ("fabric_scope_mismatch",))
        trace.append("capture_and_fabric_verified")
        if evidence.review_required or evidence.uncertainties or evidence.kind == "NEEDS_REVIEW":
            return result("NEEDS_REVIEW", "FUSED_EVIDENCE_REQUIRES_REVIEW", evidence.uncertainties or ("fused_review",))
        if evidence.supervised_status != "available" or evidence.anomaly_status != "available":
            return result("NEEDS_REVIEW", "INCOMPLETE_BRANCH_EVIDENCE", ("incomplete_branches",))
        checks = {item.rule: item.passed for item in evidence.trace}
        required = ("capture_valid", "both_branches_available", "identity_compatible", "geometry_compatible",
                    "synthetic_supervised_allowed", "anomaly_score_method_matches")
        if len(checks) != len(evidence.trace) or any(checks.get(name) is not True for name in required):
            return result("NEEDS_REVIEW", "EVIDENCE_TRACE_INCOMPLETE", ("unverified_fusion_checks",))
        if evidence.supervised_synthetic is not False and not policy.allow_synthetic_evidence:
            return result("NEEDS_REVIEW", "SYNTHETIC_OR_UNKNOWN_PROVENANCE", ("unverified_supervised_provenance",))
        if evidence.supervised_confidence_calibrated is not True and not policy.allow_uncalibrated_supervised:
            return result("NEEDS_REVIEW", "SUPERVISED_CONFIDENCE_UNCALIBRATED", ("uncalibrated_confidence",))
        trace.append("evidence_completeness_and_policy_gates_passed")
        rules = {rule.class_name: rule for rule in policy.known_rules}
        measurements = {(item.class_id, item.class_name): item for item in inputs.severities}
        assessments = []
        for item in evidence.known_defects:
            severity = measurements.get((item.class_id, item.class_name))
            if severity is None or evidence.supervised_mask is None or not evidence.supervised_mask.payload_available:
                return result("NEEDS_REVIEW", "KNOWN_SEVERITY_UNAVAILABLE", ("missing_class_severity",))
            rule = rules.get(item.class_name)
            outcome = "unmapped" if rule is None else "reject" if severity.fraction >= rule.reject_min_fraction else "allowed" if severity.fraction <= rule.allow_max_fraction else "review"
            assessments.append(QCClassAssessment(class_name=item.class_name, area_fraction=severity.fraction, outcome=outcome,
                allow_max_fraction=rule.allow_max_fraction if rule else None, reject_min_fraction=rule.reject_min_fraction if rule else None))
        trace.append("known_severity_assessed_in_model_grid_units")
        if any(item.outcome == "reject" for item in assessments):
            return result("BAD", "KNOWN_SEVERITY_REJECT", assessments=assessments)
        high_anomaly = evidence.policy is not None and evidence.anomaly_score is not None and evidence.anomaly_score >= evidence.policy.anomaly_high_threshold
        if evidence.kind == "UNKNOWN_DEFECT" or high_anomaly:
            return result(policy.unknown_high_action, "UNKNOWN_HIGH_ANOMALY", assessments=assessments)
        if checks.get("anomaly_review_at_threshold") or checks.get("localized_anomaly_review"):
            return result("NEEDS_REVIEW", "BORDERLINE_OR_LOCALIZED_ANOMALY", ("anomaly_review",), assessments)
        if any(item.outcome in {"unmapped", "review"} for item in assessments):
            return result("NEEDS_REVIEW", "KNOWN_SEVERITY_BORDERLINE_OR_UNMAPPED", ("unapproved_class_allowance",), assessments)
        if evidence.kind == "KNOWN_DEFECT" and assessments:
            return result("GOOD", "KNOWN_SEVERITY_ALLOWED", assessments=assessments)
        if evidence.kind == "NORMAL":
            return result("GOOD", "COMPLETE_NORMAL_EVIDENCE")
        return result("NEEDS_REVIEW", "UNSUPPORTED_QC_EVIDENCE", ("unsupported_kind",), assessments)
