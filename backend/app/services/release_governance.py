"""Offline acceptance and EXPLICIT manual activation. Never called by runtime.

Evidence is imported from a trusted evaluator, NOT generated model predictions.
Hashes bind bytes, not the truth of human provenance/annotation attestations.
No report generation, metric pass or validation step activates a registry.
"""

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from backend.app.core.release_governance import (AcceptancePolicy, ArtifactPin, Candidate,
    DatasetInventory, EvaluationEvidence, LockedTest, ManualApproval, METRIC_DIRECTIONS,
    ReleaseBundle, ValidationEvidence)

ROOT = Path(__file__).resolve().parents[3]
# All Phase17 ONNX/TorchScript/engines, including historical attempts. Renaming
# or declaring one of these trained does not make it eligible for release.
PHASE17_RESEARCH_HASHES = frozenset({
    "51decba9681af3d43925db55d3cb322ce8a3813d6a93e466d7eddd0b44dbce5b",
    "3d7fc1c28fb349b2bdfbe916e973c5bd3a20930fb7f2ba04963f592d13ca789c",
    "cafce9eebfa3788ab1fb55c30910974cf13cba35fa2dabf205121b79d9cb195f",
    "dc30e4cd6343a6bf464df6555f921ddaaafd1941a30ceec0a2cc941a15ba07d1",
    "051ed5c3a6c05f4c2c4c2d5c41f9d5e39bc01cfc9be22aae07cc86cb9f4fa431",
    "ac8257bebac77444b33eb6694f2d628e585142c760d178243ccf5f529e94e27d",
    "fb993b16df1121f882c263914d3ac307fd6b4fe91bfbd5d87d058055d66a1401",
    "6adc9b60f036ef05df17d8a26e7009cc1baecc091390add5820cf02fa790e99f",
    "b9f97e5661b6c398c5cc3b82c8a253cbf819c1ed9473dbf7f90586c5973d2855",
    "bfd7cb40e1130723531f5eed9c87eaf0b1533063ddf5af76eca45f3e73122123",
    "65d0bcbcc1f3c8205b092174007c7b122c0a5a3d13354b123e88a9918501a245",
    "b68edecc552fb1100bbf26399260674a70d8f19c242a7f88ada52f2dea264734",
    "ddea968b058763509bc89155384fed86a3d3293c13acd3028dd62821d16caf9e",
})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_pin(pin, limit=1024 * 1024 * 1024):
    pin = ArtifactPin.model_validate(pin) if not isinstance(pin, ArtifactPin) else pin
    with Path(pin.path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit:
            raise ValueError("Artifact exceeds size bound: " + pin.path)
        data = stream.read(size + 1)
        if len(data) != size:
            raise ValueError("Artifact changed while reading: " + pin.path)
    if len(data) > limit or digest(data) != pin.sha256:
        raise ValueError("Artifact size/SHA mismatch: " + pin.path)
    return data


def load(pin, contract):
    return contract.model_validate_json(read_pin(pin, limit=16 * 1024 * 1024))


def exposure_tokens(sample):
    # Global acquisition ids, not inferred folder names. Hash lineage also
    # prevents identical source bytes renamed/re-cropped across partitions.
    tokens = {("source", s.casefold()) for s in (sample.source_id, *sample.ancestor_source_ids)}
    tokens |= {("sha", s) for s in (sample.source_sha256, sample.image.sha256, *sample.ancestor_source_sha256s)}
    tokens |= {("lot", sample.lot_id.casefold()), ("roll", sample.roll_id.casefold()),
               ("camera", sample.camera_id.casefold()), ("path", str(Path(sample.image.path).resolve()).casefold())}
    return tokens


def legacy_exposure_tokens(root=None):
    """Deny known Company development/historical data; hash only, NO evaluation.

    Freshness still requires the supplied COMPLETE exposure inventory. This
    built-in inventory is a lower bound, not proof of all external history.
    """
    tokens = set()
    folder = Path(root or ROOT) / "storage/metadata/company_demo_manifests"
    for path in sorted(folder.glob("*_all.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("source_original"):
                    tokens.add(("source", row["source_original"].casefold()))
                image = Path(row["image_path"])
                tokens.add(("path", str(image.resolve()).casefold()))
                if image.is_file():
                    with image.open("rb") as incoming:
                        tokens.add(("sha", hashlib.file_digest(incoming, "sha256").hexdigest()))
    return tokens


def inspected_length(samples):
    if any(s.start_m is None for s in samples):
        return None
    groups = {}
    for sample in samples:
        # Do not double count the same physical roll across multiple cameras.
        groups.setdefault((sample.fabric_id, sample.lot_id, sample.roll_id), []).append((sample.start_m, sample.end_m))
    total = 0.0
    for intervals in groups.values():
        previous = None
        for start, end in sorted(intervals):
            if previous is not None and start < previous:
                raise ValueError("Overlapping inspected intervals; per100m denominator would be inflated")
            total += end - start
            previous = end
    return total if total > 0 else None


def compute_metrics(evidence, locked, classes, segmentation):
    rows = evidence.observations
    tp = sum(r.matched_events for r in rows)
    fn = sum(r.truth_events - r.matched_events for r in rows)
    fp = sum(r.false_alarm_events for r in rows)
    normal = [r for r in rows if r.truth_events == 0]
    normal_fp = sum(r.predicted_defective for r in normal)
    unknown = [r for r in rows if r.unknown_defect]
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    meters = inspected_length(locked.inventory.samples) if locked.length_calibration is not None else None
    pixel_counts_present = all(r.pixel_tp is not None for r in rows)
    ptp = sum(r.pixel_tp for r in rows) if pixel_counts_present else 0
    pfp = sum(r.pixel_fp for r in rows) if pixel_counts_present else 0
    pfn = sum(r.pixel_fn for r in rows) if pixel_counts_present else 0
    metrics = {"defect_recall": recall, "false_negatives": fn, "underkill": ratio(fn, tp + fn),
        "false_positive_rate": ratio(normal_fp, len(normal)), "overkill": ratio(normal_fp, len(normal)),
        "precision": precision, "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "segmentation_iou": ratio(ptp, ptp + pfp + pfn) if segmentation and pixel_counts_present else None,
        "segmentation_dice": ratio(2 * ptp, 2 * ptp + pfp + pfn) if segmentation and pixel_counts_present else None,
        "unknown_defect_recall": ratio(sum(r.matched_events for r in unknown), sum(r.truth_events for r in unknown)),
        "false_alarms_per_100m": fp * 100 / meters if meters else None,
        "missed_defects_per_100m": fn * 100 / meters if meters else None,
        "p50_latency_ms": float(np.percentile(evidence.latency_ms, 50)),
        "p95_latency_ms": float(np.percentile(evidence.latency_ms, 95)),
        "throughput_images_per_second": evidence.throughput_images / evidence.throughput_wall_seconds}
    per_class = {name: ratio(sum(r.matched_events for r in rows if r.defect_class == name and not r.unknown_defect),
                              sum(r.truth_events for r in rows if r.defect_class == name and not r.unknown_defect)) for name in classes}
    return metrics, per_class, {"matched_defect_events": tp, "missed_defect_events": fn, "false_alarm_events": fp,
        "normal_decision_units": len(normal), "normal_positive_units": normal_fp, "inspected_meters": meters,
        "metric_units": "recall/precision/F1/underkill: one-to-one defect-event matching; FPR/overkill: normal inspection-unit rate; IoU/Dice: micro foreground pixels"}


def blocked_report(reasons, *, status="BLOCKED BY DATA / TRAINED CANDIDATE"):
    return {"schema_version": 1, "phase": 18, "status": status, "accepted": False,
        "metrics": {name: None for name in METRIC_DIRECTIONS}, "per_class_recall": {},
        "blocking_reasons": reasons, "manual_activation_required": True, "registry_modified": False,
        "production_quality_evidence": False, "automatic_activation": False}


def assess_release(bundle_pin, *, trusted_local=False, now=None):
    if trusted_local is not True:
        raise ValueError("Explicit trusted local evidence acknowledgement required")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Timezone-aware current time required")
    bundle = load(bundle_pin, ReleaseBundle)
    candidate = load(bundle.candidate, Candidate)
    policy = load(bundle.policy, AcceptancePolicy)
    used = load(bundle.used_inventory, DatasetInventory)
    locked = load(bundle.locked_test, LockedTest)
    validation = load(bundle.validation, ValidationEvidence)
    evidence = load(bundle.evaluation, EvaluationEvidence)
    reasons = []
    if candidate.precision != "FP32":
        reasons.append("FP16_REJECTED_NO_PROMOTION_OR_FALLBACK")
    if candidate.research_only or candidate.origin != "real" or not candidate.trained or candidate.training_report is None:
        reasons.append("TRAINED_REAL_CANDIDATE_REQUIRED")
    if (any(p.sha256 in PHASE17_RESEARCH_HASHES for p in (candidate.model, *candidate.artifacts.values()))
            or candidate.candidate_id.casefold().startswith("phase17_")):
        reasons.append("PHASE17_RESEARCH_PROBE_PERMANENTLY_INELIGIBLE")
    # Every candidate/reference/runtime/threshold blob is checked without loading
    # executable models. Missing blobs are errors, never successful evidence.
    for pin in (candidate.model, *candidate.artifacts.values()):
        read_pin(pin)
    if candidate.training_report is not None:
        training = json.loads(read_pin(candidate.training_report, 16 * 1024 * 1024))
        if (training.get("trained") is not True or training.get("synthetic") is not False
                or training.get("model_sha256") != candidate.model.sha256
                or training.get("model_version") != candidate.model_version
                or training.get("used_inventory_sha256") != bundle.used_inventory.sha256):
            reasons.append("TRAINING_PROVENANCE_NOT_VERIFIED")
    recipe = json.loads(read_pin(candidate.artifacts["runtime_recipe"], 1024 * 1024))
    if (recipe.get("scope") != "full_hybrid_pipeline" or recipe.get("precision") != "FP32"
            or recipe.get("runtime_version") != candidate.runtime_version
            or recipe.get("model_sha256") != candidate.model.sha256
            or recipe.get("artifact_sha256") != {role: pin.sha256 for role, pin in candidate.artifacts.items() if role != "runtime_recipe"}
            or recipe.get("fp16_fallback_enabled") is not False):
        reasons.append("FULL_HYBRID_FP32_RUNTIME_RECIPE_REQUIRED")
    if (policy.approved_by is None or policy.approved_at is None or policy.valid_until is None
            or not policy.approved_at <= candidate.frozen_at <= locked.locked_at <= now < policy.valid_until):
        reasons.append("POLICY_UNAPPROVED_EXPIRED_OR_NOT_FROZEN_BEFORE_TEST")
    if (locked.bound_candidate_sha256 != bundle.candidate.sha256 or locked.bound_policy_sha256 != bundle.policy.sha256
            or locked.previously_opened or locked.historical_or_postmortem
            or locked.inventory.purpose != "fresh_locked"):
        reasons.append("FRESH_UNOPENED_CANDIDATE_POLICY_BOUND_LOCK_REQUIRED")
    if used.purpose != "development_and_historical" or not used.complete_exposure_inventory:
        reasons.append("COMPLETE_DEVELOPMENT_AND_HISTORICAL_EXPOSURE_INVENTORY_REQUIRED")
    if (validation.synthetic or not validation.passed or not validation.functional_checks_passed
            or not validation.numerical_parity_passed or validation.candidate_sha256 != bundle.candidate.sha256
            or validation.used_inventory_sha256 != bundle.used_inventory.sha256
            or not candidate.frozen_at <= validation.evaluated_at <= locked.locked_at):
        reasons.append("VALIDATION_STAGE_INVALID")
    read_pin(validation.raw_result)
    read_pin(evidence.raw_result)
    if (evidence.synthetic or evidence.scope != "full_hybrid_pipeline"
            or evidence.candidate_sha256 != bundle.candidate.sha256 or evidence.policy_sha256 != bundle.policy.sha256
            or evidence.locked_test_sha256 != bundle.locked_test.sha256
            or not locked.locked_at <= evidence.evaluated_at <= now
            or not evidence.warmup_excluded or not evidence.cuda_synchronized_if_used):
        reasons.append("FRESH_FULL_HYBRID_EVALUATION_EVIDENCE_REQUIRED")
    if {r.sample_id for r in evidence.observations} != {s.sample_id for s in locked.inventory.samples}:
        reasons.append("EVALUATION_MUST_COVER_EXACT_LOCKED_SAMPLE_SET")
    if set(policy.per_class_recall) != set(candidate.defect_classes):
        reasons.append("PER_CLASS_RECALL_CRITERIA_INCOMPLETE")
    used_tokens = legacy_exposure_tokens()
    for sample in used.samples:
        used_tokens |= exposure_tokens(sample)
        read_pin(sample.image)
        if sample.mask is not None:
            read_pin(sample.mask)
    for sample in locked.inventory.samples:
        if (sample.partition != "fresh_locked" or sample.origin != "real" or sample.fabric_id != candidate.fabric_id):
            reasons.append("LOCKED_SAMPLE_NOT_REAL_FRESH_OR_WRONG_FABRIC:" + sample.sample_id)
        overlap = exposure_tokens(sample) & used_tokens
        if overlap:
            reasons.append("SOURCE_LOT_ROLL_CAMERA_OR_HISTORY_LEAK:" + sample.sample_id)
        read_pin(sample.image)
        if sample.mask is not None:
            read_pin(sample.mask)
        if candidate.segmentation_applicable and sample.mask is None:
            reasons.append("REAL_PIXEL_MASK_REQUIRED:" + sample.sample_id)
        if sample.annotation_source != "human_verified" or sample.annotation_verified_by is None:
            reasons.append("HUMAN_VERIFIED_GROUND_TRUTH_REQUIRED:" + sample.sample_id)
    if len({s.image.sha256 for s in locked.inventory.samples}) != len(locked.inventory.samples):
        reasons.append("DUPLICATE_LOCKED_IMAGE_BYTES")
    if any(s.partition == "fresh_locked" for s in used.samples):
        reasons.append("FRESH_LOCKED_DATA_ALREADY_IN_DEVELOPMENT_INVENTORY")
    for row in evidence.observations:
        if not row.unknown_defect and row.defect_class is not None and row.defect_class not in candidate.defect_classes:
            reasons.append("EVALUATION_CLASS_OUTSIDE_CANDIDATE_TAXONOMY")
    if locked.length_calibration:
        calibration = json.loads(read_pin(locked.length_calibration, 1024 * 1024))
        if (calibration.get("unit") != "m" or not calibration.get("verified_by")
                or calibration.get("dataset_version") != locked.inventory.dataset_version
                or not {s.camera_id for s in locked.inventory.samples}.issubset(set(calibration.get("camera_ids", [])))
                or not {s.roll_id for s in locked.inventory.samples}.issubset(set(calibration.get("roll_ids", [])))):
            reasons.append("PHYSICAL_LENGTH_CALIBRATION_NOT_VERIFIED")
    if reasons:
        # Invalid provenance can never leak headline metrics as quality evidence.
        report = blocked_report(sorted(set(reasons)))
    else:
        metrics, per_class, counts = compute_metrics(evidence, locked, candidate.defect_classes, candidate.segmentation_applicable)
        checks, failed, missing = {}, [], []
        for name, rule in policy.criteria.items():
            if not rule.applicable:
                valid_na = not candidate.segmentation_applicable
                checks[name] = {"status": "NOT_APPLICABLE" if valid_na else "FAIL", "reason": rule.not_applicable_reason}
                if not valid_na:
                    failed.append(name)
            elif rule.threshold is None or metrics[name] is None:
                checks[name] = {"status": "MISSING", "actual": metrics[name], "threshold": rule.threshold}
                missing.append(name)
            else:
                passed = metrics[name] >= rule.threshold if rule.direction == "min" else metrics[name] <= rule.threshold
                checks[name] = {"status": "PASS" if passed else "FAIL", "actual": metrics[name],
                                "threshold": rule.threshold, "direction": rule.direction}
                if not passed:
                    failed.append(name)
        for name, threshold in policy.per_class_recall.items():
            actual = per_class[name]
            if threshold is None or actual is None:
                missing.append("per_class_recall:" + name)
            elif actual < threshold:
                failed.append("per_class_recall:" + name)
        accepted = not missing and not failed
        report = {**blocked_report([]), "status": "AWAITING_MANUAL_APPROVAL" if accepted else (
            "BLOCKED_BY_MISSING_EVIDENCE_OR_CRITERIA" if missing else "REJECTED_BY_ACCEPTANCE_CRITERIA"),
            "accepted": accepted, "metrics": metrics, "per_class_recall": per_class, "counts": counts,
            "metric_checks": checks, "missing": missing, "failed": failed, "production_quality_evidence": True}
    report.update(candidate_id=candidate.candidate_id, model_version=candidate.model_version,
        reference_id=candidate.reference_id, reference_version=candidate.reference_version,
        policy_version=policy.policy_version, lock_id=locked.lock_id,
        bindings=bundle.model_dump(mode="json"), bundle_sha256=bundle_pin.sha256,
        evaluated_at=evidence.evaluated_at.isoformat(), evaluator=evidence.evaluator)
    return report


def activate_manually(bundle_pin, report_pin, registry_path, approval, *, trusted_local=False, now=None):
    """Only explicit operator call; never invoked by assess/validation/runtime.

    Compare-and-swap + cooperative exclusive lock + retained backup + atomic
    replace. Run in a maintenance window: unrelated legacy writers do not take
    this lock. OS permissions/operator identity remain external trust boundaries.
    """
    now = now or datetime.now(timezone.utc)
    approval = ManualApproval.model_validate(approval) if not isinstance(approval, ManualApproval) else approval
    prior_report = json.loads(read_pin(report_pin, 16 * 1024 * 1024))
    fresh_report = assess_release(bundle_pin, trusted_local=trusted_local, now=now)
    if not fresh_report["accepted"] or canonical_json(prior_report) != canonical_json(fresh_report):
        raise ValueError("No current, reproducible, fully accepted report; activation denied")
    candidate_pin = ArtifactPin.model_validate(fresh_report["bindings"]["candidate"])
    candidate = load(candidate_pin, Candidate)
    if (approval.candidate_sha256 != candidate_pin.sha256 or approval.report_sha256 != report_pin.sha256
            or approval.reviewer.casefold() == fresh_report["evaluator"].casefold()
            or not datetime.fromisoformat(fresh_report["evaluated_at"]) <= approval.reviewed_at <= now
            or approval.confirm != f"ACTIVATE:{candidate.candidate_id}:{report_pin.sha256}"):
        raise ValueError("Explicit independent manual approval bound to candidate/report required")
    registry_path = Path(registry_path).resolve()
    lock_path = registry_path.with_name(registry_path.name + ".phase18.lock")
    lock_handle = lock_path.open("x", encoding="utf-8")
    temp_path = None
    try:
        old = read_pin(ArtifactPin(path=str(registry_path), sha256=approval.expected_registry_sha256), 16 * 1024 * 1024)
        registry = json.loads(old)
        fabric = registry["fabrics"][candidate.fabric_id]
        record = next(r for r in fabric["references"] if r["reference_id"] == candidate.reference_id)
        if (record.get("status") not in {"CANDIDATE", "INACTIVE"}
                or fabric.get("active_reference") not in {None, candidate.reference_id}
                or record.get("reference_version") != candidate.reference_version
                or record.get("backbone_version") != candidate.backbone_version
                or registry["shared_backbone"]["backbone_version"] != candidate.backbone_version
                or record.get("release_candidate_sha256") != candidate_pin.sha256
                or record.get("model_version") != candidate.model_version
                or record.get("runtime_version") != candidate.runtime_version
                or Path(record.get("segmentation_model_path", "")).resolve() != Path(candidate.model.path).resolve()):
            raise ValueError("Registry candidate/model/reference/runtime provenance mismatch")
        for key, role in (("metadata_path", "reference_metadata"), ("memory_bank_path", "memory_bank"), ("faiss_index_path", "faiss_index")):
            stored = Path(record[key])
            stored = stored if stored.is_absolute() else ROOT / stored
            if stored.resolve() != Path(candidate.artifacts[role].path).resolve():
                raise ValueError("Registry artifact path mismatch")
        for key, role in (("deployment_engine_path", "deployment_engine"), ("runtime_recipe_path", "runtime_recipe")):
            if key not in record or Path(record[key]).resolve() != Path(candidate.artifacts[role].path).resolve():
                raise ValueError("Registry engine/runtime recipe provenance mismatch")
        if canonical_json(record["thresholds"]) != canonical_json(json.loads(read_pin(candidate.artifacts["thresholds"]))):
            raise ValueError("Frozen reference thresholds changed")
        # No auto-registration, replacement of artifacts or threshold edits.
        record["status"] = "ACTIVE"
        fabric["status"], fabric["active_reference"] = "ACTIVE", candidate.reference_id
        audit = {"candidate_sha256": candidate_pin.sha256, "report_sha256": report_pin.sha256,
                 "approval": approval.model_dump(mode="json"), "activated_at": now.isoformat()}
        registry.setdefault("release_activation_audit", []).append(audit)
        registry["updated_at"] = now.isoformat()
        backup = registry_path.with_name(registry_path.name + ".phase18-" + uuid.uuid4().hex + ".backup")
        with backup.open("xb") as stream:
            stream.write(old)
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.NamedTemporaryFile(mode="wb", dir=registry_path.parent, prefix=".phase18-", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(canonical_json(registry))
            stream.flush()
            os.fsync(stream.fileno())
        read_pin(ArtifactPin(path=str(registry_path), sha256=approval.expected_registry_sha256), 16 * 1024 * 1024)
        os.replace(temp_path, registry_path)
        temp_path = None
        return {"activated": True, "registry": str(registry_path), "backup": str(backup), "audit": audit}
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)  # Only our owned temporary file.
        lock_handle.close()
        lock_path.unlink()
