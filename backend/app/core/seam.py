"""Calibrated seam grouping only; raw observations retained, no business QC rules."""

from datetime import datetime, timezone
import hashlib
import json
from itertools import combinations
from typing import Literal

from pydantic import Field, field_validator, model_validator

from backend.app.core.physical_coordinates import (CoordinateContract, CoordinateMappingResult, FrameCoordinates,
                                                   PhysicalCalibration, PhysicalPoint, PixelCandidate)


class SeamPolicy(CoordinateContract):
    policy_version: str = Field(default="seam_unapproved_starting_rules_v1", min_length=1, max_length=128)
    adjacent_camera_pairs: tuple[tuple[str, str], ...] = ()  # Never infer adjacency from a camera name/order.
    minimum_seam_iou: float = Field(default=0.5, gt=0, le=1, strict=True)
    maximum_center_distance_mm: float = Field(default=1.0, gt=0, le=1000, strict=True)
    maximum_time_delta_ms: float = Field(default=0.0, ge=0, le=1000, strict=True)
    max_observations: int = Field(default=128, ge=1, le=256, strict=True)

    @model_validator(mode="after")
    def explicit_unique_adjacency(self):
        seen = set()
        for pair in self.adjacent_camera_pairs:
            if (pair[0] == pair[1] or any(not item.strip() or item != item.strip() or len(item) > 128 for item in pair)
                    or tuple(sorted(pair)) in seen):
                raise ValueError("Distinct, clean, unique unordered adjacent camera pairs required")
            seen.add(tuple(sorted(pair)))
        return self


class SeamObservation(CoordinateContract):
    capture_group_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    source_kind: Literal["supervised", "anomaly"]
    model_version: str = Field(min_length=1, max_length=128)
    reference_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frame: FrameCoordinates
    candidate: PixelCandidate

    @field_validator("timestamp", mode="before")
    @classmethod
    def explicit_clock(cls, value):
        if not isinstance(value, (datetime, str)):
            raise ValueError("Use timezone-aware datetime/ISO timestamp")
        return value

    @field_validator("timestamp")
    @classmethod
    def aware_clock(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Seam matching requires an explicit timezone-aware timestamp")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def traceable_branch(self):
        if self.source_kind == "anomaly" and self.reference_version is None:
            raise ValueError("Anomaly observation requires reference version")
        return self

    @property
    def observation_id(self):
        values = (self.capture_group_id, self.frame.fabric_id, self.frame.lot_id, self.frame.roll_id,
                  self.frame.camera_id, self.frame.frame_id, self.source_kind, self.candidate.candidate_id)
        return "obs_" + hashlib.sha256(json.dumps(values, ensure_ascii=True).encode("utf-8")).hexdigest()


class SeamPairAssessment(CoordinateContract):
    observation_ids: tuple[str, str]
    eligible: bool = Field(strict=True)
    reason: str = Field(pattern=r"^[A-Z0-9_]+$")
    seam_iou: float | None = Field(default=None, ge=0, le=1, strict=True)
    center_distance_mm: float | None = Field(default=None, ge=0, strict=True)
    review_required: bool = Field(default=False, strict=True)


class SeamGroup(CoordinateContract):
    group_id: str = Field(min_length=1)
    member_ids: tuple[str, ...] = Field(min_length=1)
    merged: bool = Field(strict=True)
    label: str = Field(min_length=1)
    minimum: PhysicalPoint | None = None
    maximum: PhysicalPoint | None = None
    geometry_kind: Literal["bbox_envelope_not_mask_union"] = "bbox_envelope_not_mask_union"
    review_required: bool = Field(strict=True)
    reasons: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def truthful_group(self):
        if len(set(self.member_ids)) != len(self.member_ids) or self.merged != (len(self.member_ids) > 1):
            raise ValueError("Group membership/merged status must agree")
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("Physical bounds must be present together")
        if self.minimum is not None and (self.maximum.x_mm <= self.minimum.x_mm or self.maximum.y_mm <= self.minimum.y_mm):
            raise ValueError("Positive physical bbox required")
        if self.merged and (self.minimum is None or self.review_required):
            raise ValueError("Uncalibrated/ambiguous groups must not be merged")
        return self


class SeamMergeResult(CoordinateContract):
    status: Literal["evaluated", "unavailable"]
    policy: SeamPolicy
    calibration: PhysicalCalibration | None = None
    observations: tuple[SeamObservation, ...]
    mappings: tuple[CoordinateMappingResult, ...]
    groups: tuple[SeamGroup, ...]
    pairs: tuple[SeamPairAssessment, ...]
    issues: tuple[str, ...] = ()
    affects_qc: Literal[False] = False

    @field_validator("affects_qc", mode="before")
    @classmethod
    def no_business_activation(cls, value):
        if value is not False:
            raise ValueError("Seam grouping never activates business QC")
        return value

    @model_validator(mode="after")
    def retain_exact_lineage(self):
        keys = [item.observation_id for item in self.observations]
        members = [key for group in self.groups for key in group.member_ids]
        if len(keys) != len(set(keys)) or sorted(keys) != sorted(members) or len(self.mappings) != len(keys):
            raise ValueError("Every unique observation must be retained exactly once")
        by_id = {item.observation_id: (item, mapping) for item, mapping in zip(self.observations, self.mappings)}
        pair_keys = [tuple(sorted(pair.observation_ids)) for pair in self.pairs]
        if (len(set(pair_keys)) != len(pair_keys) or set(pair_keys) != set(combinations(sorted(keys), 2))
                or len({group.group_id for group in self.groups}) != len(self.groups)):
            raise ValueError("Complete unique pair audit and group IDs required")
        for item, mapping in zip(self.observations, self.mappings):
            if mapping.candidate is not None and (mapping.frame != item.frame or mapping.candidate.source != item.candidate
                                                  or mapping.calibration != self.calibration):
                raise ValueError("Mapping must retain exact observation/calibration lineage")
        eligible = {tuple(sorted(pair.observation_ids)) for pair in self.pairs if pair.eligible}
        for pair in self.pairs:
            if pair.eligible and (pair.review_required or pair.seam_iou is None or pair.center_distance_mm is None
                    or pair.seam_iou < self.policy.minimum_seam_iou
                    or pair.center_distance_mm > self.policy.maximum_center_distance_mm):
                raise ValueError("Eligible pair must retain passing, non-ambiguous geometry evidence")
        if self.status == "unavailable" and any(group.merged for group in self.groups):
            raise ValueError("Unavailable merger must not suppress duplicates")
        for group in self.groups:
            values = [by_id[key] for key in group.member_ids]
            if any(item.candidate.label != group.label for item, _ in values):
                raise ValueError("Group cannot hide conflicting classes")
            physical = [mapping.candidate for _, mapping in values]
            if group.merged:
                reference = values[0][0]
                scope = lambda item: (item.capture_group_id, item.frame.fabric_id, item.frame.lot_id, item.frame.roll_id,
                                      item.frame.frame_index, item.frame.position)
                adjacency = {tuple(sorted(pair)) for pair in self.policy.adjacent_camera_pairs}
                if (any(not item.frame.lot_id or not item.frame.roll_id or scope(item) != scope(reference) for item, _ in values)
                        or (max(item.timestamp for item, _ in values) - min(item.timestamp for item, _ in values)).total_seconds() * 1000 > self.policy.maximum_time_delta_ms
                        or any(tuple(sorted((left.frame.camera_id, right.frame.camera_id))) not in adjacency
                               for (left, _), (right, _) in combinations(values, 2))
                        or any(pair.review_required and set(pair.observation_ids).intersection(group.member_ids) for pair in self.pairs)):
                    raise ValueError("Merged group requires verified scope/time/adjacency and no conflict warnings")
            if group.merged and (self.calibration is None or any(box is None for box in physical)
                    or len({item.frame.camera_id for item, _ in values}) != len(values)
                    or len({(item.source_kind, item.model_version, item.model_sha256, item.reference_version) for item, _ in values}) != 1
                    or any(tuple(sorted(pair)) not in eligible for pair in combinations(group.member_ids, 2))):
                raise ValueError("Merged members require complete-link calibrated cross-camera evidence")
            if any(box is None for box in physical):
                if group.minimum is not None:
                    raise ValueError("Unmapped input cannot claim physical bounds")
            elif (group.minimum is None or group.minimum.x_mm != min(box.minimum.x_mm for box in physical)
                    or group.minimum.y_mm != min(box.minimum.y_mm for box in physical)
                    or group.maximum.x_mm != max(box.maximum.x_mm for box in physical)
                    or group.maximum.y_mm != max(box.maximum.y_mm for box in physical)):
                raise ValueError("Group envelope must preserve mapped source extents")
        return self
