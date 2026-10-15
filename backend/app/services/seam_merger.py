"""Conservative physical bbox matching; no mask fabrication, NMS score selection or QC."""

import hashlib
from itertools import combinations
import json
import math

from backend.app.core.physical_coordinates import PhysicalPoint
from backend.app.core.seam import SeamGroup, SeamMergeResult, SeamObservation, SeamPairAssessment, SeamPolicy
from backend.app.services.coordinate_mapper import CalibratedCoordinateMapper, camera_overlaps


def _clipped(box, overlap):
    low, high = max(box.minimum.x_mm, overlap.x_min_mm), min(box.maximum.x_mm, overlap.x_max_mm)
    return (low, box.minimum.y_mm, high, box.maximum.y_mm) if low < high else None


def _metrics(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area, right_area = (left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    distance = math.hypot((left[0] - right[0]) / 2 + (left[2] - right[2]) / 2,
                          (left[1] - right[1]) / 2 + (left[3] - right[3]) / 2)
    if not all(math.isfinite(value) for value in (intersection, left_area, right_area, union, distance)) or union <= 0:
        raise ValueError("Unusable numeric geometry")
    return min(1.0, max(0.0, intersection / union)), distance


class SeamMerger:
    def __init__(self, calibration, policy: SeamPolicy | None = None):
        self.mapper = CalibratedCoordinateMapper(calibration)
        self.policy = SeamPolicy.model_validate((policy or SeamPolicy()).model_dump())

    def _assess(self, left, right, left_map, right_map, overlaps, adjacency, policy):
        ids = tuple(sorted((left.observation_id, right.observation_id)))
        def result(reason, *, eligible=False, review=False, iou=None, distance=None):
            return SeamPairAssessment(observation_ids=ids, reason=reason, eligible=eligible, review_required=review,
                                      seam_iou=iou, center_distance_mm=distance)
        if left.frame.camera_id == right.frame.camera_id:
            return result("SAME_CAMERA")
        cameras = tuple(sorted((left.frame.camera_id, right.frame.camera_id)))
        if cameras not in adjacency:
            return result("CAMERAS_NOT_DECLARED_ADJACENT")
        if left_map.candidate is None or right_map.candidate is None:
            return result("PHYSICAL_MAPPING_UNAVAILABLE", review=True)
        identity = lambda item: (item.capture_group_id, item.frame.fabric_id, item.frame.lot_id, item.frame.roll_id, item.frame.frame_index)
        if not left.frame.lot_id or not right.frame.lot_id or not left.frame.roll_id or not right.frame.roll_id:
            return result("PRODUCTION_IDENTITY_UNVERIFIED", review=True)
        if identity(left) != identity(right):
            return result("CAPTURE_SCOPE_MISMATCH")
        if (left.frame.position != right.frame.position
                or abs((left.timestamp - right.timestamp).total_seconds()) * 1000 > policy.maximum_time_delta_ms):
            return result("CAPTURE_SYNCHRONIZATION_UNVERIFIED", review=True)
        overlap = overlaps.get(cameras)
        if overlap is None:
            return result("NO_CALIBRATED_SHARED_FOV")
        left_box, right_box = _clipped(left_map.candidate, overlap), _clipped(right_map.candidate, overlap)
        if left_box is None or right_box is None:
            return result("OUTSIDE_SHARED_FOV")
        try:
            iou, distance = _metrics(left_box, right_box)
        except (ValueError, OverflowError):
            return result("NUMERIC_GEOMETRY_UNAVAILABLE", review=True)
        if iou < policy.minimum_seam_iou or distance > policy.maximum_center_distance_mm:
            return result("GEOMETRY_NOT_DUPLICATE", iou=iou, distance=distance)
        if left.source_kind != right.source_kind or left.candidate.label != right.candidate.label:
            return result("CLASS_OR_BRANCH_CONFLICT", review=True, iou=iou, distance=distance)
        provenance = lambda item: (item.model_version, item.model_sha256, item.reference_version)
        if provenance(left) != provenance(right):
            return result("MODEL_OR_REFERENCE_MISMATCH", review=True, iou=iou, distance=distance)
        return result("CALIBRATED_SEAM_DUPLICATE", eligible=True, iou=iou, distance=distance)

    def merge(self, observations) -> SeamMergeResult:
        policy = SeamPolicy.model_validate(self.policy.model_dump())
        if not isinstance(observations, (list, tuple)) or len(observations) > policy.max_observations:
            raise ValueError("Explicit bounded observation list/tuple required; never truncate input")
        if any(not isinstance(item, SeamObservation) for item in observations):
            raise ValueError("Typed seam observations required; invalid batches are rejected, never filtered")
        observations = tuple(sorted((SeamObservation.model_validate(item.model_dump()) for item in observations), key=lambda item: item.observation_id))
        ids = [item.observation_id for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate observation identity; caller must disambiguate source candidate IDs")
        # Rebuild from calibration snapshot; never trust caller-declared physical rectangles.
        mapper = CalibratedCoordinateMapper(self.mapper.calibration) if self.mapper.error is None else self.mapper
        calibration, issues = mapper.calibration, []
        if mapper.error:
            issues.append(mapper.error)
        adjacency = {tuple(sorted(pair)) for pair in policy.adjacent_camera_pairs}
        if calibration is not None and any(camera not in {item.camera_id for item in calibration.cameras} for pair in adjacency for camera in pair):
            issues.append("ADJACENCY_CAMERA_NOT_CALIBRATED")
        overlaps = {} if calibration is None else {tuple(sorted(item.camera_ids)): item for item in camera_overlaps(calibration)}
        mappings = tuple(mapper.map_candidate(item.frame, item.candidate) for item in observations)
        pairs = []
        graph, warnings = {key: set() for key in ids}, {key: set(issues) for key in ids}
        for item, mapping in zip(observations, mappings):
            if mapping.candidate is None:
                warnings[item.observation_id].add(mapping.reason)
            if not item.frame.lot_id or not item.frame.roll_id:
                warnings[item.observation_id].add("PRODUCTION_IDENTITY_UNVERIFIED")
        for i, j in combinations(range(len(observations)), 2):
            pair = self._assess(observations[i], observations[j], mappings[i], mappings[j], overlaps, adjacency, policy)
            pairs.append(pair)
            if pair.eligible and not issues:
                graph[ids[i]].add(ids[j])
                graph[ids[j]].add(ids[i])
            if pair.review_required:
                for key in pair.observation_ids:
                    warnings[key].add(pair.reason)
        by_id = {item.observation_id: (item, mapping) for item, mapping in zip(observations, mappings)}
        groups, remaining = [], set(ids)
        while remaining:
            root, component, pending = min(remaining), set(), []
            pending.append(root)
            while pending:
                key = pending.pop()
                if key not in component:
                    component.add(key)
                    pending.extend(graph[key] - component)
            remaining -= component
            cameras = [by_id[key][0].frame.camera_id for key in component]
            complete = (len(set(cameras)) == len(cameras) and not any(warnings[key] for key in component)
                        and all(right in graph[left] for left, right in combinations(sorted(component), 2)))
            if len(component) > 1 and not complete:
                for key in component:
                    warnings[key].add("AMBIGUOUS_MATCH_SET")
            partitions = [tuple(sorted(component))] if len(component) > 1 and complete else [(key,) for key in sorted(component)]
            for members in partitions:
                values = [by_id[key] for key in members]
                boxes = [mapping.candidate for _, mapping in values]
                physical = all(box is not None for box in boxes)
                review = sorted(set().union(*(warnings[key] for key in members)))
                digest = json.dumps([policy.model_dump(mode="json"), calibration.model_dump(mode="json") if calibration else None, members], sort_keys=True)
                groups.append(SeamGroup(group_id="seam_" + hashlib.sha256(digest.encode("utf-8")).hexdigest(), member_ids=members,
                    merged=len(members) > 1, label=values[0][0].candidate.label, review_required=bool(review),
                    reasons=tuple(review or (["CALIBRATED_SEAM_DUPLICATE"] if len(members) > 1 else ["NO_UNAMBIGUOUS_DUPLICATE"])),
                    minimum=PhysicalPoint(x_mm=min(box.minimum.x_mm for box in boxes), y_mm=min(box.minimum.y_mm for box in boxes)) if physical else None,
                    maximum=PhysicalPoint(x_mm=max(box.maximum.x_mm for box in boxes), y_mm=max(box.maximum.y_mm for box in boxes)) if physical else None))
        return SeamMergeResult(status="unavailable" if issues else "evaluated", policy=policy, calibration=calibration,
            observations=observations, mappings=mappings, groups=tuple(sorted(groups, key=lambda item: item.member_ids)),
            pairs=tuple(pairs), issues=tuple(issues))


def seam_observations_from_chunks(bundle, candidates_by_camera, *, source_kind, model_version,
                                  reference_version=None, model_sha256=None):
    """Explicit offline Phase 13 bridge; no detector invocation or physical-unit inference."""
    from backend.app.core.line_scan import ChunkBundle
    from backend.app.core.physical_coordinates import PixelCandidate
    bundle = ChunkBundle.model_validate(bundle.model_dump())
    camera_ids = {chunk.camera_id for chunk in bundle.chunks}
    if not isinstance(candidates_by_camera, dict) or not set(candidates_by_camera).issubset(camera_ids):
        raise ValueError("Candidate keys must refer to the exact chunk bundle camera IDs")
    if (any(not isinstance(values, (list, tuple)) for values in candidates_by_camera.values())
            or sum(len(values) for values in candidates_by_camera.values()) > 256):
        raise ValueError("Explicit bounded candidate lists required; never truncate candidates")
    first = bundle.chunks[0]
    cohort = [first.simulation_id, first.fabric_id, first.lot_id, first.roll_id, first.chunk_index,
              first.start_line, first.encoder_id, first.encoder_start, first.encoder_end_exclusive,
              first.timestamp.isoformat(), first.timestamp_end_exclusive.isoformat()]
    capture_id = "capture_" + hashlib.sha256(json.dumps(cohort).encode("utf-8")).hexdigest()
    results = []
    for chunk in bundle.chunks:
        for candidate in candidates_by_camera.get(chunk.camera_id, ()):
            if not isinstance(candidate, PixelCandidate):
                raise ValueError("Typed oriented-source pixel candidates required")
            candidate = PixelCandidate.model_validate(candidate.model_dump())
            results.append(SeamObservation(capture_group_id=capture_id, timestamp=chunk.timestamp, source_kind=source_kind,
                model_version=model_version, reference_version=reference_version, model_sha256=model_sha256,
                frame=chunk.to_frame_coordinates(), candidate=candidate))
    return tuple(results)
