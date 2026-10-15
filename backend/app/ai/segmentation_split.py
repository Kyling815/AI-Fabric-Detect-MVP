"""Connected-component source/roll/lot/camera split, never per-patch random split."""

from dataclasses import dataclass
import random

from backend.app.core.segmentation_data import GroupedSplitConfig, SegmentationManifest


@dataclass(frozen=True)
class GroupedSplitResult:
    assignments: dict[str, str]
    groups: tuple[tuple[str, ...], ...]
    sample_counts: dict[str, int]
    requested_ratios: dict[str, float]
    warnings: tuple[str, ...]


def grouped_segmentation_split(manifest: SegmentationManifest, config: GroupedSplitConfig) -> GroupedSplitResult:
    samples = sorted(manifest.samples, key=lambda sample: sample.sample_id)
    parent = list(range(len(samples)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)
    keys = {}
    missing_roll = False
    for index, sample in enumerate(samples):
        protected = sample.split in {"historical_locked", "excluded"}
        if not protected and config.require_roll_id and sample.roll_id is None:
            raise ValueError("Verified roll_id required; source-only split cannot prove roll disjointness")
        if not protected and config.group_by_lot and sample.lot_id is None:
            raise ValueError("Explicit lot grouping requires lot_id on every eligible sample")
        if not protected and config.group_by_camera and sample.camera_id is None:
            raise ValueError("Explicit camera grouping requires camera_id on every eligible sample")
        missing_roll |= sample.roll_id is None
        tokens = [("source", source) for source in (sample.source_id, *sample.ancestor_source_ids)]
        tokens += [("source_hash", value) for value in (sample.image_sha256, sample.source_sha256, *sample.ancestor_source_sha256s) if value]
        if sample.roll_id is not None:
            tokens.append(("roll", sample.fabric_id, sample.roll_id))
        if config.group_by_lot and sample.lot_id is not None:
            tokens.append(("lot", sample.fabric_id, sample.lot_id))
        if config.group_by_camera and sample.camera_id is not None:
            tokens.append(("camera", sample.camera_id))
        for token in tokens:
            if token in keys:
                union(index, keys[token])
            else:
                keys[token] = index
    components = {}
    for index, sample in enumerate(samples):
        components.setdefault(find(index), []).append(sample)
    groups = [tuple(group) for _, group in sorted(components.items())]
    ratios = {"train": config.train_ratio, "validation": config.validation_ratio, "test": config.test_ratio}
    active = [split for split, ratio in ratios.items() if ratio > 0]
    assignments, pending = {}, []
    counts = {split: 0 for split in (*ratios, "historical_locked", "excluded")}
    group_counts = {split: 0 for split in active}
    eligible_groups = 0
    for group in groups:
        protected = any(sample.split in {"historical_locked", "excluded"} for sample in group)
        pins = {sample.split for sample in group if sample.split in ratios}
        if protected:
            if pins:
                raise ValueError("Protected historical/excluded source component is pinned to a training/evaluation split")
            for sample in group:
                split = "historical_locked" if sample.split == "historical_locked" else "excluded"
                assignments[sample.sample_id] = split
                counts[split] += 1
            continue
        if len(pins) > 1:
            raise ValueError("Existing split leaks a source/roll/lot/camera component across partitions")
        eligible_groups += 1
        if pins:
            split = next(iter(pins))
            if ratios[split] == 0:
                raise ValueError("Pinned split conflicts with requested zero ratio")
            for sample in group:
                assignments[sample.sample_id] = split
            counts[split] += len(group)
            group_counts[split] += 1
        else:
            pending.append(group)
    if eligible_groups < len(active):
        raise ValueError("Not enough independent source/roll groups for all requested nonempty partitions")
    total = sum(counts[split] for split in ratios) + sum(len(group) for group in pending)
    rng = random.Random(config.seed)
    pending = sorted([(group, rng.random()) for group in pending], key=lambda value: (-len(value[0]), value[1]))
    for index, (group, _) in enumerate(pending):
        empty = [split for split in active if group_counts[split] == 0]
        candidates = empty if len(pending) - index <= len(empty) else active
        split = max(candidates, key=lambda name: (ratios[name] * total - counts[name], -active.index(name)))
        for sample in group:
            assignments[sample.sample_id] = split
        counts[split] += len(group)
        group_counts[split] += 1
    if any(group_counts[split] == 0 for split in active):
        raise ValueError("Pinned groups leave a requested partition empty; cannot weaken disjointness")
    warnings = ["Ratios are approximate because entire connected groups are indivisible."]
    if missing_roll:
        warnings.append("Roll disjointness is unverified for missing roll metadata; no roll ids were inferred.")
    if any(sample.source_sha256 is None for sample in samples):
        warnings.append("Original source content is not fully verified where root-source hashes are absent; lineage ids remain caller-declared.")
    return GroupedSplitResult(assignments, tuple(tuple(sample.sample_id for sample in group) for group in groups),
                              counts, ratios, tuple(warnings))
