from dataclasses import dataclass

from backend.app.ai.anomaly_detector import PatchAnomaly


@dataclass(frozen=True)
class MergedBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    patch_count: int = 1
    label: str = "suspicious_defect"

    @property
    def area(self) -> int:
        return self.w * self.h


def merge_suspicious_patches(patches: list[PatchAnomaly], merge_gap: int = 0) -> list[MergedBox]:
    suspicious = [patch for patch in patches if patch.suspicious]
    if not suspicious:
        return []

    remaining = set(range(len(suspicious)))
    components: list[list[PatchAnomaly]] = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [suspicious[start]]

        while stack:
            current_index = stack.pop()
            current = suspicious[current_index]
            neighbors = [
                candidate_index
                for candidate_index in list(remaining)
                if _touches_or_overlaps(current, suspicious[candidate_index], merge_gap)
            ]
            for neighbor_index in neighbors:
                remaining.remove(neighbor_index)
                stack.append(neighbor_index)
                component.append(suspicious[neighbor_index])

        components.append(component)

    return [_component_to_box(component) for component in components]


def _touches_or_overlaps(left: PatchAnomaly, right: PatchAnomaly, gap: int) -> bool:
    left_x2 = left.box.x + left.box.w
    left_y2 = left.box.y + left.box.h
    right_x2 = right.box.x + right.box.w
    right_y2 = right.box.y + right.box.h

    separated = (
        left_x2 + gap < right.box.x
        or right_x2 + gap < left.box.x
        or left_y2 + gap < right.box.y
        or right_y2 + gap < left.box.y
    )
    return not separated


def _component_to_box(component: list[PatchAnomaly]) -> MergedBox:
    x1 = min(item.box.x for item in component)
    y1 = min(item.box.y for item in component)
    x2 = max(item.box.x + item.box.w for item in component)
    y2 = max(item.box.y + item.box.h for item in component)
    return MergedBox(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        score=max(item.anomaly_score for item in component),
        patch_count=len(component),
    )


def scale_boxes_to_original(boxes: list[MergedBox], scale: float) -> list[MergedBox]:
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    if scale == 1.0:
        return boxes
    inverse = 1.0 / scale
    return [
        MergedBox(
            x=round(box.x * inverse),
            y=round(box.y * inverse),
            w=max(1, round(box.w * inverse)),
            h=max(1, round(box.h * inverse)),
            score=box.score,
            patch_count=box.patch_count,
            label=box.label,
        )
        for box in boxes
    ]


def filter_boxes_by_cluster_size(boxes: list[MergedBox], min_patch_count: int) -> list[MergedBox]:
    if min_patch_count <= 1:
        return boxes
    return [box for box in boxes if box.patch_count >= min_patch_count]
