from dataclasses import dataclass
import math

import cv2
import numpy as np

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


def localize_suspicious_patches(
    patches: list[PatchAnomaly],
    *,
    image_width: int,
    image_height: int,
    min_patch_count: int = 1,
) -> list[MergedBox]:
    """Convert patch anomaly scores into compact defect regions.

    A plain overlap merge can turn a dense, overlapping patch grid into one
    full-frame rectangle. When at least half of the grid is above the absolute
    anomaly threshold, use robust, image-relative hysteresis thresholds:

    * strong outlier patches seed a defect;
    * moderately elevated neighboring patches grow that defect;
    * elevated components without a strong seed are treated as background
      texture/domain shift.

    Output bounds use the Voronoi-like cells around patch centers instead of
    the full overlapping patch footprints. This keeps localization at the grid
    stride resolution while still covering the complete processed image at its
    borders.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not patches:
        return []

    suspicious = [patch for patch in patches if patch.suspicious]
    if not suspicious:
        return []

    candidates = suspicious
    strong_patch_indexes: set[int] | None = None
    suspicious_ratio = len(suspicious) / len(patches)
    anomaly_scores = np.asarray([patch.anomaly_score for patch in patches], dtype=np.float64)

    if len(patches) >= 9 and suspicious_ratio >= 0.5 and _has_localizable_score_contrast(anomaly_scores):
        median = float(np.median(anomaly_scores))
        mad = float(np.median(np.abs(anomaly_scores - median)))
        robust_sigma = 1.4826 * mad
        low_threshold = max(
            float(np.quantile(anomaly_scores, 0.75)),
            median + robust_sigma,
        )
        high_threshold = min(
            float(np.max(anomaly_scores)),
            max(
                float(np.quantile(anomaly_scores, 0.90)),
                median + 2.5 * robust_sigma,
            ),
        )
        candidates = [
            patch for patch in suspicious if patch.anomaly_score >= low_threshold
        ]
        strong_patch_indexes = {
            patch.patch_index
            for patch in suspicious
            if patch.anomaly_score >= high_threshold
        }

    components = _overlapping_components(candidates)
    if strong_patch_indexes is not None:
        components = [
            component
            for component in components
            if any(patch.patch_index in strong_patch_indexes for patch in component)
        ]

    minimum = max(1, min_patch_count)
    components = [component for component in components if len(component) >= minimum]
    if not components:
        return []

    x_centers = sorted({_center_x(patch) for patch in patches})
    y_centers = sorted({_center_y(patch) for patch in patches})
    x_cells = _axis_cells(x_centers, image_width)
    y_cells = _axis_cells(y_centers, image_height)
    return [
        _component_to_tight_box(component, x_cells=x_cells, y_cells=y_cells)
        for component in components
    ]


def localize_pixel_defects(
    image_rgb: np.ndarray,
    *,
    anomaly_score: float,
) -> list[MergedBox]:
    """Find compact high-contrast defects when patch scoring activates globally.

    CNN patch embeddings are useful for deciding that an image differs from its
    reference, but a domain-shifted fabric can make every overlapping patch
    suspicious. In that case patch boxes contain no spatial information. This
    fallback estimates the slowly varying fabric background, thresholds only
    the strongest local residuals, and retains dominant connected components.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Pixel localization expects an RGB image.")
    height, width = image_rgb.shape[:2]
    if width <= 0 or height <= 0:
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(3.0, min(width, height) / 80.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = cv2.absdiff(gray, background)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual.astype(np.float32) - median)))
    robust_threshold = median + 4.0 * 1.4826 * mad
    threshold = max(float(np.quantile(residual, 0.998)), robust_threshold)
    mask = (residual >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(4, math.ceil(width * height * 0.00002))
    maximum_area = width * height * 0.10
    components: list[tuple[int, int, int, int, int, int]] = []
    for component_id in range(1, component_count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component_id])
        if area < minimum_area or area > maximum_area:
            continue
        peak = int(residual[labels == component_id].max())
        components.append((x, y, box_width, box_height, area, peak))
    largest_area = max((component[4] for component in components), default=0)
    retained_area = max(minimum_area, math.ceil(largest_area * 0.10))
    padding = max(3, round(min(width, height) * 0.007))
    point_boxes: list[MergedBox] = []
    for x, y, box_width, box_height, area, peak in components:
        if area < retained_area:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_width + padding)
        y2 = min(height, y + box_height + padding)
        point_boxes.append(
            MergedBox(
                x=x1,
                y=y1,
                w=max(1, x2 - x1),
                h=max(1, y2 - y1),
                score=max(anomaly_score, peak / 255.0),
                patch_count=1,
                label="pixel_localized_defect",
            )
        )

    elongated_boxes = _localize_axis_aligned_defects(
        gray,
        anomaly_score=anomaly_score,
        point_boxes=point_boxes,
    )
    boxes = elongated_boxes or point_boxes
    return sorted(boxes, key=lambda box: (box.y, box.x))


def _localize_axis_aligned_defects(
    gray: np.ndarray,
    *,
    anomaly_score: float,
    point_boxes: list[MergedBox],
) -> list[MergedBox]:
    """Localize long yarn/row defects that a point residual mask fragments.

    End-out, mixed-yarn and float defects often change the texture along one
    complete yarn while only a few individual pixels exceed a 2-D residual
    threshold.  Collapsing horizontal-gradient and column/row-variance maps
    produces a stable one-dimensional signal for these defects.  Robust
    hysteresis keeps only bands with a >=3-sigma interior seed; this threshold
    is relative to the uploaded image, so changes in lighting and fabric color
    do not directly change it.
    """

    height, width = gray.shape
    gray_float = gray.astype(np.float32)
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    vertical = _merge_axis_bands(
        [
            *_axis_outlier_bands(gradient_x.mean(axis=0)),
            *_axis_outlier_bands(gray_float.std(axis=0)),
        ],
        axis_length=width,
    )
    horizontal = _merge_axis_bands(
        [
            *_axis_outlier_bands(gradient_y.mean(axis=1)),
            *_axis_outlier_bands(gray_float.std(axis=1)),
        ],
        axis_length=height,
    )
    if not vertical and not horizontal:
        return []

    padding = max(3, round(min(width, height) * 0.01))
    boxes: list[MergedBox] = []

    # A compact defect raises both axis profiles.  Their intersection is much
    # tighter than returning a full-width and a full-height stripe.
    if vertical and horizontal:
        for x1, x2, x_score in vertical:
            for y1, y2, y_score in horizontal:
                boxes.append(
                    _padded_axis_box(
                        x1,
                        y1,
                        x2,
                        y2,
                        width=width,
                        height=height,
                        padding=padding,
                        score=anomaly_score,
                    )
                )
        return boxes

    if vertical:
        for x1, x2, band_score in vertical:
            aligned = [
                box
                for box in point_boxes
                if x1 - padding <= box.x + box.w / 2.0 <= x2 + padding
            ]
            if len(aligned) >= 2:
                y1 = min(box.y for box in aligned)
                y2 = max(box.y + box.h for box in aligned)
            else:
                y1, y2 = 0, height
            boxes.append(
                _padded_axis_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    width=width,
                    height=height,
                    padding=padding,
                    score=anomaly_score,
                )
            )
        return boxes

    for y1, y2, band_score in horizontal:
        aligned = [
            box
            for box in point_boxes
            if y1 - padding <= box.y + box.h / 2.0 <= y2 + padding
        ]
        if len(aligned) >= 2:
            x1 = min(box.x for box in aligned)
            x2 = max(box.x + box.w for box in aligned)
        else:
            x1, x2 = 0, width
        boxes.append(
            _padded_axis_box(
                x1,
                y1,
                x2,
                y2,
                width=width,
                height=height,
                padding=padding,
                score=anomaly_score,
            )
        )
    return boxes


def _axis_outlier_bands(signal: np.ndarray) -> list[tuple[int, int, float]]:
    """Return low-threshold bands containing a strong robust-z seed."""

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    length = int(values.size)
    if length < 16:
        return []
    smoothed = cv2.GaussianBlur(
        values.reshape(1, -1),
        (0, 0),
        sigmaX=max(1.0, length / 200.0),
    ).reshape(-1)
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    robust_scale = 1.4826 * mad
    if robust_scale <= 1e-6:
        return []
    # A long, coherent defect must change the axis profile materially.  This
    # rejects high z-scores caused only by the tiny sampling variance of an
    # otherwise uniform random texture.
    z_scores = (smoothed - median) / robust_scale
    peak_delta = float(np.max(smoothed)) - median
    if peak_delta < 8.0 and float(np.max(z_scores)) < 5.0:
        return []

    edge = max(2, round(length * 0.04))
    strong = z_scores >= 3.0
    strong[:edge] = False
    strong[length - edge :] = False
    candidate = z_scores >= 1.5

    bands: list[tuple[int, int, float]] = []
    start = 0
    while start < length:
        if not candidate[start]:
            start += 1
            continue
        end = start + 1
        while end < length and candidate[end]:
            end += 1
        if strong[start:end].any() and end - start <= max(4, round(length * 0.20)):
            bands.append((start, end, float(z_scores[start:end].max())))
        start = end
    return bands


def _merge_axis_bands(
    bands: list[tuple[int, int, float]],
    *,
    axis_length: int,
) -> list[tuple[int, int, float]]:
    if not bands:
        return []
    gap = max(2, round(axis_length * 0.025))
    merged: list[tuple[int, int, float]] = []
    for start, end, score in sorted(bands):
        if merged and start <= merged[-1][1] + gap:
            old_start, old_end, old_score = merged[-1]
            merged[-1] = (old_start, max(old_end, end), max(old_score, score))
        else:
            merged.append((start, end, score))

    strongest = max(score for _, _, score in merged)
    minimum_score = max(3.0, strongest * 0.70)
    return [band for band in merged if band[2] >= minimum_score][:3]


def _padded_axis_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    width: int,
    height: int,
    padding: int,
    score: float,
) -> MergedBox:
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(width, x2 + padding)
    bottom = min(height, y2 + padding)
    return MergedBox(
        x=left,
        y=top,
        w=max(1, right - left),
        h=max(1, bottom - top),
        score=score,
        patch_count=1,
        label="elongated_pixel_defect",
    )


def _has_localizable_score_contrast(scores: np.ndarray) -> bool:
    if scores.size < 2:
        return False
    median = float(np.median(scores))
    required_span = max(0.03, abs(median) * 0.05)
    return float(np.max(scores) - np.min(scores)) >= required_span


def _overlapping_components(patches: list[PatchAnomaly]) -> list[list[PatchAnomaly]]:
    if not patches:
        return []
    remaining = set(range(len(patches)))
    components: list[list[PatchAnomaly]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [patches[start]]
        while stack:
            current = patches[stack.pop()]
            neighbors = [
                candidate_index
                for candidate_index in list(remaining)
                if _overlaps_with_positive_area(current, patches[candidate_index])
            ]
            for neighbor_index in neighbors:
                remaining.remove(neighbor_index)
                stack.append(neighbor_index)
                component.append(patches[neighbor_index])
        components.append(component)
    return components


def _overlaps_with_positive_area(left: PatchAnomaly, right: PatchAnomaly) -> bool:
    return (
        left.box.x < right.box.x + right.box.w
        and right.box.x < left.box.x + left.box.w
        and left.box.y < right.box.y + right.box.h
        and right.box.y < left.box.y + left.box.h
    )


def _center_x(patch: PatchAnomaly) -> float:
    return patch.box.x + patch.box.w / 2.0


def _center_y(patch: PatchAnomaly) -> float:
    return patch.box.y + patch.box.h / 2.0


def _axis_cells(centers: list[float], limit: int) -> dict[float, tuple[int, int]]:
    if not centers:
        return {}
    if len(centers) == 1:
        return {centers[0]: (0, limit)}

    cells: dict[float, tuple[int, int]] = {}
    for index, center in enumerate(centers):
        left = 0 if index == 0 else math.floor((centers[index - 1] + center) / 2.0)
        right = limit if index == len(centers) - 1 else math.ceil((center + centers[index + 1]) / 2.0)
        cells[center] = (max(0, left), min(limit, right))
    return cells


def _component_to_tight_box(
    component: list[PatchAnomaly],
    *,
    x_cells: dict[float, tuple[int, int]],
    y_cells: dict[float, tuple[int, int]],
) -> MergedBox:
    x1 = min(x_cells[_center_x(item)][0] for item in component)
    y1 = min(y_cells[_center_y(item)][0] for item in component)
    x2 = max(x_cells[_center_x(item)][1] for item in component)
    y2 = max(y_cells[_center_y(item)][1] for item in component)
    return MergedBox(
        x=x1,
        y=y1,
        w=max(1, x2 - x1),
        h=max(1, y2 - y1),
        score=max(item.anomaly_score for item in component),
        patch_count=len(component),
        label="localized_defect",
    )
