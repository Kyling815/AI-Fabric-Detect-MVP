# Core logic v5 ablation — 2026-09-03

## Objective

Measure whether post-embedding core changes can address the two v4 failure modes without changing the v4 adapter weights or curated dataset:

- Fabric A: high false-positive rate under lot/camera domain shift.
- Fabric B: one image-level false negative despite localized anomaly boxes.

This is an ablation, not a release evaluation. The previously opened v4 locked set is treated only as hard-development/postmortem data. Neither v5 reference is activated.

## Structural changes

### Dense anomaly result

`DenseAnomalyResult` now carries two independent signals:

- `image_score`: calibrated aggregate anomaly evidence.
- `localized_score`: strongest local deviation after subtracting the image median.

This prevents the decision layer from discarding strong local evidence merely because an aggregate score is below its review threshold.

### Threshold schema v2

The calibrated threshold payload now records:

- `threshold_schema_version = 2`
- `core_logic_version = dense_anomaly_v2`
- image-score method and top fraction
- review/bad thresholds
- pixel-threshold mode and optional MAD multiplier
- minimum defect area
- localized-review threshold

Old threshold payloads remain supported and keep absolute localization plus the previous decision behavior.

## Algorithm changes

### Image-level scoring

The calibrator still tests `max`, `top_k_mean`, and `percentile_995`, and additionally tests median-centered variants:

- `max_median_contrast`
- `top_k_median_contrast`
- `percentile_995_median_contrast`

Median centering removes a uniform image-wide anomaly shift while preserving locally elevated patches.

### Adaptive localization

Calibration now compares the legacy absolute threshold against per-image robust thresholds:

```text
effective threshold = max(
    configured floor,
    image median + multiplier × 1.4826 × MAD
)
```

MAD multipliers from 3 to 10 are evaluated with minimum component areas from 8 to 128 pixels. Calibration retains the candidate with the best defect localization recall, lowest normal localization rate, and highest mean mask IoU.

### Decision fusion

The decision order is now:

1. Use image score for `GOOD`, `NEEDS_REVIEW`, or `BAD` as before.
2. If image score is below review but boxes exist and localized score exceeds its independently calibrated threshold, return `NEEDS_REVIEW` with reason `STRONG_LOCALIZED_ANOMALY`.
3. Never promote localized evidence directly to `BAD`.

The localized threshold is placed immediately above the maximum localized score of validation Good images, so it adds zero localized-fallback false positives on validation by construction.

## Candidate artifacts

- `fabric_001_ref_v5`: 33 Real Good images, 12,000 reference vectors.
- `fabric_002_ref_v5`: 40 Real Good images, 12,000 reference vectors.
- Both use the unchanged `shared_dense_adapter_company_v4` checkpoint.
- Both remain `CALIBRATED/INACTIVE`.

## Validation results

| Fabric | Selected image method | TP | FP | TN | FN | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fabric A | `top_k_median_contrast`, top 2% | 12 | 3 | 5 | 0 | 100% | 37.5% |
| Fabric B | `percentile_995` | 13 | 0 | 9 | 0 | 100% | 0% |

Median contrast was selected for Fabric A, but the normal and defect score distributions still overlap. The best validation separation remains negative, so no scalar image threshold can meet both zero false negatives and low false positives with the current representation/reference coverage.

## Hard-development comparison

| Fabric | Version | TP | FP | TN | FN | Recall | FPR | Mean mask IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fabric A | v4 baseline | 20 | 18 | 2 | 0 | 100% | 90% | 2.80% |
| Fabric A | v5 core | 20 | 18 | 2 | 0 | 100% | 90% | 2.80% |
| Fabric B | v4 baseline | 19 | 0 | 20 | 1 | 95% | 0% | 17.23% |
| Fabric B | v5 core | 20 | 1 | 19 | 0 | 100% | 5% | 17.23% |

Fabric B's missed `broken_or_loose_yarn` image is now routed to `NEEDS_REVIEW`. One hard-development Good image also becomes `NEEDS_REVIEW`; this remains within the 10% FPR target on this postmortem set.

Fabric A does not improve. Its issue is not primarily aggregation or thresholding: Good and defective embeddings overlap across lots, and the current localization map does not align closely enough with the masks.

## Decision and next step

- Keep the v5 core changes because they eliminate Fabric B's false negative while preserving backward compatibility.
- Do not activate Fabric A or B from this ablation because there is no fresh locked v5 set.
- For Fabric A, the next material change must be reference/representation work: more source-diverse Real Good, lot-balanced memory banks or sub-banks, and source-grouped adapter training.
- After using v4 locked samples as hard development, create a completely new source-disjoint locked set before any release claim.

## Verification

- Full automated suite: 95 passed.
- v4 adapter weights modified: no.
- v4 reference artifacts overwritten: no.
- v5 candidate automatically activated: no.
