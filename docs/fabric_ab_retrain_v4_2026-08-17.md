# Fabric A/B retrain and locked evaluation v4 — 2026-08-17

## Scope

This run used `company_demo_curated_manifest_v4` and executed the release sequence in order:

1. Train one shared dense projection adapter from reference Good and masked adapter-train defects.
2. Build an isolated Real-Good memory bank for each fabric.
3. Calibrate image and localization thresholds on validation only.
4. Freeze those thresholds and evaluate the locked set once.
5. Activate neither reference because the locked acceptance gate failed.

## Adapter

- Checkpoint: `storage/models/shared_dense_adapter_company_v4.pt`
- SHA-256: `0B455192D0EAE4C5AA1725B12A7C16DFFB691EE5A81D44039DBF1DE29904858C`
- Best epoch: 29/30
- Best pair-validation loss: 0.0010133
- Best pair-validation accuracy: 99.9792%
- Locked data used by training: no

Pair-validation uses pairs sampled from the development feature pool. It verifies that optimization converged, but it is not an estimate of image-level production performance.

## Reference build

| Fabric | Reference | Real Good images | Coreset vectors | Memory bank SHA-256 |
| --- | --- | ---: | ---: | --- |
| Fabric A | `fabric_001_ref_v4` | 33 | 12,000 | `23E7CACB5791674E63634DF4048EC6022BC91F2E0B67DE94A121FA25772A2FA4` |
| Fabric B | `fabric_002_ref_v4` | 40 | 12,000 | `6798530AEE170403D0000B6EACF39DEA5504EB71CA109955314355807F1A0700` |

Both references were built from `reference_real_good` only. Validation and locked images were excluded from their memory banks.

## Validation calibration

| Fabric | Method | Review threshold | Bad threshold | Pixel threshold | Min area | TP | FP | TN | FN | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fabric A | `top_k_mean`, top 2% | 0.232341 | 0.313990 | 0.449464 | 128 | 12 | 3 | 5 | 0 | 100% | 37.5% |
| Fabric B | `percentile_995` | 1.087498 | 1.087498 | 0.258673 | 128 | 13 | 0 | 9 | 0 | 100% | 0% |

The Fabric A validation false-positive rate was already above the release target. Thresholds were nevertheless frozen before locked evaluation; no threshold was selected from locked results.

## Locked evaluation

Acceptance rule: zero false negatives, defect recall 100%, and false-positive rate at most 10% for every fabric.

| Fabric | Good | Defect | TP | FP | TN | FN | Recall | FPR | Precision | F1 | BBox recall | Mean mask IoU | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fabric A | 20 | 20 | 20 | 18 | 2 | 0 | 100% | 90% | 52.63% | 68.97% | 100% | 2.80% | No |
| Fabric B | 20 | 20 | 19 | 0 | 20 | 1 | 95% | 0% | 100% | 97.44% | 100% | 17.23% | No |

Fabric A classified all defects as positive, but only 2/20 locked Good images remained `GOOD`; 6 became `NEEDS_REVIEW` and 12 became `BAD`. Its locked normal scores overlap substantially with defect scores, so adjusting one global threshold cannot satisfy both recall and FPR.

Fabric B separated all locked Good images, but one `broken_or_loose_yarn` defect fell below the image decision threshold. Localization produced at least one box for all defect images, including that false-negative image, which shows a mismatch between the image-level gate and localization evidence.

Locked report: `storage/metadata/fabric_demo_evaluation/v4/locked_test_evaluation.json`

## Release decision

- `all_accepted`: false
- Fabric A registry status: `INACTIVE`
- Fabric B registry status: `INACTIVE`
- Active references: none
- v4 reference status: `CALIBRATED`, not `ACTIVE`
- Backend restart for deployment: not performed because no candidate passed the gate

The v4 locked set has now been opened. Its results may be used for postmortem analysis, but if they influence architecture, training, or thresholds, a new source-disjoint locked set must be created for the next release decision.

## Recommended v5 direction

1. Treat Fabric A as a domain-shift problem: expand Real Good reference and validation coverage across the variations represented by `lot_003`; do not raise the current threshold from locked scores.
2. Rework image-level evidence so a strong localized component can trigger review even when an aggregate score falls below threshold; this specifically targets Fabric B's missed loose-yarn sample.
3. Improve localization supervision or anomaly-map resolution because mean mask IoU remains low, especially for Fabric A.
4. Split pair-validation by source image rather than random pair, so optimization metrics do not share source-derived patches across train and validation.
5. Create a fresh locked set before evaluating v5.

## Verification

- Full regression suite: 90 passed
- Versioned adapter validator tests: passed
- References automatically activated: no
