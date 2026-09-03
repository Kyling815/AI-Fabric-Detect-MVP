# Fabric A/B shared-adapter retraining and reference v3

## Result

One shared dense adapter was retrained and two new immutable references were built successfully:

- `fabric_001_ref_v3` at `storage/references/fabric_001/v3`;
- `fabric_002_ref_v3` at `storage/references/fabric_002/v3`;
- shared checkpoint `storage/models/shared_dense_adapter_company_v1.pt`.

V3 did not pass the locked quality gate and was not activated. It remains available for diagnosis; v1/v2 remain unchanged for rollback.

## Leakage-safe redesign

The previous validation sources were divided before retraining into `adapter_train` and a smaller calibration split. Locked-test sources were left unchanged and were never loaded by the training script.

| Fabric | Adapter-train normal real/synthetic | Adapter-train defects | Calibration normal real/synthetic | Calibration defects |
|---|---:|---:|---:|---:|
| A | 2 / 4 | 2 | 2 / 5 | 3 |
| B | 1 / 4 | 3 | 2 / 4 | 2 |

The normal training pool also used reference Real Good and development-only synthetic Good. Synthetic/real descendants remain grouped by `source_original`; manifest generation reported zero leakage.

## Training

The frozen shared ResNet18 still produces 384-D layer2/layer3 dense features. One shared `384 → 256 → 128` projection adapter was trained from both fabrics using:

- same-fabric normal/normal positive patch pairs;
- mask-selected defect patches;
- nearest normal patches as hard negative pairs;
- no locked-test images;
- one adapter shared by A and B, not two backbone models.

Best training checkpoint:

| Metric | Value |
|---|---:|
| Best epoch | 19 |
| Train pair loss | 0.002247 |
| Validation pair loss | 0.000873 |
| Pair accuracy | 99.979% |
| Retrain duration | approximately 104 seconds |

Pair accuracy only measures the sampled adapter-training task and is not treated as demo accuracy.

## V3 reference build

Both memory banks contain only approved Real Good features:

| Reference | Real Good sources | Coreset | Build time |
|---|---:|---:|---:|
| `fabric_001_ref_v3` | 27 | 12,000 | 9.48 s |
| `fabric_002_ref_v3` | 20 | 12,000 | 7.26 s |

Representation version: `shared_resnet18_imagenet1k_dense_adapter_company_v1`.

## Calibrated thresholds

| Threshold | Fabric A | Fabric B |
|---|---:|---:|
| Image score | max | top-K mean, top 0.1% |
| Review | 1.125808 | 1.117082 |
| Bad | 1.138323 | 1.235011 |
| Pixel | 0.880968 | 1.048871 |
| Minimum area | 128 | 8 |
| Calibration recall | 100% | 100% |
| Calibration FPR | 28.57% | 16.67% |

## Locked-test result

| Metric | Fabric A | Fabric B |
|---|---:|---:|
| Good images | 12 | 13 |
| Defect images | 5 | 5 |
| TP | 2 | 1 |
| TN | 10 | 10 |
| FP | 2 | 3 |
| FN | 3 | 4 |
| Defect recall | 40% | 20% |
| FPR | 16.67% | 23.08% |
| FNR | 60% | 80% |
| Average inference | 0.441 s | 0.453 s |

The adapter overfit the small set of trained defect appearances. It reduced false positives compared with v1 but collapsed recall on unseen categories. Therefore the v3 reference is `CALIBRATED`, the fabric remains `INACTIVE`, and `active_reference` remains `null`.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe scripts\prepare_company_demo_manifests.py

.\.venv\Scripts\python.exe scripts\train_shared_dense_adapter.py `
  --epochs 30 --batch-size 256 --pairs-per-fabric 6000

.\.venv\Scripts\python.exe scripts\build_fabric_references.py --version 3
.\.venv\Scripts\python.exe scripts\calibrate_fabric_references.py --reference-version 3

.\.venv\Scripts\python.exe scripts\evaluate_fabric_demo.py `
  --split locked_test --reference-version 3 `
  --output-dir storage/metadata/fabric_demo_evaluation_v3
```

Do not add `--activate-if-accepted` for the recorded v3 result: it correctly fails the gate.

## What is required to finish the quality gate

The code/rebuild pipeline is complete. Model acceptance now needs more independent, labeled data rather than further threshold tuning on the existing locked samples:

1. Collect real defects for A and B with masks or boxes, especially thin yarn, line, stain, hole and local texture defects.
2. Add multiple production sessions/lighting conditions for Real Good.
3. Reserve new source-disjoint locked lots that have never been inspected during model design.
4. Retrain a new immutable adapter version and build reference v4.
5. Activate only after recall is 100% and Good FPR is at most 10% on that new locked set.
