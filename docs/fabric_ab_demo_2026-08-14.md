# Fabric A/B demo migration report — 2026-08-14

## Outcome and safety status

The repository now contains the requested architecture: one shared dense ResNet18 backbone, one isolated versioned memory bank per fabric, a data-driven Fabric Registry, per-fabric validation thresholds, source-disjoint manifests, three-state decisions, dense localization, mandatory operator fabric selection, and traceable API/database results.

The locked quality gate did **not** pass. Therefore neither candidate was activated. This is intentional: the runtime will return an explicit inactive-reference error rather than fall back to the stale mixed Company reference. The current implementation is structurally demo-ready, but the available A/B data and anomaly separation are not accurate enough to claim a passed demo or production readiness.

## Current-system diagnosis

The legacy runtime used independently cropped image patches, the final globally pooled 512-D ResNet18 feature and an optional 128-D projection head. `company_iqc_raw_all_v2` mixed the two commercial fabrics into one normal distribution, used a single threshold (`0.15`), localized after coarse patch scoring, and classified by bbox area. This allowed a locally obvious defect to disappear in a global representation; if thresholding or component filtering removed every bbox, the old classifier could return Good. The UI also allowed direct selection of stale references and displayed a heuristic confidence as if it were a probability.

Contributors confirmed in the repository were:

- mixed Fabric A/B normal reference and cross-texture nearest-neighbor contamination;
- globally pooled patch embeddings that weaken small/thin local defects;
- one threshold for different fabric distributions;
- aggressive legacy component filtering/post-processing;
- decision coupling to bbox existence/area;
- stale runtime artifact selection through `reference_id`/environment defaults;
- a non-calibrated confidence interpretation.

## Dataset inventory and source-disjoint split

Original images remain under `Data/raw/Company`; the migration does not rename, move or overwrite them.

```text
Data/raw/Company/
├── real/
│   ├── fabric_A_warp_knit_mesh/good/             36 real Good
│   ├── fabric_B_brushed_pile_velour_like/good/   27 real Good
│   └── unclassified/defect/                      14 unclassified real defects
└── fabric_synthetic_testset/
    ├── fabric_A_warp_knit_mesh/
    │   ├── synthetic_good/                       40
    │   ├── synthetic_defect/                     10
    │   └── defect_masks/                         10
    └── fabric_B_brushed_pile_velour_like/
        ├── synthetic_good/                       40
        ├── synthetic_defect/                     10
        └── defect_masks/                         10
```

Splits use `source_original`, so descendants of one original cannot cross reference/validation/locked-test boundaries.

| Fabric | Reference real Good | Validation real/synth Good/defect | Locked real/synth Good/defect | Development synth Good |
|---|---:|---:|---:|---:|
| Fabric A (`fabric_001`) | 27 | 4 / 9 / 5 | 5 / 7 / 5 | 24 |
| Fabric B (`fabric_002`) | 20 | 3 / 8 / 5 | 4 / 9 / 5 | 23 |

The 14 unclassified real defects are not used for calibration or locked scoring because they do not yet have reliable A/B identity. Synthetic defects and masks never enter a normal memory bank.

## New architecture

```text
Operator selects immutable fabric_id
              │
              ▼
       Fabric Registry
              │ exact active reference only; no fallback
              ▼
Shared cached ResNet18 backbone (ImageNet, layer2 + layer3)
              │ dense 384-D local features, stride 8
              ├───────────────┬───────────────┐
              ▼               ▼               ▼ future
      Fabric A FAISS    Fabric B FAISS    Fabric C FAISS
      memory bank       memory bank       memory bank
              │               │
       A thresholds      B thresholds
              └───────┬───────┘
                      ▼
 anomaly map → light closing → components → bbox
                      │
              GOOD / NEEDS_REVIEW / BAD
```

The backbone is cached once per process. Reference indexes load per selected fabric. An image selected as Fabric B can only resolve Fabric B's registered active artifact. Missing/inactive/incompatible references return an error. Significant anomaly without a surviving bbox returns `NEEDS_REVIEW / ANOMALY_WITHOUT_LOCALIZATION`.

## Immutable artifacts and thresholds

Shared backbone version: `shared_resnet18_imagenet1k_v1`.

Real-Good-only v1 artifacts:

- `storage/references/fabric_001/v1` — `fabric_001_ref_v1`, 27 sources, 12,000-vector coreset;
- `storage/references/fabric_002/v1` — `fabric_002_ref_v1`, 20 sources, 12,000-vector coreset.

Validation-only v1 thresholds:

| Setting | Fabric A | Fabric B |
|---|---:|---:|
| image score | top-K mean, top 2% | 99.5 percentile |
| review threshold | 0.1795017 | 0.1213344 |
| bad threshold | 0.1858405 | 0.1350869 |
| pixel threshold | 0.1850580 | 0.1347012 |
| minimum area | 8 px | 8 px |

An immutable v2 experiment additionally included only development synthetic-Good descendants (A: 51 total Good; B: 43). No validation or locked source entered v2. It improved validation FPR for B but did not pass locked evaluation, so it also remains calibrated/inactive.

## Evaluation

Legacy Company-v2 held-out baseline (mixed A/B, 24 synthetic images): TP 5, FP 11, TN 6, FN 2, recall 71.43%, FPR 64.71%.

Real-Good-only dense v1 locked result:

| Metric | Fabric A | Fabric B |
|---|---:|---:|
| Good test | 12 | 13 |
| Defect test | 5 | 5 |
| TP | 5 | 5 |
| TN | 6 | 4 |
| FP | 6 | 9 |
| FN | 0 | 0 |
| Defect recall | 100% | 100% |
| FPR | 50.00% | 69.23% |
| FNR | 0% | 0% |
| bbox recall | 100% | 100% |
| mean mask IoU | 0.0791 | 0.0425 |

Controlled-synthetic-Good v2 locked result:

| Metric | Fabric A | Fabric B |
|---|---:|---:|
| Good test | 12 | 13 |
| Defect test | 5 | 5 |
| TP | 4 | 5 |
| TN | 6 | 4 |
| FP | 6 | 9 |
| FN | 1 | 0 |
| Defect recall | 80% | 100% |
| FPR | 50.00% | 69.23% |
| FNR | 20% | 0% |
| bbox recall | 80% | 100% |
| mean mask IoU | 0.1163 | 0.0431 |
| average inference | 0.312 s | 0.313 s |

False-negative disclosure: v1 produced no locked synthetic defect as `GOOD`, but failed the <=10% Good FPR target. In v2, Fabric A `defect_10_crease_pressure_band.png` returned `GOOD` (score `0.162210`, just below review threshold `0.162418`). This is a blocking failure. No threshold was retuned from this locked sample.

## Files introduced or materially changed for this migration

| Path | Purpose |
|---|---|
| `config/company_demo_fabrics.json` | Stable identities and source-level split configuration |
| `scripts/prepare_company_demo_manifests.py` | Leakage-safe A/B manifests |
| `backend/app/ai/dense_feature_extractor.py` | Cached shared layer2/layer3 extractor |
| `backend/app/ai/dense_anomaly.py` | Isolated FAISS scoring, aggregation, anomaly map and localization |
| `backend/app/ai/fabric_decision.py` | GOOD/NEEDS_REVIEW/BAD fail-safe logic |
| `backend/app/services/fabric_registry.py` | Registry validation and exact reference routing |
| `scripts/build_fabric_references.py` | Immutable per-fabric candidates and coreset build |
| `scripts/calibrate_fabric_references.py` | Per-fabric validation-only threshold calibration |
| `scripts/evaluate_fabric_demo.py` | Locked acceptance gate and guarded activation |
| `backend/app/api/routes_fabric.py` | `GET /api/fabrics` |
| `backend/app/api/routes_inspection.py` | Mandatory multipart `fabric_id` |
| `backend/app/services/inspection_service.py` | Dense runtime path and exact fabric routing |
| `backend/app/db/models.py`, `schemas.py`, `database.py` | Persist/return trace metadata and additive SQLite migration |
| `frontend/src/services/api.ts`, `App.tsx`, `components/ReferenceList.tsx`, `components/InspectionBay.tsx` | Operator fabric selection and traceable result display |
| `tests/test_fabric_registry.py`, `test_dense_anomaly.py`, `test_fabric_decision.py`, `test_inspection_upload.py` | Routing, compatibility, fail-safe and API regressions |

Legacy modules remain available for rollback and historical evaluation, but the new `/api/inspect` contract no longer routes through the legacy reference selector.

## Commands

```powershell
# Generate source-disjoint manifests
.\.venv\Scripts\python.exe scripts\prepare_company_demo_manifests.py

# Build immutable real-Good-only v1 references
.\.venv\Scripts\python.exe scripts\build_fabric_references.py --version 1

# Calibrate using validation only
.\.venv\Scripts\python.exe scripts\calibrate_fabric_references.py --reference-version 1

# Locked gate; activation occurs only when all criteria pass
.\.venv\Scripts\python.exe scripts\evaluate_fabric_demo.py `
  --split locked_test --reference-version 1 `
  --output-dir storage/metadata/fabric_demo_evaluation_v1 `
  --activate-if-accepted

# Optional candidate experiment; always use a new version
.\.venv\Scripts\python.exe scripts\build_fabric_references.py `
  --version 2 --include-development-synthetic-good
.\.venv\Scripts\python.exe scripts\calibrate_fabric_references.py --reference-version 2
.\.venv\Scripts\python.exe scripts\evaluate_fabric_demo.py `
  --split locked_test --reference-version 2 `
  --output-dir storage/metadata/fabric_demo_evaluation_v2

# Verification
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run build

# Backend (port 8001 avoids the known local port-8000 conflict)
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.main:app `
  --reload --reload-dir backend --host 127.0.0.1 --port 8001

# Frontend, second terminal
cd frontend
npm.cmd run dev
```

## Adding Fabric C without rebuilding A/B

1. Add `fabric_003` to `config/company_demo_fabrics.json` with an editable `display_name`, `source_name`, `real_good_dir`, `synthetic_dir`, validation sources and locked sources.
2. Put only QC-approved real Good C images in its configured real-Good folder. Label C defects/masks and preserve `source_original` relationships.
3. Regenerate manifests. Existing A/B images and artifacts are untouched.
4. Build only C: `scripts/build_fabric_references.py --version 1 --fabric-id fabric_003`.
5. Calibrate only C: `scripts/calibrate_fabric_references.py --reference-version 1 --fabric-id fabric_003`.
6. Evaluate only C on locked test: `scripts/evaluate_fabric_demo.py --split locked_test --reference-version 1 --fabric-id fabric_003 --activate-if-accepted`.
7. Change `display_name` later without rebuilding; `fabric_id` and `reference_id` remain immutable.

New operator-confirmed Good images should enter a candidate pool, receive QC approval, and be used to build `vN+1`; they must never mutate an active memory bank in place.
