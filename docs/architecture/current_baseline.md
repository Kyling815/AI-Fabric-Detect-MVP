# Current baseline — AI Fabric Detect MVP

Audit date: 2026-10-13, Asia/Bangkok. Branch: `main`.
Baseline commit: `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`.

This document describes implemented behavior at Phase 0. PAH-CNN features proposed in `deep-research-report.md` are future work. The current system already uses CNN features, nearest-neighbor anomaly detection, and rule-based QC decisions.

## 1. Repository map and audit scope

All 176 tracked paths were inventoried. The 115 existing Python files in `backend/`, `scripts/`, and `tests/` were parsed and their definitions/calls inspected; runtime, registries, artifact builders, model loaders, preprocessing, schemas, frontend components, configuration, and evaluation reports were examined. The Python source inventory contains 13,548 lines before Phase 0 additions. Generated dependencies in `.venv/` and `frontend/node_modules/` were checked through installed versions and execution, rather than reviewed as project source. Binary datasets and checkpoints were inspected through metadata, hashes, and loading; they were not edited.

| Area | Responsibility |
| --- | --- |
| `backend/main.py` | FastAPI construction, DB initialization, CORS, route registration, public image mounts |
| `backend/app/api/` | HTTP validation and service invocation |
| `backend/app/services/` | Inspection orchestration, fabric/legacy registries, dataset discovery, feedback, storage |
| `backend/app/ai/` | Dense and legacy patch encoders, projection, FAISS, anomaly scoring, localization, decisions |
| `backend/app/ai/preprocessing/` | Existing optional v2 processing, ROI, quality, orientation, texture/FFT, batch CLI |
| `backend/app/db/` | SQLAlchemy models, CRUD, SQLite initialization and response schemas |
| `scripts/` | Offline preparation, training, building, calibration, evaluation and search |
| `frontend/src/` | React UI, fabric selector, upload, SVG boxes, health indicator and logs |
| `config/` | Company demo backbone/dataset configuration and preprocessing v2 |
| `tests/` | 29 existing test modules, 95 executed tests |
| `Data/` | Ignored original/processed dataset assets; 643 existing files |
| `storage/` | Ignored runtime artifacts, registries, DB and images; 311 existing files |

No applicable `AGENTS.md` was found in the workspace/checked ancestor paths. `.agents/` contained no instruction files. Existing uncommitted changes were `tempCodeRunnerFile.exe`, `Revision.md`, `deep-research-report.md`, and `test_image/`.

## 2. Actual runtime map

```text
frontend/src/main.tsx -> App.tsx
  FabricSelector -> GET /api/fabrics
  InspectionBay -> apiService.inspectImage(file, fabricId)
        |
        v
backend.main:app = create_app()
  POST /api/inspect (multipart image + required fabric_id)
        |
        v
routes_inspection.inspect_image()
  image MIME validation, Depends(get_db)
        |
        v
InspectionService.create_fabric_inspection()
  FabricRegistry.get_active_reference(fabric_id)
  StorageService.save_upload() -> inspect_image_file()
        |
        v
get_shared_dense_extractor(backbone_version, device, pretrained, adapter_path)
  SharedDenseResNet18.extract_path()
  EXIF -> RGB -> aspect-preserving resize <= 768 -> ImageNet normalization
  ResNet18 layer2(128ch) + layer3(256ch)
  avg_pool3 -> interpolate layer3 -> L2 each -> concat384 -> L2
  shared ProjectionHead.forward_features() -> normalized128
        |
        v
DenseReferenceDetector(reference.metadata_path)
  FAISS IndexFlatIP / k=1 -> max(0, 1 - inner_product)
  aggregation + localized(max - median) -> anomaly map
  pixel threshold -> closing3x3 -> connected components -> area filter
        |
        v
map dense boxes to original pixels
  decide_fabric_inspection() -> GOOD / NEEDS_REVIEW / BAD
  annotated image -> Inspection + DetectedDefect -> SQLite
  InspectionCreateResponse -> React image + SVG bbox overlay
```

Actual entrypoints:

- `backend/main.py:create_app`; module-level `app` initializes runtime directories and SQLite.
- `backend/app/api/routes_inspection.py:inspect_image` calls only `create_fabric_inspection`.
- `backend/app/services/inspection_service.py:InspectionService.create_fabric_inspection` orchestrates the current public inspection path.
- `backend/app/ai/pipeline.py` is a placeholder, not the live orchestrator.

Model loading:

- Registry `shared_backbone` selects `shared_resnet18_imagenet1k_dense_adapter_company_v4`.
- `SharedDenseResNet18` builds `torchvision.models.resnet18` with ImageNet weights; cached weights are outside project storage, in `C:/Users/Admin/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` in this environment.
- `projection_head.load_projection_head` uses `torch.load(..., weights_only=True)`, loads a linear/ReLU/dropout/linear projection, calls `eval()`, and produces L2-normalized features. The binary classification head in this checkpoint is not the dense runtime QC classifier.
- The dense extractor is cached with `lru_cache(maxsize=2)`. Each inspection loads its FAISS index anew through `DenseReferenceDetector`.
- Base features: 384 dimensions, stride 8. Adapter v4 output: 128 dimensions. Device `auto` resolves to CPU in this checkout.

## 3. Public contracts and persistence

| Endpoint | Implemented behavior |
| --- | --- |
| `GET /api/health` | Health/settings response |
| `GET /api/fabrics` | Stable ID, display name, status, active reference, inspectable flag |
| `POST /api/inspect` | Required `image` and `fabric_id`; successful response HTTP 201 |
| `GET /api/inspections` | Persisted inspection list |
| `GET /api/inspections/{id}` | Persisted result or 404 |
| `POST /api/reference/build` | Existing legacy patch reference builder; has artifact/registry side effects |
| `GET /api/reference/list` | Legacy ReferenceRegistry entries |
| `GET /api/reference/status` and `/{reference_id}` | Legacy reference status |
| `GET /api/datasets/paths` and `/discover` | Dataset roots and discovery |
| `POST /api/feedback` | Persist operator feedback; does not train a model |
| `GET /api/model-info` | Shared backbone metadata plus legacy settings |

Inspection response contains `inspection_id`, fabric/reference/backbone IDs, `final_result`, `anomaly_score`, `reason`, `confidence`, area/count, boxes, image URLs/metadata, preprocessing/patch/embedding/anomaly summaries, `model_version`, and timestamp. Dense inspections set `confidence=None`; anomaly score is not a calibrated defect probability.

Current rejection paths:

- missing required form fields: 422;
- unknown fabric: 400;
- inactive fabric, absent active reference, missing artifact, registry/backbone mismatch, incomplete thresholds: 409;
- non-image MIME or undecodable image: 400;
- upload beyond configured limit: 413.

Registry validation happens before upload persistence. Errors after upload clean unpersisted upload/annotated files. Successful results save both images and the inspection/defects. Only `/storage/uploads` and `/storage/annotated` are publicly mounted.

SQLite uses SQLAlchemy with foreign keys enabled. `init_db()` creates tables and adds absent inspection metadata columns; importing `backend.main` can therefore write to the configured DB. Phase 0 execution overrides DB/storage to temp before that import. Existing DB was examined read-only: 99 inspections, 73 detected defects, zero feedback rows, four model-version rows. Tables also include reference images, reference patches and datasets.

## 4. Registries, artifacts and thresholds

Two registries coexist:

| Registry | Used by | Actual state |
| --- | --- | --- |
| `storage/metadata/fabric_registry.json` | Public dense `/api/inspect`, `/api/fabrics` | A/B INACTIVE; no active reference; candidates v1-v5 CALIBRATED |
| `storage/metadata/reference_registry.json` | Legacy reference API and legacy patch service | One legacy entry; `active_reference_version=resnet18_company_iqc_raw_all_v2_20260814052015690169` |

The legacy active pointer does not enable dense Fabric A/B inspection and is not proof of production acceptance.

Checkpoint files:

- `storage/models/shared_dense_adapter_company_v4.pt`: current dense adapter, 384 -> 128.
- `storage/models/shared_dense_adapter_company_v1.pt`: historical dense adapter.
- `storage/models/projection_head_raw_all_v2.pt`: legacy patch projection configured in local settings.

Fabric artifact layout for each of `fabric_001` and `fabric_002`, versions v1-v5:

```text
storage/references/<fabric_id>/v<N>/
  reference.json      model, dataset, dimensions, image provenance, thresholds
  thresholds.json     selected calibration profile and validation evidence
  memory_bank.npy     reference vectors
  memory_bank.faiss   search index
  manifest.json       source rows
  metrics.json        evaluation state
```

`storage/references/shared/<backbone_version>/backbone.json` records backbone metadata. `config/company_demo_fabrics.json` selects the offline dataset/backbone profile. `storage/metadata/company_demo_manifests/` contains existing split manifests. Phase 0 runs none of the builders, calibrators, evaluators or activation scripts.

| Versions | Backbone | Feature dimension | Reference vectors per fabric |
| --- | --- | ---: | ---: |
| v1-v2 | Base ImageNet ResNet18 | 384 | 12,000 |
| v3 | Dense adapter v1 | 128 | 12,000 |
| v4-v5 | Dense adapter v4 | 128 | 12,000 |

Current v5 profiles (CALIBRATED, not ACTIVE):

| Threshold field | Fabric A | Fabric B |
| --- | --- | --- |
| image score | top_k_median_contrast, top 2% | percentile_995 |
| review | 0.22188054025173184 | 1.0874984264373777 |
| bad | 0.301334485411644 | 1.087498426437378 |
| pixel mode | absolute | absolute |
| pixel threshold | 0.44946396350860596 | 0.25867345929145813 |
| minimum component area | 128 pixels | 128 pixels |
| localized review | 0.6686849594116212 | 1.142270237207413 |

For these versions, thresholds appear in the fabric registry record, `reference.json`, and `thresholds.json`. Runtime passes the registry profile explicitly to scoring/decision; `thresholds.json` is not independently read on every inspection.

`FaissIndexStore` supports IP and squared L2, but the dense detector explicitly uses cosine/IP semantics and computes `1 - score`. The store itself does not normalize vectors; the dense encoder/projection does. `deterministic_coreset` currently means seeded random subsampling, not representative greedy k-center selection.

## 5. Legacy path and existing preprocessing

`InspectionService.create_placeholder_inspection` remains in source for legacy behavior. It uses `ReferenceRegistry`, prepared patches, `CNNFeatureExtractor`, optional projection, `FaissAnomalyDetector`, legacy bbox heuristics and `classify_inspection`. Legacy labels are `Good`, `Acceptable`, `Bad`, or `ReferenceMissing`. The public `/api/inspect` route no longer calls this method.

Preprocessing v2 already includes quality metrics/validation, ROI, orientation, illumination, color handling, denoising, CLAHE, resizing/padding, overlapping patches, texture and FFT metadata, coordinate transforms and batch reporting. The dense public route directly invokes `SharedDenseResNet18.extract_path` and does not invoke that v2 sequence. These modules are candidates for later reuse, not absent capabilities to duplicate.

Local settings enable pretrained weights and legacy projection, with max image size 768, legacy model input 224, patches 96, stride 48 and batch 16. Dense runtime takes backbone/adapter from FabricRegistry. Model-info and preprocessing/patch summaries still contain legacy settings: they must not be interpreted as the actual dense feature stride or an indication that the dense adapter is disabled.

## 6. Frontend and compatibility observations

`main.tsx` mounts `App` in StrictMode. `FabricSelector` loads `/api/fabrics` and permits selection only for inspectable entries. With the actual registry, both entries are disabled. `InspectionBay` previews the local image and overlays result boxes in original-image coordinates. `StatusIndicator` polls health; `ResultLog` displays session logs. API helpers exist for reference building, feedback, and persisted history, although the main screen does not expose every helper as a full UI workflow.

Vite dev proxy maps `/api` to `http://127.0.0.1:8001`. `/storage` is not proxied; the current inspection canvas uses a local FileReader preview and SVG boxes, so its main display does not depend on annotated image URL proxying. A frontend build success is not browser E2E coverage, nor does `vite preview` automatically provide a production API reverse proxy.

## 7. Reproducibility, evidence and known limits

Environment: Python 3.11.9; torch 2.5.1+cpu; torchvision 0.20.1+cpu; FAISS 1.9.0 (`faiss-cpu` distribution 1.9.0.post1); CUDA unavailable. Node 22.19.0, npm 10.9.3; installed React/ReactDOM 19.2.7, Vite 8.0.16, TypeScript 6.0.3. Full installed Python dependency snapshot and 55 artifact/config/cache checksums are in [phase_0_baseline_manifest.json](../implementation/phase_0_baseline_manifest.json). No packages were installed or upgraded.

Existing tests: 95 passed, one FAISS/NumPy deprecation warning. Frontend build passed; no frontend test script exists. Added diagnostic runner has 77 passing smoke assertions, covering model loading/forward, normalization/repeatability, all ten existing indexes, existing inactive-registry API behavior, API success/error routes, DB persistence and static image serving. Synthetic thresholds deliberately exercise each decision branch; this is software verification, not real-data evaluation.

Historical v5 development/postmortem evidence remains A: TP20/FP18/TN2/FN0, B: TP20/FP1/TN19/FN0. It is not a fresh locked-test result. No production-ready candidate is established. Old locked data was not re-evaluated or changed in Phase 0.

Open items for review, without Phase 0 runtime fixes:

1. README mixes required `fabric_id` with the obsolete optional `reference_id` inspection contract and describes older reference generations.
2. Legacy reference building automatically updates its legacy active pointer for nonempty builds. Dense acceptance gating is distinct; migration must preserve the distinction and later address governance explicitly.
3. Dense model-info hardcodes `use_projection_head=False` while a dense adapter is configured, and shows legacy projection path/input settings. Post-persistence GET responses do not restore all transient embedding/anomaly summaries.
4. Existing dense extractor tests validate version names, not actual checkpoint forward execution. Phase 0 smoke fills that gap for this local artifact set.
5. Dense detector validates query/index dimensions, but does not independently prove all metadata fabric/encoder identity fields. A wrong registry pointer could still be a future correctness risk.
6. Native FAISS deserialization can raise errors outside the service's current `(OSError, KeyError, ValueError)` catch; model-loading exceptions also occur outside that reference catch. Comprehensive branch failure normalization is not established by this audit.
7. Adapter training randomly splits generated feature pairs; source-derived pairs can share provenance between train/validation. Manifest-level partition checking does not prove pair-level source independence.
8. A v4/v5 bank uses 33 images but 17 distinct `source_original` labels; B uses 40 images but 14 labels. Counts were read from artifact provenance and do not establish independent captures.
9. No physical calibration, camera/encoder SDK, supervised segmentation, hybrid fusion, streaming scheduler or measured production GPU throughput exists in the audited runtime.
10. Artifact/registry paths include local absolute Windows paths; data/models are ignored by Git. A fresh clone alone cannot reproduce this installed environment/artifact set.

Before/after SHA-256 scope checks confirmed all original tracked files, 311 storage files, 643 dataset files, ten user files and three cached weight files remained byte-identical. The freeze is an audit fingerprint and dependency record, not an archive of the data or a promise of identical numerical output on another device.

Re-run the baseline from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

The runner reads existing artifacts, requires cached pretrained weights, writes only temp smoke/regression fixtures and ignored frontend build outputs, and leaves the production registry/DB unchanged. Its only fixture ACTIVE record exists inside the temporary registry; it does not activate an existing candidate.
