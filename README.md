# AI Fabric Detect MVP

AI Fabric Detect MVP is a local-first computer vision web application for fabric defect inspection. The MVP uses lightweight CNN / vision embeddings, patch comparison, and a normal-reference library to detect suspicious texture anomalies in close-up fabric images.

The core defect detection pipeline is not an LLM. It is designed around pretrained computer vision encoders, patch embeddings, FAISS or cosine similarity search, anomaly scoring, and rule-based Good / Acceptable / Bad classification.

## Fabric A/B demo architecture (2026-08-14)

The current migration adds one cached ResNet18 dense feature extractor (`layer2` + `layer3`) and isolated, versioned memory banks for `fabric_001` (Fabric A) and `fabric_002` (Fabric B). The operator must select a stable `fabric_id`; the backend never falls back to another fabric reference. Decisions are `GOOD`, `NEEDS_REVIEW`, or `BAD`, and a significant image anomaly without a bbox is fail-safe `NEEDS_REVIEW`.

The source-disjoint v1/v2 candidates did not meet the locked-test acceptance gate. A shared supervised dense adapter and isolated v3 references were subsequently retrained, but v3 generalized worse to unseen defect categories (A recall 40%, B recall 20%). The registry therefore intentionally has no active reference. The UI shows both fabrics as inactive rather than silently using a stale or rejected artifact. See [the Fabric A/B implementation report](docs/fabric_ab_demo_2026-08-14.md) and [the v3 retraining report](docs/fabric_ab_retrain_v3_2026-08-15.md).

New API contract:

```text
GET  /api/fabrics
POST /api/inspect   multipart: image=<file>, fabric_id=fabric_001
```

Prepare, build, calibrate, evaluate, and activate only if accepted:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_company_demo_manifests.py
.\.venv\Scripts\python.exe scripts\build_fabric_references.py --version 1
.\.venv\Scripts\python.exe scripts\calibrate_fabric_references.py --reference-version 1
.\.venv\Scripts\python.exe scripts\evaluate_fabric_demo.py --split locked_test --reference-version 1 --activate-if-accepted
```

## Current Phase

Phase 1 is the project setup and backend skeleton:

- FastAPI backend application structure
- Environment and path configuration
- SQLite database models
- API router skeletons
- Service and AI module placeholders for later phases
- Dataset discovery for Company IQC, DME, WFDD, Lusitano, and MVTec-style folders
- Upload persistence for incoming inspection images
- Image readability validation with width, height, mode, and file-size metadata
- Preprocessing plan generation for resize and patch settings
- Patch-grid generation over the resized image, including patch count, rows, columns, coverage, and sample patch boxes
- Lightweight CNN feature extraction over extracted patches with CUDA/CPU fallback
- Normal-reference library building from `non defective` / `train/good` folders, saving embeddings and metadata
- FAISS index creation from saved reference embeddings, with load/search helper support
- Runtime patch anomaly scoring against the latest FAISS reference index
- Suspicious patch merging into defect boxes, annotated image output, SQLite defect persistence, and rule-based Good / Acceptable / Bad classification

The backend now includes the full local MVP inspection flow.

## Preprocessing v2

The project now includes an additive, versioned preprocessing-v2 path for already-captured fabric photos. The existing v1 Pillow resize/patch path remains the default and remains compatible with existing model and FAISS artifacts.

V2 provides quality validation, ROI support, Lab luminance correction, mild bilateral denoising, luminance-only CLAHE, aspect-ratio-safe resizing, overlapping reflection-padded patches, optional orientation/color/sharpening, optional texture view and FFT metadata, batch CSV reports, visual previews, v2 reference rebuilding, threshold recalibration, and P0-P7 ablation configs.

Start with:

```powershell
python -m backend.app.ai.preprocessing.cli --input Data/raw/external --output Data/processed/preprocessed_v2 --config config/preprocessing_v2.json --split train --extract-patches
```

See [docs/preprocessing_v2.md](docs/preprocessing_v2.md) for the configuration, registry, training, FAISS, calibration, ablation, migration, and failure-handling guide.

## Planned Local Development

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

API docs will be available at:

```text
http://127.0.0.1:8001/docs
```

### Frontend

The frontend is a Vite-based React application.

```powershell
cd frontend
npm install
npm run dev
```

The application will be available at:

```text
http://localhost:5173
```

Environment variables use the `FABRIC_` prefix, as shown in `.env.example`, to avoid collisions with system-wide variables.

## Dataset and Artifact Layout

The project keeps original data, optional prepared image inputs, and generated AI artifacts separate:

```text
Data/
  raw/                         original datasets; do not modify during pipeline runs
    Company/                   Company IQC photos; current loose files are defect samples
      normal/                  optional future Company normal-reference photos
    external/
      20250129_Fabric_defects/
        Dataset/
          broken thread/
          damage/
          holes/
          non defective/
          Stain/
      Lusitano/
        nondefects/
      WFDD/
        grey_cloth/
        grid_cloth/
        pink_flower/
        yellow_cloth/
  processed/                   optional prepared image datasets
    reference_ready/           curated normal-image sets for future multi-fabric references

storage/                       generated AI/runtime artifacts
  indexes/                     FAISS indexes and embedding matrices
  metadata/                    reference metadata, registry, split manifests, evaluation JSON
  models/                      trained checkpoints such as the projection head
  uploads/                     original uploaded inspection images
  annotated/                   inspection outputs with defect boxes
```

`Data/processed/reference_ready` is an optional input layer. Put one curated normal-image set per folder, for example `Data/processed/reference_ready/floral_print`, then build it with `--dataset-name reference_ready --fabric-type floral_print`.

Existing generated artifacts stay in `storage`; they should not be moved into `Data/processed`.

All fabric photos below `Data/raw/Company` and the supported datasets below `Data/raw/external` are collected recursively into a unified manifest. WFDD `ground_truth` images are retained as mask metadata and are not treated as input photos. Because the current loose Company files are marked IQC/defect samples and there is no Company normal set yet, they participate in projection-head training as defects but do not create a Company FAISS reference. Put future normal Company images in `Data/raw/Company/normal`, `good`, or `nondefects` to enable that reference automatically.

## Useful API Endpoints

- `GET /api/health`
- `GET /api/datasets/paths`
- `GET /api/datasets/discover`
- `POST /api/inspect`
- `GET /api/inspections`
- `GET /api/inspections/{inspection_id}`
- `POST /api/reference/build`
- `GET /api/reference/list`
- `GET /api/reference/status`
- `GET /api/reference/status/{reference_id}`
- `POST /api/feedback`
- `GET /api/model-info`

`POST /api/inspect` performs the full MVP inspection pipeline: it saves the uploaded image, verifies that it can be decoded, stores image metadata, returns the resize preprocessing plan, extracts a CNN-ready patch grid, runs a lightweight CNN feature extractor, scores patch anomalies against a selected FAISS reference index, merges suspicious patches into boxes, saves an annotated image, persists detected defects, and assigns `Good`, `Acceptable`, or `Bad` using per-reference thresholds when available.

`POST /api/inspect` accepts an optional multipart form field:

```text
reference_id=dme_plain_v1
```

If `reference_id` is omitted, the backend uses `FABRIC_DEFAULT_REFERENCE_ID`.

`FABRIC_USE_PRETRAINED_WEIGHTS=false` by default keeps local setup offline-safe. Set it to `true` after torchvision weights are available locally or network download is acceptable.

Uploads are limited to 25 MiB by default. Override `FABRIC_MAX_UPLOAD_SIZE_BYTES` when a different local capture limit is required.

Prepare or check pretrained CNN weights:

```powershell
python scripts/prepare_model_weights.py --model-name resnet18
python scripts/prepare_model_weights.py --model-name resnet18 --download
```

After weights are cached, set `FABRIC_USE_PRETRAINED_WEIGHTS=true` in `.env` and rebuild the reference library. Query embeddings and reference embeddings must be created with the same model/weight configuration.

`POST /api/reference/build` performs the Phase 8 reference preparation work. It resolves a normal-reference folder, processes up to `FABRIC_REFERENCE_MAX_IMAGES` images by default, extracts patch embeddings, and saves:

- embeddings: `storage/indexes/{reference_version}_embeddings.npy`
- FAISS index: `storage/indexes/{reference_version}_faiss.index`
- metadata: `storage/metadata/{reference_version}_metadata.json`
- latest pointer: `storage/metadata/latest_reference.json`
- reference registry: `storage/metadata/reference_registry.json`

For example:

```powershell
python scripts/build_reference_library.py --dataset-name dme_20250129 --max-images 10
python scripts/build_reference_library.py --dataset-name wfdd --fabric-type grey_cloth --max-images 10
python scripts/build_reference_library.py --dataset-name reference_ready --fabric-type floral_print --max-images 25
```

### Train with all Company and external images

Create the unified manifest first:

```powershell
python scripts/prepare_extended_datasets.py
```

The main output is `storage/metadata/extended_dataset_manifests/raw_all.csv` (`extended_all.csv` is kept as a compatible alias). It contains all Company, DME, WFDD input photos, and all Lusitano `nondefects` photos. The report at `manifest_report.json` records the count from each source.

Train the projection head with every manifest image:

```powershell
python scripts/train_extended_projection_head.py --manifest-csv storage/metadata/extended_dataset_manifests/raw_all.csv --epochs 20 --batch-size 16 --output storage/models/projection_head_raw_all_v1.pt --report-output storage/metadata/training_reports/projection_head_raw_all_v1_report.json
```

`--max-images-per-group` is optional and has no default cap. Omit it for the full dataset; use a small value only for a smoke test. Every image is processed, while 16 spatially distributed patches per image are selected by default to control memory and training time. Change `--max-patches-per-image` as needed, or pass `0` to use every patch.

Build separate normal-reference libraries for DME, each WFDD fabric, Lusitano, and Company when a Company normal folder exists:

```powershell
python scripts/build_extended_references.py --variant raw_all_v1
```

This command now uses every available normal image by default and includes Lusitano automatically. Use `--max-images N` for a quick test or `--exclude-lusitano` when intentionally omitting that large normal set. References remain separate by fabric type because combining unrelated fabric textures into one FAISS index can weaken anomaly scoring.

If a long build is interrupted, preview and resume the same variant without rebuilding completed compatible references:

```powershell
python scripts/build_extended_references.py --variant raw_all_v2 --resume --plan
python scripts/build_extended_references.py --variant raw_all_v2 --resume
```

Resume validates the registered metadata, embedding/index files, projection checkpoint, model configuration, image cap, and source-image count before skipping an existing reference. Use a new variant whenever the model or intended reference dataset changes.

After training the extended projection head, enable it in `.env`, rebuild separate references, and evaluate:

```text
FABRIC_USE_PROJECTION_HEAD=true
FABRIC_PROJECTION_HEAD_PATH=storage/models/projection_head_raw_all_v1.pt
```

```powershell
python scripts/build_extended_references.py --variant raw_all_projection_v1
python scripts/evaluate_extended_pipeline.py --include-cross-reference
python scripts/evaluate_extended_pipeline.py --include-cross-reference --update-reference-thresholds
```

Keep `FABRIC_DEFAULT_REFERENCE_ID=dme_plain_v1` until the extended DME result maintains recall and improves or preserves baseline quality.

Query the latest saved FAISS index by reference embedding offset:

```powershell
python scripts/search_reference_index.py --query-offset 0 --k 5
```

On Windows, the backend sets `KMP_DUPLICATE_LIB_OK=TRUE` inside the FAISS module because the available PyTorch/torchvision and `faiss-cpu` wheels can load different OpenMP runtimes in one process. This keeps the local MVP usable for combined CNN embedding extraction and FAISS search.

Evaluate DME thresholds before training:

```powershell
python scripts/evaluate_pipeline.py --max-images-per-class 5
```

The evaluator runs the current pretrained CNN + FAISS pipeline over DME `non defective` and defect folders, sweeps anomaly thresholds, and writes a JSON report to `storage/metadata/dme_evaluation_latest.json`. Use the reported `best_threshold.threshold` as a candidate for `FABRIC_ANOMALY_THRESHOLD`, then rebuild/rerun evaluation before considering fine-tuning.

False-positive controls:

- `FABRIC_GOOD_AREA_THRESHOLD`: minimum merged defect area ratio before an image stops being `Good`
- `FABRIC_MIN_DEFECT_CLUSTER_PATCHES`: minimum number of connected suspicious patches needed for a merged defect box
- `FABRIC_ANOMALY_THRESHOLD`: patch anomaly threshold; increasing it usually reduces false positives but can reduce recall

Evaluator equivalents:

```powershell
python scripts/evaluate_pipeline.py --split-manifest storage/metadata/latest_dme_split.json --min-area-ratio 0.05 --min-cluster-patches 10
```

For a no-leakage held-out DME evaluation, create a split manifest, rebuild the reference library only from `reference_normal`, then evaluate the `evaluation` section:

```powershell
python scripts/prepare_dme_split.py --reference-normal-ratio 0.7 --seed 42
python scripts/build_reference_library.py --dataset-name dme_20250129 --split-manifest storage/metadata/latest_dme_split.json
python scripts/evaluate_pipeline.py --split-manifest storage/metadata/latest_dme_split.json --max-images-per-class 5
```

For a production-style DME reference library, use all currently available DME normal images:

```powershell
python scripts/build_reference_library.py --dataset-name dme_20250129
```

For fabric subtype separation, build and evaluate WFDD references per cloth folder:

```powershell
python scripts/evaluate_wfdd_fabrics.py --fabric grey_cloth --max-reference-images 25 --max-images-per-class 10
python scripts/evaluate_wfdd_fabrics.py --max-reference-images 25 --max-images-per-class 10
```

This writes `storage/metadata/wfdd_fabric_thresholds_latest.json`, including candidate thresholds per fabric subtype.

Training-ready projection head workflow:

```powershell
python scripts/train_projection_head.py --split-manifest storage/metadata/latest_dme_split.json --epochs 20 --output storage/models/projection_head_dme.pt
```

After training finishes, enable the head in `.env`:

```text
FABRIC_USE_PROJECTION_HEAD=true
FABRIC_PROJECTION_HEAD_PATH=storage/models/projection_head_dme.pt
```

Then rebuild the reference library and rerun held-out evaluation:

```powershell
python scripts/build_reference_library.py --dataset-name dme_20250129 --split-manifest storage/metadata/latest_dme_split.json
python scripts/evaluate_pipeline.py --split-manifest storage/metadata/latest_dme_split.json --max-images-per-class 999
```
