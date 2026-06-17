# AI Fabric Detect MVP

AI Fabric Detect MVP is a local-first computer vision web application for fabric defect inspection. The MVP uses lightweight CNN / vision embeddings, patch comparison, and a normal-reference library to detect suspicious texture anomalies in close-up fabric images.

The core defect detection pipeline is not an LLM. It is designed around pretrained computer vision encoders, patch embeddings, FAISS or cosine similarity search, anomaly scoring, and rule-based Good / Acceptable / Bad classification.

## Current Phase

Phase 1 is the project setup and backend skeleton:

- FastAPI backend application structure
- Environment and path configuration
- SQLite database models
- API router skeletons
- Service and AI module placeholders for later phases
- Dataset discovery for the current DME, WFDD, and MVTec-style folders
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

## Planned Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

API docs will be available at:

```text
http://127.0.0.1:8000/docs
```

Environment variables use the `FABRIC_` prefix, as shown in `.env.example`, to avoid collisions with system-wide variables.

## Dataset and Artifact Layout

The project keeps original data, optional prepared image inputs, and generated AI artifacts separate:

```text
Data/
  raw/                         original datasets; do not modify during pipeline runs
    Company/
    external/
      20250129_Fabric_defects/
        Dataset/
          broken thread/
          damage/
          holes/
          non defective/
          Stain/
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

Extended multi-dataset workflow:

```powershell
python scripts/prepare_extended_datasets.py
python scripts/build_extended_references.py --variant v1 --include-lusitano
python scripts/train_extended_projection_head.py --manifest-csv storage/metadata/extended_dataset_manifests/extended_all.csv --epochs 20 --batch-size 16 --output storage/models/projection_head_extended_v1.pt --report-output storage/metadata/training_reports/projection_head_extended_v1_report.json
```

After training the extended projection head, enable it in `.env`, rebuild separate references, and evaluate:

```powershell
python scripts/build_extended_references.py --variant extended_v1 --include-lusitano
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
