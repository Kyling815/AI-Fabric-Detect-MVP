# Fabric preprocessing v2

## Why v2 exists

The legacy v1 path resizes a decoded RGB image and extracts a fixed patch grid. It remains available for existing models and FAISS references. Preprocessing v2 is a separately versioned, deterministic pipeline for already-captured fabric photos. It reduces camera and lighting variation while retaining real holes, stains, broken yarns, scratches, and weave structure.

Raw inputs are read only. Generated files are written under `Data/processed` and `storage`.

## Stable sequence

The first production candidate performs:

1. Decode and quality validation.
2. Optional fabric ROI (`none` by default; fixed and conservative automatic modes are supported).
3. Optional orientation normalization (off by default).
4. Lab luminance illumination correction.
5. Optional per-fabric Lab color normalization (off by default).
6. Mild bilateral denoising.
7. Mild CLAHE on Lab `L`.
8. Optional unsharp mask (off by default).
9. Optional aspect-ratio-preserving resize and reflection letterbox.
10. Deterministic, overlapping reflection-padded patches.

Optional texture-view and FFT features never replace the primary RGB input.

## Output layout

```text
Data/processed/
  preprocessed_v2/
    rgb/{train|validation|test}/
    texture_view/{train|validation|test}/
    rejected_quality/{train|validation|test}/
    metadata/
  patches_v2/{train|validation|test}/

storage/
  annotated/preprocessing_v2_preview/
  evaluation_results/v2/
  references/v2/
```

Split source images before invoking patch extraction. Pass the correct `--split` on each independent source-image split. Never put images from one roll, session, sample, or original source into multiple splits.

## Configuration

The checked-in starting configuration is `config/preprocessing_v2.json`. `PreprocessConfig` validates all ranges and produces a stable SHA-256 hash from canonical JSON.

Important fields:

- Quality: `min_width`, `min_height`, `blur_threshold`, `max_dark_ratio`, `max_bright_ratio`.
- ROI: `roi_mode`, `fixed_roi`, `automatic_roi_min_area_ratio`, `automatic_roi_margin_ratio`.
- Orientation: `apply_orientation_normalization`, confidence threshold, expected angle, maximum correction.
- Illumination: `apply_illumination_correction`, `illumination_sigma_ratio`.
- Color: enable flag, Lab training-derived mean/std, luminance and chromatic strengths.
- Denoise/contrast: bilateral parameters, CLAHE clip limit and grid.
- Sharpen: enable flag, alpha, sigma.
- Resolution: aspect-ratio-preserving `target_max_size` (1024 by default) and optional `letterbox_size`.
- Patches: `patch_sizes`, overlap ratio, optional minimum fabric coverage.
- Auxiliary views: `generate_texture_view`, `calculate_fft_features`.

JSON is always supported. YAML is accepted only when PyYAML is installed; no new YAML dependency is required.

## Registry integration

Existing `schema_version: 1` registries and `references` entries remain valid. Fabric overrides may be added without rewriting current entries:

```json
{
  "schema_version": 1,
  "references": [],
  "fabric_preprocessing": {
    "grey_woven": {
      "preprocessing_version": "v2",
      "roi_mode": "none",
      "illumination_sigma_ratio": 0.08,
      "clahe_clip_limit": 1.5,
      "orientation_normalization": false,
      "patch_sizes": [224],
      "patch_overlap_ratio": 0.5,
      "color_normalization": {
        "enabled": false
      }
    }
  }
}
```

The loader also accepts a legacy direct top-level fabric key. Missing fields use `PreprocessConfig` defaults. Malformed JSON, invalid values, and non-v2 fabric overrides raise clear errors. The preprocessing CLI never rewrites the registry.

Color reference statistics must be fitted only from verified normal training images. Do not derive them from validation or test data.

## Batch preprocessing

PowerShell:

```powershell
python -m backend.app.ai.preprocessing.cli `
  --input Data/raw/external `
  --output Data/processed/preprocessed_v2 `
  --registry storage/metadata/reference_registry.json `
  --metadata storage/metadata/preprocessing_v2.csv `
  --config config/preprocessing_v2.json `
  --split train `
  --extract-patches
```

Useful options:

```text
--fabric-type grey_woven
--save-texture-view
--patch-sizes 128 224 384
--dry-run
--overwrite
```

Overwrite is off by default. Individual decode, quality, or write failures are recorded without aborting the batch. Rejected source files are copied, never moved or deleted. Metadata includes split, operations, ROI/orientation values, quality metrics, config hash, commit, timestamp, and processing time.

Process validation and test roots separately:

```powershell
python -m backend.app.ai.preprocessing.cli --input Data/splits/validation --output Data/processed/preprocessed_v2 --split validation --extract-patches --config config/preprocessing_v2.json
python -m backend.app.ai.preprocessing.cli --input Data/splits/test --output Data/processed/preprocessed_v2 --split test --extract-patches --config config/preprocessing_v2.json
```

## Training

The projection-head trainer selects v1 or v2 explicitly and records the preprocessing config and hash in both its checkpoint extras and JSON report:

```powershell
python scripts/train_projection_head.py `
  --split-manifest storage/metadata/latest_dme_split.json `
  --preprocessing-version v2 `
  --preprocessing-config config/preprocessing_v2.json `
  --output storage/models/projection_head_dme_v2.pt
```

Training, reference construction, evaluation, and runtime inference all call the same `preprocessing.adapter.prepare_image` function. Dataset loaders do not contain a duplicate v2 implementation.

## Rebuild FAISS references

Do not reuse v1 embeddings. Build new normal and optionally defective indexes:

```powershell
python scripts/rebuild_v2_references.py `
  --normal-path "Data/raw/external/20250129_Fabric_defects/Dataset/non defective" `
  --defective-path "Data/raw/external/20250129_Fabric_defects/Dataset/holes" `
  --fabric-type dme_plain `
  --config config/preprocessing_v2.json
```

The command refuses to overwrite by default and writes:

```text
storage/references/v2/normal.index
storage/references/v2/defective.index
storage/references/v2/metadata.json
```

The general reference builder also supports v2:

```powershell
python scripts/build_reference_library.py --dataset-name dme_20250129 --preprocessing-version v2 --preprocessing-config config/preprocessing_v2.json
```

Reference metadata records the version, exact configuration, hash, backbone, embedding dimension, patch/source counts, and creation time. Runtime inference chooses its preparation path from this metadata, so a v1 query cannot silently use a v2 reference contract.

## Threshold recalibration

An old threshold is not valid after preprocessing changes:

```powershell
python scripts/recalibrate_v2_thresholds.py `
  --dataset-root "Data/raw/external/20250129_Fabric_defects/Dataset" `
  --metadata-path storage/references/v2/metadata.json `
  --fabric-type dme_plain `
  --split-manifest storage/metadata/latest_dme_split.json `
  --update-reference-metadata
```

Reports are stored under `storage/evaluation_results/v2/<fabric>/`. They contain accuracy, precision, recall, F1, TP, FP, TN, FN, false-positive rate, false-negative rate, every candidate threshold, the chosen threshold, preprocessing version, and configuration hash. Run one calibration per fabric family for per-fabric thresholds.

Without `--update-reference-metadata`, the command only writes the evaluation report. The explicit update flag stores the chosen threshold in the reference metadata and matching registry entry. Runtime inference refuses to use an uncalibrated v2 reference rather than falling back to the old global v1 threshold.

## Ablations and v1/v2 comparison

Generate P0-P7 configs and aggregate completed evaluation reports:

```powershell
python scripts/compare_preprocessing_ablations.py --base-config config/preprocessing_v2.json
```

This creates `config/ablations/P0.json` through `P7.json`. Build a matching reference and evaluation for each config under `storage/evaluation_results/ablations/P*/`, then rerun the command to create `comparison.csv`.

For a direct v1/v2 comparison, evaluate both reference metadata files with `scripts/evaluate_pipeline.py`, then place the reports in named ablation folders or compare their `best_threshold` records.

## Visual inspection

```powershell
python -m backend.app.ai.preprocessing.preview `
  --input Data/raw/external/20250129_Fabric_defects/Dataset `
  --output storage/annotated/preprocessing_v2_preview `
  --config config/preprocessing_v2.json `
  --limit 12
```

Each grid contains original, illumination-corrected, final RGB, and texture view panels plus quality metrics.

## Tests

```powershell
pytest -q
pytest -q tests/test_preprocessing_v2.py
```

The v2 suite covers quality/rejections, flat/dark/gradient illumination, ROI fallbacks, orientation safety, denoise/CLAHE determinism, resizing, padding, coordinates/overlap, texture and FFT shapes, registry fallback, corrupt input, batch isolation, non-overwrite behavior, and deterministic output.

## Common failures

- `image_too_blurry`: recapture; sharpening intentionally does not rescue rejected images.
- `image_too_dark` / `image_too_bright`: correct capture exposure. Do not weaken thresholds without inspecting examples.
- Automatic ROI fallback: use `none` or a verified fixed crop for patterned/irregular fabric.
- Color normalization config error: provide all three Lab mean/std values derived from training normals, or disable it.
- Configuration-hash mismatch: rebuild the FAISS reference and recalibrate thresholds with the exact config used by inference.
- Too many patches: return to `[224]` before enabling multiscale patches.

## Adding a fabric family

1. Group source images by roll/session/sample, then create train/validation/test source splits.
2. Add a registry override for the fabric, starting with all optional modules disabled.
3. If needed, derive Lab statistics from verified normal training images only.
4. Preprocess and visually inspect representative normal and defective captures.
5. Build a dedicated v2 reference.
6. Recalibrate a dedicated threshold.
7. Compare P0-P3 first; enable orientation, multiscale, texture, and FFT one at a time.

## Known limitations

- Automatic ROI is intentionally conservative and is not semantic segmentation.
- Orientation estimation assumes visible, regular directional texture.
- FFT features are metadata only; score fusion is an extension point, not enabled in the primary anomaly score.
- Texture view is generated and stored but needs a separately evaluated model before score fusion.
- The runtime response’s legacy preprocess summary still describes the v1 resize schema; actual patch preparation follows reference metadata.

The recommended next experiment is P3 versus P5 on held-out, roll-grouped captures. Compare recall for tiny defects against false-positive rate and patch-search latency before enabling multiscale patches in production.
