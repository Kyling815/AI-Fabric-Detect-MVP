# AI Fabric Defect Detection MVP — Preprocessing Upgrade Specification

## 1. Purpose

Upgrade the image-preprocessing stage of the existing AI Fabric Defect Detection MVP so that already-captured raw fabric photos become more consistent and easier for the model to learn from.

The preprocessing system must:

- Improve visibility of real fabric defects.
- Reduce variation caused by lighting, camera settings, blur, noise, image orientation, and background.
- Preserve small faults such as broken yarns, scratches, narrow abnormal lines, holes, and stains.
- Avoid generating artificial texture or removing valid defects.
- Work with the existing CNN embedding, FAISS retrieval, anomaly-threshold, and classification pipeline.
- Produce reproducible outputs for training, validation, testing, and inference.

This upgrade focuses only on **already-captured images**, not real-time video.

---

## 2. Existing MVP Context

The current MVP uses a fabric-defect detection pipeline based on:

```text
Input fabric image
    ↓
Patch extraction
    ↓
CNN feature extraction
    ↓
Embedding generation
    ↓
FAISS similarity search
    ↓
Comparison with normal/defective reference libraries
    ↓
Anomaly score and threshold
    ↓
Good / Acceptable / Bad result
```

Current project folders include structures similar to:

```text
backend/app/ai/
Data/raw/
Data/processed/
storage/annotated/
storage/metadata/
storage/evaluation_results/
storage/models/
storage/references/
```

The preprocessing upgrade should be introduced as a new version instead of overwriting the current pipeline.

---

## 3. Main Design Principle

The preprocessing stage must make normal fabric appearance more consistent without making real defects disappear.

Do not overwrite original images.

Recommended structure:

```text
Data/
├── raw/
│   └── external/
│
├── processed/
│   ├── preprocessed_v1/
│   ├── preprocessed_v2/
│   │   ├── rgb/
│   │   ├── texture_view/
│   │   ├── rejected_quality/
│   │   └── metadata/
│   │
│   ├── patches_v2/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   │
│   └── reference_ready/
│
storage/
├── metadata/
│   ├── preprocessing_v2.csv
│   └── quality_report_v2.csv
│
└── references/
    └── reference_registry.json
```

The upgraded processing sequence should be:

```text
Raw captured photo
    ↓
Image-quality validation
    ↓
Fabric-region extraction
    ↓
Optional orientation normalization
    ↓
Illumination correction
    ↓
Conservative color normalization
    ↓
Mild edge-preserving denoising
    ↓
Local contrast enhancement
    ↓
Optional mild sharpening
    ↓
Resolution normalization
    ↓
Multi-scale overlapping patch extraction
    ↓
CNN / FAISS / classifier training
```

---

## 4. Stage 1 — Image-Quality Validation

Before preprocessing, evaluate whether an image is suitable for model training.

### 4.1 Required checks

| Problem | Suggested measurement |
|---|---|
| Corrupted image | OpenCV or Pillow decoding |
| Image too small | Minimum width and height |
| Blur | Variance of Laplacian |
| Underexposure | Percentage of very dark pixels |
| Overexposure | Percentage of very bright pixels |
| Unexpected channels | Convert grayscale, RGB, RGBA, or BGR safely |
| Invalid format | Supported-extension validation |

### 4.2 Suggested starting thresholds

```python
minimum_width = 512
minimum_height = 512
minimum_laplacian_variance = 70.0
maximum_dark_pixel_ratio = 0.20
maximum_bright_pixel_ratio = 0.20
```

These thresholds must be configurable rather than hard-coded.

### 4.3 Rejected-image handling

Do not delete rejected images.

Copy or move them to:

```text
Data/processed/preprocessed_v2/rejected_quality/
```

Save a quality report:

```csv
image_path,width,height,blur_score,dark_ratio,bright_ratio,status,reason
```

Example rejection reasons:

```text
cannot_decode
image_width_too_small
image_height_too_small
image_too_blurry
image_too_dark
image_too_bright
unsupported_channel_count
```

---

## 5. Stage 2 — Fabric Region of Interest

Background objects can cause shortcut learning. The model should see mostly fabric.

Possible background distractions include:

- Tables
- Floors
- Hands
- Labels
- Rulers
- Machine components
- Shadows outside the fabric
- Empty borders

### 5.1 Controlled-image option

When the camera position is consistent, use a configured fixed crop:

```python
fabric_roi = image[y1:y2, x1:x2]
```

This is the safest MVP option.

### 5.2 Automatic ROI option

For less controlled images:

1. Resize a temporary copy.
2. Convert to Lab or HSV.
3. Estimate the dominant foreground region.
4. Apply morphology.
5. Select the largest connected component.
6. Calculate its bounding rectangle.
7. Add a small margin.
8. Validate that the retained area is sufficiently large.

Automatic ROI must be optional because it may accidentally remove dark stains, holes, or fabric edges.

### 5.3 Configuration example

```json
{
  "roi_mode": "fixed",
  "fixed_roi": {
    "x1_ratio": 0.05,
    "y1_ratio": 0.05,
    "x2_ratio": 0.95,
    "y2_ratio": 0.95
  }
}
```

Supported modes should include:

```text
none
fixed
automatic
```

---

## 6. Stage 3 — Optional Orientation Normalization

Woven fabric usually contains directional warp and weft structures. Different image rotations can make normal fabric patches look artificially different.

### 6.1 Suggested method

1. Convert to grayscale.
2. Apply mild Gaussian smoothing.
3. Compute Sobel gradients.
4. Build a weighted orientation histogram.
5. Estimate the dominant fabric angle.
6. Rotate only when the estimate is reliable.
7. Preserve the full image with border reflection or safe cropping.

### 6.2 Safety conditions

Apply rotation only when:

- Orientation normalization is enabled for that fabric type.
- Confidence exceeds a configured threshold.
- Estimated correction is within a safe range.
- The image contains a visible regular weave.

Suggested starting rule:

```python
if confidence >= 0.60 and abs(angle_degrees) <= 15:
    rotate_image()
else:
    keep_original()
```

### 6.3 Disable for unsuitable fabrics

Disable automatic orientation correction for:

- Floral fabrics
- Irregular printed patterns
- Highly textured velvet
- Fabrics with folds
- Images without visible yarn direction

### 6.4 Registry example

```json
{
  "grey_woven": {
    "orientation_normalization": true,
    "expected_orientation_degrees": 0,
    "maximum_rotation_degrees": 15
  },
  "pink_flower": {
    "orientation_normalization": false
  }
}
```

---

## 7. Stage 4 — Illumination Correction

Uneven illumination is one of the largest sources of false positives.

Normal fabric under shadow may appear defective, while bright regions may hide small faults.

### 7.1 Recommended method

Perform correction on the Lab luminance channel:

1. Convert BGR/RGB to Lab.
2. Extract the `L` channel.
3. Estimate a smooth illumination field using a large Gaussian blur.
4. Divide the original luminance by the smooth field.
5. Rescale using the mean illumination.
6. Clip to the valid range.
7. Recombine with the original `a` and `b` channels.

Conceptual formula:

```text
L_corrected(x, y)
    = L(x, y) / (L_background(x, y) + epsilon)
      × mean(L_background)
```

### 7.2 Suggested scale

```python
sigma = 0.05 to 0.12 * min(image_height, image_width)
```

For a `1024 × 1024` image:

```text
sigma ≈ 50 to 120
```

The illumination field must be much smoother than actual yarn structures and defects.

### 7.3 Expected benefit

This stage should reduce:

- Shadows
- Vignetting
- Gradual brightness gradients
- Uneven LED illumination
- Bright center and dark border effects

It must not remove local stains or actual discoloration.

---

## 8. Stage 5 — Conservative Color Normalization

Color is important for detecting stains, discoloration, and material differences.

Do not force every fabric to match one global color distribution.

Normalize within each fabric family.

Example:

```text
grey woven fabric   → grey woven reference statistics
yellow fabric       → yellow reference statistics
floral fabric       → floral reference statistics
```

### 8.1 Reference statistics

Calculate Lab statistics from verified normal images:

```json
{
  "lab_reference_mean": [142.3, 128.4, 129.1],
  "lab_reference_std": [31.8, 4.2, 5.1]
}
```

### 8.2 Recommended normalization strength

Use full normalization for luminance and weak normalization for chromatic channels:

```json
{
  "luminance_strength": 1.0,
  "chromatic_strength": 0.25
}
```

### 8.3 Safety requirements

- Preserve the original image.
- Store normalization metadata.
- Allow chromatic normalization to be disabled.
- Do not use strong normalization for stain-heavy defect classes.
- Compute statistics from the training split only.
- Never use validation or test data to calculate reference statistics.

---

## 9. Stage 6 — Mild Edge-Preserving Denoising

Image noise can be amplified by gradients and texture filters.

Use light denoising only.

### 9.1 Recommended default

```python
cv2.bilateralFilter(
    image,
    d=5,
    sigmaColor=20,
    sigmaSpace=5
)
```

Alternative:

```python
cv2.fastNlMeansDenoisingColored(...)
```

### 9.2 Avoid

- Large median kernels
- Strong Gaussian blur
- Heavy non-local means
- Repeated filtering
- AI-based smoothing that may hallucinate texture

Over-smoothing can remove:

- Broken yarn ends
- Narrow scratches
- Tiny holes
- Thin abnormal lines

The processed image should still visibly retain the weave pattern.

---

## 10. Stage 7 — Local Contrast Enhancement

Apply CLAHE only to luminance.

Recommended starting configuration:

```python
clahe = cv2.createCLAHE(
    clipLimit=1.5,
    tileGridSize=(8, 8)
)
```

Suggested search range:

```text
clipLimit: 1.2–2.0
tileGridSize: 8×8 or 12×12
```

Expected improvements:

- Weak scratch visibility
- Subtle yarn discontinuities
- Small local texture changes
- Low-contrast holes

Avoid aggressive CLAHE because it may exaggerate normal weave texture and increase false positives.

---

## 11. Stage 8 — Optional Mild Sharpening

Sharpening must be optional and conservative.

Use a mild unsharp mask:

```text
I_sharp = I + alpha * (I - GaussianBlur(I))
```

Suggested starting values:

```python
sigma = 1.0
alpha = 0.20 to 0.40
```

Recommended default:

```python
apply_sharpening = False
```

Enable only after ablation testing.

Do not sharpen:

- Highly noisy images
- Images that fail blur validation
- Images with compression artifacts
- Fabrics whose natural weave becomes exaggerated

---

## 12. Stage 9 — Resolution Normalization

Do not directly stretch every image into a square.

Incorrect:

```python
cv2.resize(image, (512, 512))
```

Better:

1. Preserve aspect ratio.
2. Resize according to target scale.
3. Apply letterbox padding if required.
4. Use reflection padding where appropriate.
5. Extract patches from the standardized image.

For image classification:

```text
Aspect-ratio resize
    ↓
Letterbox padding
    ↓
224×224 or 256×256 model input
```

For anomaly localization:

```text
Preserve standardized high-resolution image
    ↓
Extract overlapping patches
```

---

## 13. Stage 10 — Multi-Scale Patch Extraction

The existing CNN and FAISS pipeline depends strongly on patch quality.

Apply preprocessing before patch extraction.

### 13.1 Suggested patch settings

Default:

```python
patch_size = 224
stride = 112
overlap_ratio = 0.50
```

Multi-scale option:

```python
patch_sizes = [128, 224, 384]
```

### 13.2 Defect-scale mapping

| Patch size | Typical target |
|---|---|
| `128 × 128` | Tiny holes, broken yarn, small stains |
| `224 × 224` | General defect recognition |
| `384 × 384` | Broad uneven texture, long scratches, large defects |

### 13.3 Border handling

- Use reflection padding instead of zero padding where possible.
- Record original patch coordinates.
- Save patch metadata.
- Skip patches containing too little fabric.
- Maintain mapping from each patch back to the original image.

Example metadata:

```csv
patch_path,source_image,x1,y1,x2,y2,patch_size,scale,fabric_type,split
```

---

## 14. Prevent Data Leakage

Split the dataset before patch extraction.

Correct order:

```text
Original source images
    ↓
Group by source, fabric roll, capture session, or physical sample
    ↓
Train / validation / test split
    ↓
Fit preprocessing reference statistics using training data only
    ↓
Preprocess each split
    ↓
Extract patches independently
```

Do not allow adjacent patches from the same source image or fabric roll to appear in different splits.

Potential grouping keys:

```text
fabric_roll_id
capture_session_id
source_dataset
original_image_id
physical_sample_id
```

---

## 15. Texture-Enhanced Secondary View

The main model should continue receiving normalized RGB images.

Create an optional secondary texture representation.

### 15.1 Three-channel texture view

```text
Channel 1: normalized grayscale luminance
Channel 2: maximum Gabor response
Channel 3: absolute Laplacian-of-Gaussian response
```

This view emphasizes:

- Warp/weft discontinuity
- Thin scratches
- Broken yarn
- Abnormal lines
- Local directional changes

### 15.2 Suggested Gabor orientations

```python
orientations = [
    0,
    pi / 4,
    pi / 2,
    3 * pi / 4
]
```

### 15.3 Training options

Option A — separate models:

```text
Model A: normalized RGB
Model B: texture view

Final score = 0.65 * RGB score + 0.35 * texture score
```

Option B — additional channels:

```text
RGB + texture features
```

Use Option A first because it requires fewer changes to the existing model.

The texture model should not replace RGB because stain detection depends on color.

---

## 16. Optional FFT Features

Fabric weave is periodic. Frequency features can detect disruption of this periodicity.

For each patch, optionally calculate:

- Dominant horizontal frequency
- Dominant vertical frequency
- Frequency-peak amplitude
- Spectral energy ratio
- Distance from normal reference spectrum
- Peak width
- Orientation-specific frequency energy

Store as auxiliary metadata:

```csv
patch_path,fft_horizontal_peak,fft_vertical_peak,spectral_distance
```

Recommended initial use:

- Auxiliary anomaly score
- Quality-control feature
- Classical baseline
- Input to a small secondary classifier

Do not replace the main RGB image with an FFT spectrum in the first implementation.

---

## 17. Recommended Configuration Model

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    min_width: int = 512
    min_height: int = 512

    blur_threshold: float = 70.0
    max_dark_ratio: float = 0.20
    max_bright_ratio: float = 0.20

    roi_mode: str = "none"

    apply_orientation_normalization: bool = False
    orientation_confidence_threshold: float = 0.60
    maximum_rotation_degrees: float = 15.0

    illumination_sigma_ratio: float = 0.08

    apply_color_normalization: bool = False
    luminance_strength: float = 1.0
    chromatic_strength: float = 0.25

    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 20.0
    bilateral_sigma_space: float = 5.0

    clahe_clip_limit: float = 1.5
    clahe_grid_size: tuple[int, int] = (8, 8)

    apply_sharpening: bool = False
    sharpen_alpha: float = 0.25
    sharpen_sigma: float = 1.0

    patch_sizes: tuple[int, ...] = (224,)
    patch_overlap_ratio: float = 0.50
```

---

## 18. Reference Registry Extension

Extend `storage/references/reference_registry.json`.

Example:

```json
{
  "grey_woven": {
    "preprocessing_version": "v2",
    "roi_mode": "fixed",
    "illumination_sigma_ratio": 0.08,
    "clahe_clip_limit": 1.5,
    "orientation_normalization": true,
    "expected_orientation_degrees": 0,
    "maximum_rotation_degrees": 15,
    "patch_sizes": [128, 224, 384],
    "patch_overlap_ratio": 0.5,
    "color_normalization": {
      "enabled": true,
      "luminance_strength": 1.0,
      "chromatic_strength": 0.25,
      "lab_reference_mean": [142.3, 128.4, 129.1],
      "lab_reference_std": [31.8, 4.2, 5.1]
    }
  },
  "pink_flower": {
    "preprocessing_version": "v2",
    "roi_mode": "none",
    "illumination_sigma_ratio": 0.08,
    "clahe_clip_limit": 1.3,
    "orientation_normalization": false,
    "patch_sizes": [224, 384],
    "patch_overlap_ratio": 0.5,
    "color_normalization": {
      "enabled": false
    }
  }
}
```

All missing fields must receive safe defaults.

---

## 19. Recommended Python Module Structure

```text
backend/app/ai/preprocessing/
├── __init__.py
├── config.py
├── quality.py
├── roi.py
├── orientation.py
├── illumination.py
├── color.py
├── denoise.py
├── contrast.py
├── sharpen.py
├── resize.py
├── patches.py
├── texture_view.py
├── fft_features.py
├── pipeline.py
└── cli.py
```

Suggested responsibilities:

| Module | Responsibility |
|---|---|
| `config.py` | Configuration dataclasses and loading |
| `quality.py` | Blur, exposure, resolution, and decode checks |
| `roi.py` | Fixed and automatic fabric cropping |
| `orientation.py` | Dominant-angle estimation and safe rotation |
| `illumination.py` | Luminance background correction |
| `color.py` | Fabric-specific Lab normalization |
| `denoise.py` | Mild edge-preserving denoising |
| `contrast.py` | CLAHE on luminance |
| `sharpen.py` | Optional unsharp masking |
| `resize.py` | Aspect-ratio resize and padding |
| `patches.py` | Multi-scale overlapping patch extraction |
| `texture_view.py` | Gabor and LoG secondary representation |
| `fft_features.py` | Frequency-domain patch features |
| `pipeline.py` | End-to-end orchestration |
| `cli.py` | Command-line batch processing |

---

## 20. Batch Command Examples

Example CLI:

```bash
python -m backend.app.ai.preprocessing.cli \
  --input Data/raw/external \
  --output Data/processed/preprocessed_v2 \
  --registry storage/references/reference_registry.json \
  --metadata storage/metadata/preprocessing_v2.csv \
  --split train
```

Optional arguments:

```bash
--fabric-type grey_woven
--save-texture-view
--extract-patches
--patch-sizes 128 224 384
--dry-run
--overwrite
```

The default behavior must not overwrite existing output files.

---

## 21. Metadata Requirements

For every processed image, save:

```csv
source_path,
output_path,
fabric_type,
split,
preprocessing_version,
width_before,
height_before,
width_after,
height_after,
blur_score,
dark_ratio,
bright_ratio,
quality_valid,
rejection_reasons,
roi_mode,
orientation_angle,
orientation_confidence,
orientation_applied,
illumination_applied,
color_normalization_applied,
denoising_applied,
clahe_applied,
sharpening_applied,
processing_time_ms
```

For reproducibility, also save:

- Configuration hash
- Code version or Git commit when available
- Registry version
- Processing timestamp
- Random seed when any randomized operation exists

---

## 22. Error Handling

The pipeline must handle:

- Corrupted files
- Unsupported formats
- Missing registry entries
- Empty ROI
- Very small crop
- Numerical division by zero
- Invalid patch size
- Duplicate output path
- Permission errors
- Interrupted batch processing

A failure on one image must not stop the entire batch.

Errors should be logged to:

```text
storage/metadata/preprocessing_v2_errors.csv
```

---

## 23. Testing Requirements

### 23.1 Unit tests

Create tests for:

- Quality metrics
- Quality rejection reasons
- Illumination correction output range
- Aspect-ratio preservation
- Padding dimensions
- Patch coordinates
- Patch overlap
- Orientation threshold behavior
- Registry fallback behavior
- Texture-view output shape
- FFT feature shape
- Corrupted-image handling

### 23.2 Integration tests

Create a small temporary image directory and verify:

1. Batch preprocessing completes.
2. Valid outputs are created.
3. Rejected images are recorded.
4. Metadata rows match processed files.
5. Patch coordinates are valid.
6. Running twice without `--overwrite` does not duplicate outputs.
7. The same input and configuration produce identical output.

### 23.3 Visual sanity-check utility

Generate comparison grids:

```text
Original | Illumination corrected | Final RGB | Texture view
```

Save examples to:

```text
storage/annotated/preprocessing_v2_preview/
```

---

## 24. Training-Pipeline Integration

The existing training pipeline should support selecting a preprocessing version.

Example configuration:

```yaml
dataset:
  preprocessing_version: v2
  image_root: Data/processed/preprocessed_v2/rgb
  patch_root: Data/processed/patches_v2
  use_texture_view: false
```

The training code must:

- Use v2 paths when configured.
- Keep v1 behavior available.
- Record preprocessing version in model metadata.
- Record preprocessing configuration in evaluation outputs.
- Refuse to mix incompatible preprocessing versions accidentally.
- Use the same preprocessing implementation during inference.

---

## 25. FAISS Reference Rebuild

After changing preprocessing, existing embeddings may no longer be compatible.

Required procedure:

```text
Preprocess normal reference images with v2
    ↓
Extract v2 patches
    ↓
Generate embeddings with the current backbone
    ↓
Rebuild FAISS index
    ↓
Recalculate anomaly threshold
    ↓
Evaluate on validation data
```

Store versioned references:

```text
storage/references/
├── v1/
│   ├── normal.index
│   └── metadata.json
└── v2/
    ├── normal.index
    ├── defective.index
    └── metadata.json
```

Reference metadata should include:

```json
{
  "preprocessing_version": "v2",
  "backbone": "resnet18",
  "embedding_dimension": 512,
  "patch_sizes": [224],
  "created_at": "ISO-8601 timestamp",
  "source_count": 0,
  "patch_count": 0
}
```

---

## 26. Threshold Recalibration

Do not reuse the old best threshold after preprocessing changes.

For each fabric family:

1. Generate validation embeddings.
2. Calculate anomaly scores.
3. Search candidate thresholds.
4. Record precision, recall, F1, false-positive rate, and false-negative rate.
5. Select a threshold according to project goals.
6. Store the selected threshold by fabric and preprocessing version.

Example:

```json
{
  "grey_woven": {
    "v1": {
      "best_threshold": 0.075
    },
    "v2": {
      "best_threshold": 0.061
    }
  }
}
```

---

## 27. Ablation Plan

Do not enable every operation simultaneously.

Evaluate the following versions:

| Version | Processing |
|---|---|
| `P0` | Current preprocessing or resize only |
| `P1` | Quality validation + illumination correction |
| `P2` | `P1` + mild denoising |
| `P3` | `P2` + CLAHE |
| `P4` | `P3` + orientation normalization |
| `P5` | `P3` + multi-scale patches |
| `P6` | `P5` + texture-view model |
| `P7` | `P6` + FFT auxiliary score |

Measure:

- Accuracy
- Precision
- Recall
- F1
- True positives
- False positives
- True negatives
- False negatives
- Per-fabric performance
- Per-defect performance
- Average normal anomaly score
- Average defective anomaly score
- Score-distribution separation
- Processing time per image
- Patch count
- FAISS search time

---

## 28. Recommended First Production Candidate

Start with:

```text
Quality validation
+ verified fabric ROI
+ luminance illumination correction
+ mild bilateral filtering
+ mild CLAHE
+ aspect-ratio-preserving resize
+ overlapping 224×224 patches
```

Recommended initial settings:

```python
blur_threshold = 70.0
illumination_sigma_ratio = 0.08

bilateral_diameter = 5
bilateral_sigma_color = 20.0
bilateral_sigma_space = 5.0

clahe_clip_limit = 1.5
clahe_grid_size = (8, 8)

patch_sizes = (224,)
patch_overlap_ratio = 0.50

apply_orientation_normalization = False
apply_color_normalization = False
apply_sharpening = False
```

Then evaluate optional features individually.

---

## 29. Definition of Done

The preprocessing upgrade is complete when:

- Original raw data remains unchanged.
- `preprocessing_v2` can process a directory safely.
- Invalid images are rejected with documented reasons.
- Valid images receive deterministic outputs.
- Metadata is generated for every image.
- Images preserve real defects and fine weave structure.
- Patch extraction preserves source coordinates.
- Train, validation, and test data remain isolated.
- The FAISS reference index is rebuilt with v2 images.
- The anomaly threshold is recalibrated.
- v1 and v2 can be compared through the same evaluation script.
- Unit and integration tests pass.
- A visual before/after report is generated.
- README documentation explains how to use the new pipeline.
