# Preprocessing v2 implementation work log

## Existing architecture found

- `backend/app/ai/preprocess.py`: v1 Pillow decode inspection and aspect-ratio downscale plan.
- `backend/app/ai/patching.py`: deterministic fixed-size Pillow patch grid with edge inclusion.
- `backend/app/ai/feature_extractor.py`: ResNet-family CNN embeddings and ImageNet normalization.
- `backend/app/ai/reference_library.py`: normal-image patches, embeddings, NumPy storage, cosine FAISS index, metadata, and registry update.
- `backend/app/ai/anomaly_detector.py`: nearest-reference search and cosine/L2 anomaly conversion.
- `backend/app/services/inspection_service.py`: upload, repeated v1 preparation, embeddings, anomaly boxes, and Good/Acceptable/Bad classification.
- `scripts/evaluate_pipeline.py`: repeated v1 preparation, validation scoring, threshold sweep, confusion metrics, and best-F1 selection.
- `scripts/train_projection_head.py`: grouped/split-manifest image collection, repeated v1 patch preparation, frozen-backbone embeddings, and projection-head training.
- `backend/app/services/reference_registry.py`: schema-v1 reference lookup with per-reference thresholds.
- Tests use pytest. Configuration uses Pydantic settings and `FABRIC_` environment variables.

The pre-existing `frontend/package-lock.json` and `frontend/vite.config.ts` modifications were unrelated and were not changed as part of this work.

## Implementation decisions

- Preserve the existing v1 modules and paths.
- Add v2 under `backend/app/ai/preprocessing`.
- Centralize v1/v2 dispatch in `preprocessing/adapter.py` for training, references, evaluation, and inference.
- Treat preprocessing version plus canonical configuration hash as the reference compatibility contract.
- Keep orientation, color normalization, sharpening, multiscale patches, texture view, and FFT disabled by default.
- Keep batch processing non-destructive and non-overwriting by default.
- Keep source split selection outside patch extraction and record the split on every patch.

## Verification

- Added v2 unit and temporary-directory integration tests.
- Generated a real four-panel preview from a captured DME raw image.
- Full pytest result: 54 passed, one third-party FAISS/NumPy deprecation warning.
