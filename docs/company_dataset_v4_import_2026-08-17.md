# Company curated dataset v4 import — 2026-08-17

## Import source

- Archive: `company_fabric_additional_dataset.zip`
- SHA-256: `189EAF075BE298F893815EC8FA83F1FA05824AD5F7EA36E724649FF89AA15DD4`
- Dataset root: `Data/raw/Company/fabrics`
- Dataset version: `company_demo_curated_manifest_v4`
- Fabric A (`fabric_001`): warp-knitted mesh / tricot-like knitted fabric
- Fabric B (`fabric_002`): brushed pile / velour-like fabric

The archive was imported without overwriting an existing file. All 250 images and 120 masks decode successfully, every defect mask exists and matches its image dimensions, and no image hash is duplicated internally or against the pre-existing Company dataset.

## Authoritative split counts

| Fabric | Reference real good | Adapter-train defect | Validation good | Validation defect | Locked good | Locked defect | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fabric_001` | 33 | 28 | 8 | 12 | 20 | 20 | 121 |
| `fabric_002` | 40 | 27 | 9 | 13 | 20 | 20 | 129 |

The original per-fabric `manifest.csv` files are authoritative. The older Company images remain on disk for provenance, but they are not appended to the v4 manifests because some new crops descend from those original images. Mixing the old and new split definitions would create source leakage.

Generated pipeline manifests are under `storage/metadata/company_demo_manifests`:

- `fabric_001_reference.csv`: 33 rows
- `fabric_001_adapter_train.csv`: 28 rows
- `fabric_001_validation.csv`: 20 rows
- `fabric_001_locked_test.csv`: 40 rows
- `fabric_002_reference.csv`: 40 rows
- `fabric_002_adapter_train.csv`: 27 rows
- `fabric_002_validation.csv`: 22 rows
- `fabric_002_locked_test.csv`: 40 rows

## Leakage policy

`reference`, `adapter_train`, and legacy `development_only` are one development partition. A source may occur in both reference and adapter-train because synthetic defects can be generated from a reference-good source. The import records 17 such sources for Fabric A and 14 for Fabric B.

Sources may not cross from development into validation or locked test, and validation sources may not cross into locked test. The v4 manifest report has zero cross-partition leakage.

## Current state and next run

This import only prepares and validates the dataset. It does not retrain an adapter, rebuild references, calibrate thresholds, or score the locked test. Therefore the current adapter/reference artifacts must not be described as v4 artifacts yet.

Prepare the manifests again at any time with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_company_demo_manifests.py
```

For the next model iteration, train a new versioned adapter (for example `shared_dense_adapter_company_v4.pt`), update the matching adapter identity/path in `config/company_demo_fabrics.json`, build new versioned references for both fabrics, calibrate on validation only, and run locked evaluation exactly once for the release decision.

## Verification

- Curated import tests: 3 passed
- Full regression suite: 88 passed
- Manifest warnings: 0
- Cross-partition source leakage: 0
- Locked test used for training/calibration during this import: no
