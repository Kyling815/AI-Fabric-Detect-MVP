"""Phase 9 TEMP synthetic checkpoint + isolated shadow inference, never production activation."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.segmentation_shadow import SupervisedShadowConfig
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner


def demo() -> dict:
    # Training is exclusively a temporary software fixture inherited from Phase 8.
    from backend.app.ai.segmentation_dataset import sha256_file
    from backend.app.ai.segmentation_trainer import train_segmentation_baseline
    from scripts.check_segmentation_pipeline import create_demo_manifest
    from scripts.train_segmentation_baseline import demo_config
    with TemporaryDirectory(prefix="fabric-phase9-shadow-") as directory:
        root = Path(directory)
        manifest = create_demo_manifest(root)
        candidate = root / "phase9_tiny_synthetic_v1"
        training = train_segmentation_baseline(manifest, demo_config(), output_dir=candidate, allow_synthetic=True)
        checkpoint = candidate / training["best_checkpoint"]
        config = SupervisedShadowConfig(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                    expected_model_version=candidate.name, trusted_local=True, allowed_fabric_ids=("fabric_demo",),
                    allow_synthetic=True, timeout_seconds=15)
        runner = ProcessSupervisedShadowRunner()
        image = Path(manifest.samples[1].image_path)
        first = runner.observe(image, "fabric_demo", config)
        second = runner.observe(image, "fabric_demo", config)
        timeout = runner.observe(image, "fabric_demo", config.model_copy(update={"timeout_seconds": 0.05}))
        unavailable = runner.observe(image, "fabric_demo", SupervisedShadowConfig())
        if first.status != "observed" or second.status != "observed" or first.mask_sha256 != second.mask_sha256:
            raise AssertionError("Real isolated shadow worker did not produce deterministic fixture evidence")
        if timeout.status != "timeout" or unavailable.status != "unavailable":
            raise AssertionError("Shadow failure isolation demo failed")
        return {"phase": 9, "status": "SYNTHETIC_SOFTWARE_CHECK_ONLY", "observed": first.audit_payload(),
                "deterministic_mask_across_workers": True, "timeout": timeout.audit_payload(),
                "unavailable": unavailable.audit_payload(), "artifacts_persisted": False,
                "production_model_created": False, "production_accuracy_evaluated": False, "registry_modified": False,
                "public_response_schema_modified": False, "qc_integration": False,
                "note": "Tiny temporary pretrained synthetic checkpoint only. API/QC parity is checked separately by golden integration tests; no fusion, physical localization or production accuracy claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
