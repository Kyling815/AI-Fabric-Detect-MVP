from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.core.acquisition import AcquisitionMetadata
from backend.app.core.config import Settings
from backend.app.core.inspection_contracts import InspectionInput


def test_existing_input_constructor_keeps_metadata_optional() -> None:
    value = InspectionInput(Path("capture.png"), "fabric_001")
    assert value.metadata == AcquisitionMetadata()
    assert value.fabric_id == "fabric_001"


def test_acquisition_metadata_is_frozen_and_preserves_fields() -> None:
    timestamp = datetime(2026, 10, 13, tzinfo=timezone.utc)
    metadata = AcquisitionMetadata("LOT_1", "ROLL_1", "CAM_1", "FRAME_1", timestamp, 42.5)
    value = InspectionInput(Path("capture.png"), "fabric_001", metadata)
    assert value.metadata is metadata and metadata.timestamp is timestamp
    assert metadata.encoder_position == 42.5
    with pytest.raises(FrozenInstanceError):
        metadata.roll_id = "OTHER"


def test_quality_feature_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FABRIC_QUALITY_GATE_ENABLED", raising=False)
    monkeypatch.delenv("FABRIC_QUALITY_GATE_CONFIG_PATH", raising=False)
    settings = Settings(_env_file=None)
    assert settings.quality_gate_enabled is False
    assert settings.quality_gate_config_path == ""
