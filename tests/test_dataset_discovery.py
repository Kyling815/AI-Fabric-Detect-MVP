from pathlib import Path

from PIL import Image

from backend.app.core.config import Settings
from backend.app.services.dataset_service import DatasetService


def _service_with_fixture_dataset(tmp_path: Path) -> DatasetService:
    dme = tmp_path / "dme"
    normal = dme / "non defective"
    normal.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="white").save(normal / "sample.png")
    company = tmp_path / "raw" / "Company"
    company.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="black").save(company / "IQC_Stain 1.png")
    lusitano = tmp_path / "raw" / "external" / "Lusitano" / "nondefects"
    lusitano.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="gray").save(lusitano / "normal.png")
    settings = Settings(
        _env_file=None,
        dme_dataset_dir=str(dme),
        raw_company_dir=str(company),
        raw_external_dir=str(tmp_path / "raw" / "external"),
        wfdd_dataset_dir=str(tmp_path / "wfdd"),
        lusitano_dataset_dir=str(tmp_path / "raw" / "external" / "Lusitano"),
        mvtec_dataset_dir=str(tmp_path / "mvtec"),
        data_raw_dir=str(tmp_path / "raw"),
        data_processed_dir=str(tmp_path / "processed"),
        reference_ready_dir=str(tmp_path / "reference_ready"),
    )
    return DatasetService(settings)


def test_dataset_discovery_finds_configured_datasets(tmp_path: Path) -> None:
    discovery = _service_with_fixture_dataset(tmp_path).discover()
    names = {dataset.name for dataset in discovery.datasets}

    assert {"company_iqc", "dme_20250129", "wfdd", "lusitano", "mvtec_ad"} <= names
    assert discovery.total_images == 3


def test_dme_discovery_finds_normal_reference_folder(tmp_path: Path) -> None:
    discovery = _service_with_fixture_dataset(tmp_path).discover()
    dme = next(dataset for dataset in discovery.datasets if dataset.name == "dme_20250129")

    assert dme.exists
    assert any("non defective" in path.lower() for path in dme.normal_paths)
