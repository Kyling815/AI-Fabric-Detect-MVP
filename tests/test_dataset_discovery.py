from backend.app.services.dataset_service import DatasetService


def test_dataset_discovery_finds_configured_datasets() -> None:
    discovery = DatasetService().discover()
    names = {dataset.name for dataset in discovery.datasets}

    assert {"dme_20250129", "wfdd", "mvtec_ad"} <= names
    assert discovery.total_images > 0


def test_dme_discovery_finds_normal_reference_folder() -> None:
    discovery = DatasetService().discover()
    dme = next(dataset for dataset in discovery.datasets if dataset.name == "dme_20250129")

    assert dme.exists
    assert any("non defective" in path.lower() for path in dme.normal_paths)
