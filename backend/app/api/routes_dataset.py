from fastapi import APIRouter

from backend.app.db.schemas import DatasetDiscoveryResponse, DatasetPathsResponse
from backend.app.services.dataset_service import DatasetService

router = APIRouter()


@router.get("/datasets/paths", response_model=DatasetPathsResponse)
def dataset_paths() -> DatasetPathsResponse:
    service = DatasetService()
    return service.get_configured_paths()


@router.get("/datasets/discover", response_model=DatasetDiscoveryResponse)
def discover_datasets() -> DatasetDiscoveryResponse:
    service = DatasetService()
    return service.discover()
