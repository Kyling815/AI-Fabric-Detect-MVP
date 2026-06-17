from fastapi import APIRouter

from backend.app.db.schemas import ModelInfoResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    service = ModelService()
    return service.get_model_info()

