from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.schemas import (
    ReferenceBuildRequest,
    ReferenceBuildResponse,
    ReferenceRegistryListResponse,
    ReferenceStatusResponse,
)
from backend.app.services.reference_service import ReferenceService

router = APIRouter()


@router.post("/reference/build", response_model=ReferenceBuildResponse, status_code=status.HTTP_202_ACCEPTED)
def build_reference_library(
    request: ReferenceBuildRequest,
    db: Session = Depends(get_db),
) -> ReferenceBuildResponse:
    service = ReferenceService(db)
    return service.enqueue_build(request)


@router.get("/reference/status", response_model=ReferenceStatusResponse)
def reference_status(db: Session = Depends(get_db)) -> ReferenceStatusResponse:
    service = ReferenceService(db)
    return service.get_status()


@router.get("/reference/list", response_model=ReferenceRegistryListResponse)
def list_references(db: Session = Depends(get_db)) -> ReferenceRegistryListResponse:
    service = ReferenceService(db)
    return service.list_references()


@router.get("/reference/status/{reference_id}", response_model=ReferenceStatusResponse)
def reference_status_by_id(reference_id: str, db: Session = Depends(get_db)) -> ReferenceStatusResponse:
    service = ReferenceService(db)
    return service.get_reference_status(reference_id)
