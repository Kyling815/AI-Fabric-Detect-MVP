from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.schemas import InspectionCreateResponse, InspectionListResponse
from backend.app.services.inspection_service import InspectionService

router = APIRouter()


@router.post("/inspect", response_model=InspectionCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def inspect_image(
    image: UploadFile = File(...),
    reference_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> InspectionCreateResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    service = InspectionService(db)
    return service.create_placeholder_inspection(image, reference_id=reference_id)


@router.get("/inspections", response_model=InspectionListResponse)
def list_inspections(db: Session = Depends(get_db)) -> InspectionListResponse:
    service = InspectionService(db)
    return service.list_inspections()


@router.get("/inspections/{inspection_id}", response_model=InspectionCreateResponse)
def get_inspection(inspection_id: str, db: Session = Depends(get_db)) -> InspectionCreateResponse:
    service = InspectionService(db)
    inspection = service.get_inspection(inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return inspection
