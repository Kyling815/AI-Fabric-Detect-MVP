from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.schemas import FeedbackRequest, FeedbackResponse
from backend.app.services.feedback_service import FeedbackService

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    service = FeedbackService(db)
    return service.create_feedback(request)

