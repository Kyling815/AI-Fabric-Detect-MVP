from sqlalchemy.orm import Session

from backend.app.db import crud
from backend.app.db.models import Feedback
from backend.app.db.schemas import FeedbackRequest, FeedbackResponse


class FeedbackService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        feedback = Feedback(
            inspection_id=request.inspection_id,
            user_label=request.user_label,
            corrected_result=request.corrected_result,
            notes=request.notes,
        )
        feedback = crud.create_feedback(self.db, feedback)
        return FeedbackResponse(
            feedback_id=feedback.id,
            inspection_id=feedback.inspection_id,
            status="created",
        )

