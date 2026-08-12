from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.db import crud
from backend.app.db.models import Feedback
from backend.app.db.schemas import FeedbackRequest, FeedbackResponse


class FeedbackService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        if crud.get_inspection(self.db, request.inspection_id) is None:
            raise HTTPException(status_code=404, detail="Inspection not found.")
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
