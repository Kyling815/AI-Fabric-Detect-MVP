from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.database import Base
from backend.app.db.models import Inspection
from backend.app.db.schemas import FeedbackRequest
from backend.app.services.feedback_service import FeedbackService


def _session(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{(tmp_path / 'feedback.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    return Session(engine)


def test_feedback_rejects_unknown_inspection(tmp_path: Path) -> None:
    with _session(tmp_path) as db:
        with pytest.raises(HTTPException) as exc_info:
            FeedbackService(db).create_feedback(
                FeedbackRequest(inspection_id="missing", user_label="false_positive")
            )
    assert exc_info.value.status_code == 404


def test_feedback_accepts_existing_inspection(tmp_path: Path) -> None:
    with _session(tmp_path) as db:
        inspection = Inspection(filename="fabric.png")
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
        inspection_id = inspection.id

        response = FeedbackService(db).create_feedback(
            FeedbackRequest(inspection_id=inspection_id, user_label="confirmed")
        )

    assert response.inspection_id == inspection_id
    assert response.status == "created"
