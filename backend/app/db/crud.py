from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import DetectedDefect, Feedback, Inspection, ModelVersion


def create_inspection(db: Session, inspection: Inspection) -> Inspection:
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    return inspection


def list_inspections(db: Session) -> list[Inspection]:
    statement = select(Inspection).order_by(Inspection.created_at.desc())
    return list(db.scalars(statement).all())


def get_inspection(db: Session, inspection_id: str) -> Inspection | None:
    return db.get(Inspection, inspection_id)


def create_feedback(db: Session, feedback: Feedback) -> Feedback:
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def create_model_version(db: Session, model_version: ModelVersion) -> ModelVersion:
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return model_version


def create_detected_defects(db: Session, defects: list[DetectedDefect]) -> list[DetectedDefect]:
    if not defects:
        return []
    db.add_all(defects)
    db.commit()
    for defect in defects:
        db.refresh(defect)
    return defects
