from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base


def new_uuid() -> str:
    return str(uuid4())


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    original_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    annotated_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_mode: Mapped[str | None] = mapped_column(String(40), nullable=True)
    final_result: Mapped[str] = mapped_column(String(32), default="Pending")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    defect_area_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    defect_count: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str] = mapped_column(String(120), default="not_built")
    fabric_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    fabric_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    reference_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    backbone_version: Mapped[str | None] = mapped_column(String(160), nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    defects: Mapped[list["DetectedDefect"]] = relationship(back_populates="inspection")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="inspection")


class DetectedDefect(Base):
    __tablename__ = "detected_defects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"))
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(80), default="suspicious_defect")
    area_pixels: Mapped[int | None] = mapped_column(Integer, nullable=True)

    inspection: Mapped[Inspection] = relationship(back_populates="defects")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"))
    user_label: Mapped[str] = mapped_column(String(80))
    corrected_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="feedback")


class ReferenceImage(Base):
    __tablename__ = "reference_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    dataset_name: Mapped[str] = mapped_column(String(120))
    fabric_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    image_path: Mapped[str] = mapped_column(String(500))
    split: Mapped[str] = mapped_column(String(40), default="train")
    label: Mapped[str] = mapped_column(String(80), default="good")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReferencePatch(Base):
    __tablename__ = "reference_patches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    reference_image_id: Mapped[str] = mapped_column(ForeignKey("reference_images.id"))
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)
    embedding_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    model_name: Mapped[str] = mapped_column(String(120))
    patch_size: Mapped[int] = mapped_column(Integer)
    patch_stride: Mapped[int] = mapped_column(Integer)
    image_max_size: Mapped[int] = mapped_column(Integer)
    anomaly_threshold: Mapped[float] = mapped_column(Float)
    reference_version: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    source_path: Mapped[str] = mapped_column(String(500))
    dataset_type: Mapped[str] = mapped_column(String(80))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
