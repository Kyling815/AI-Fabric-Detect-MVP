from collections.abc import Generator

from sqlalchemy import event, inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


class Base(DeclarativeBase):
    pass


def _database_url() -> str:
    settings = get_settings()
    if settings.database_url.startswith("sqlite:///./"):
        relative = settings.database_url.replace("sqlite:///./", "", 1)
        return f"sqlite:///{resolve_project_path(relative).as_posix()}"
    return settings.database_url


engine = create_engine(
    _database_url(),
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    del connection_record
    if engine.url.get_backend_name() != "sqlite":
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    from backend.app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_inspection_metadata_columns()


def _ensure_inspection_metadata_columns() -> None:
    inspector = inspect(engine)
    if "inspections" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("inspections")}
    required_columns = {
        "content_type": "VARCHAR(120)",
        "file_size_bytes": "INTEGER",
        "image_width": "INTEGER",
        "image_height": "INTEGER",
        "image_mode": "VARCHAR(40)",
    }
    missing_columns = {
        column_name: column_type
        for column_name, column_type in required_columns.items()
        if column_name not in existing_columns
    }
    if not missing_columns:
        return

    with engine.begin() as connection:
        for column_name, column_type in missing_columns.items():
            connection.execute(text(f"ALTER TABLE inspections ADD COLUMN {column_name} {column_type}"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
