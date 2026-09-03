from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes_dataset import router as dataset_router
from backend.app.api.routes_feedback import router as feedback_router
from backend.app.api.routes_fabric import router as fabric_router
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_inspection import router as inspection_router
from backend.app.api.routes_model import router as model_router
from backend.app.api.routes_reference import router as reference_router
from backend.app.core.config import get_settings
from backend.app.core.paths import ensure_runtime_dirs, resolve_project_path
from backend.app.db.database import init_db


def create_app() -> FastAPI:
    settings = get_settings()
    ensure_runtime_dirs(settings)
    init_db()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Local-first fabric defect detection MVP backend.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Only user-facing image artifacts are public. Model, index, metadata, and
    # database files remain outside the static surface.
    app.mount(
        "/storage/uploads",
        StaticFiles(directory=resolve_project_path(settings.uploads_dir)),
        name="storage-uploads",
    )
    app.mount(
        "/storage/annotated",
        StaticFiles(directory=resolve_project_path(settings.annotated_dir)),
        name="storage-annotated",
    )

    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(dataset_router, prefix="/api", tags=["datasets"])
    app.include_router(fabric_router, prefix="/api", tags=["fabrics"])
    app.include_router(inspection_router, prefix="/api", tags=["inspections"])
    app.include_router(reference_router, prefix="/api", tags=["reference"])
    app.include_router(feedback_router, prefix="/api", tags=["feedback"])
    app.include_router(model_router, prefix="/api", tags=["model"])

    return app


app = create_app()
