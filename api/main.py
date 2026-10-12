"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from fastapi import FastAPI

from .constants import SERVICE_TITLE, SERVICE_VERSION
from .routes import health_router, inquiry_router, learning_router


def create_app() -> FastAPI:
    """Build a configured FastAPI application instance."""
    application = FastAPI(
        title=SERVICE_TITLE,
        version=SERVICE_VERSION,
        description=(
            "HTTP adapter for tattoo inquiry extraction, style detection, "
            "artist routing, risk classification, and draft replies."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    application.include_router(health_router)
    application.include_router(inquiry_router)
    application.include_router(learning_router)
    return application


app = create_app()
