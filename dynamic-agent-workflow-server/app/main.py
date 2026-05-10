from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, models
from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError, WorkflowServerError
from app.core.logging import configure_logging
from app.core.security import sanitize_error
from app.db.indexes import ensure_indexes
from app.db.mongodb import MongoDB
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.service import LLMService

# Default location for the model registry. Tests can swap via
# create_app(model_registry_path=...) or by setting MODEL_REGISTRY_PATH.
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _build_app(
    settings: Settings,
    *,
    model_registry_path: Path | None = None,
) -> FastAPI:
    configure_logging(settings.LOG_LEVEL)
    log = logging.getLogger("app.main")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("startup app=%s env=%s", settings.APP_NAME, settings.APP_ENV)

        # --- Mongo -------------------------------------------------------
        mongo = MongoDB(settings.MONGODB_URI, settings.MONGODB_DATABASE)
        await mongo.connect()
        if mongo.available:
            try:
                await ensure_indexes(mongo.db)
            except Exception as e:  # noqa: BLE001 - never crash startup on indexes
                log.warning("index provisioning failed: %s", e)
        app.state.mongo = mongo

        # --- Shared HTTP client (LLM providers, metadata API, HTTP node) -
        http_client = httpx.AsyncClient(timeout=60.0)
        app.state.http_client = http_client

        # --- LLM service -------------------------------------------------
        registry_path = model_registry_path or DEFAULT_MODEL_REGISTRY_PATH
        try:
            model_registry = ModelRegistry.from_yaml(registry_path)
        except ConfigurationError as e:
            log.warning("model registry could not be loaded: %s", e)
            model_registry = ModelRegistry([])
        provider_registry = ProviderRegistry.from_settings(settings, http_client=http_client)
        app.state.llm = LLMService(models=model_registry, providers=provider_registry)
        log.info(
            "llm service ready: %d models, providers=%s",
            len(model_registry.list()),
            provider_registry.names(),
        )

        try:
            yield
        finally:
            await http_client.aclose()
            await mongo.close()
            log.info("shutdown app=%s", settings.APP_NAME)

    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.FRONTEND_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(WorkflowServerError)
    async def _domain_error_handler(request: Request, exc: WorkflowServerError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=sanitize_error(exc.to_payload()))

    # Health is mounted at the root, not under API_PREFIX (so probes don't depend on prefix).
    app.include_router(health.router)
    app.include_router(models.router)

    return app


def create_app(*, model_registry_path: Path | None = None) -> FastAPI:
    return _build_app(get_settings(), model_registry_path=model_registry_path)


app = create_app()
