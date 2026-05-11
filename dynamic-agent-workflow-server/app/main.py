from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Importing executors triggers @register side-effects.
import app.workflow.node_executors  # noqa: F401

from app.api.routes import health, models, observability, runs, tools, workflows
from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError, WorkflowServerError
from app.core.logging import configure_logging
from app.core.security import sanitize_error
from app.db.indexes import ensure_indexes
from app.db.mongodb import MongoDB
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.service import LLMService
from app.observability.langfuse_client import LangfuseClient
from app.repositories.event_repository import EventRepository
from app.repositories.run_repository import RunRepository
from app.runtime.event_bus import EventBus
from app.runtime.run_manager import RunManager
from app.tools.registry import ToolRegistry
from app.workflow.loader import WorkflowLoader

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

        # --- Mongo ------------------------------------------------------
        mongo = MongoDB(settings.MONGODB_URI, settings.MONGODB_DATABASE)
        await mongo.connect()
        if mongo.available:
            try:
                await ensure_indexes(mongo.db)
            except Exception as e:  # noqa: BLE001
                log.warning("index provisioning failed: %s", e)
        app.state.mongo = mongo

        run_repo = RunRepository(mongo.db) if mongo.available else None
        event_repo = EventRepository(mongo.db) if mongo.available else None
        app.state.run_repo = run_repo
        app.state.event_repo = event_repo

        # --- Shared HTTP client ----------------------------------------
        http_client = httpx.AsyncClient(timeout=60.0)
        app.state.http_client = http_client

        # --- LLM service -----------------------------------------------
        registry_path = model_registry_path or DEFAULT_MODEL_REGISTRY_PATH
        try:
            model_registry = ModelRegistry.from_yaml(registry_path)
        except ConfigurationError as e:
            log.warning("model registry could not be loaded: %s", e)
            model_registry = ModelRegistry([])
        provider_registry = ProviderRegistry.from_settings(settings, http_client=http_client)
        llm_service = LLMService(models=model_registry, providers=provider_registry)
        app.state.llm = llm_service

        # --- Tool registry ---------------------------------------------
        from app.tools.mock_tool import EchoTool, StaticAnswerTool

        tool_registry = ToolRegistry([EchoTool(), StaticAnswerTool()])
        app.state.tool_registry = tool_registry

        # --- Workflow loader -------------------------------------------
        workflow_loader = WorkflowLoader(settings, mongo=mongo, http_client=http_client)
        app.state.workflow_loader = workflow_loader

        # --- Langfuse --------------------------------------------------
        app.state.langfuse = LangfuseClient(settings)

        # --- Run manager (the central orchestrator) -------------------
        app.state.event_bus = EventBus()
        app.state.run_manager = RunManager(
            settings=settings,
            loader=workflow_loader,
            run_repo=run_repo,
            event_repo=event_repo,
            llm_service=llm_service,
            tool_registry=tool_registry,
            http_client=http_client,
            event_bus=app.state.event_bus,
        )

        log.info(
            "ready: %d models, %d tools, mongo=%s, langfuse=%s",
            len(model_registry.list()),
            len(tool_registry.list()),
            "ok" if mongo.available else "off",
            "on" if app.state.langfuse.enabled else "off",
        )

        try:
            yield
        finally:
            try:
                app.state.langfuse.flush()
            except Exception:  # noqa: BLE001
                pass
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

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(workflows.router)
    app.include_router(runs.router)
    app.include_router(tools.router)
    app.include_router(observability.router)

    return app


def create_app(*, model_registry_path: Path | None = None) -> FastAPI:
    return _build_app(get_settings(), model_registry_path=model_registry_path)


app = create_app()
