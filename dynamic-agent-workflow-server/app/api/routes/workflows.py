"""Workflow compile / validate endpoints (spec §13)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.runtime.run_manager import RunManager
from app.workflow.loader import WorkflowLoader
from app.workflow.validation import validate

router = APIRouter(tags=["workflows"])


def _manager(request: Request) -> RunManager:
    rm = getattr(request.app.state, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="run manager not initialized")
    return rm


def _loader(request: Request) -> WorkflowLoader:
    ld = getattr(request.app.state, "workflow_loader", None)
    if ld is None:
        raise HTTPException(status_code=503, detail="workflow loader not initialized")
    return ld


class InlinePayload(BaseModel):
    workflow_id: str | None = None
    version: int | None = None
    payload: dict[str, Any]
    allow_cycles: bool = True


@router.post("/api/workflows/{workflow_id}/compile")
async def compile_by_id(workflow_id: str, request: Request) -> dict[str, Any]:
    loader = _loader(request)
    wf = await loader.load_by_id(workflow_id)
    report = validate(wf, allow_cycles=True)
    return {"summary": wf.summary().model_dump(), "validation": report.model_dump()}


@router.post("/api/workflows/compile-inline")
async def compile_inline(body: InlinePayload, request: Request) -> dict[str, Any]:
    wf = WorkflowLoader.load_inline(
        body.payload, workflow_id=body.workflow_id, version=body.version
    )
    report = validate(wf, allow_cycles=body.allow_cycles)
    return {"summary": wf.summary().model_dump(), "validation": report.model_dump()}


@router.post("/api/workflows/{workflow_id}/validate-runtime")
async def validate_runtime(workflow_id: str, request: Request) -> dict[str, Any]:
    """Like compile, but with strict node-type checking against the registry."""
    from app.workflow.node_executors.base import registered_types

    loader = _loader(request)
    wf = await loader.load_by_id(workflow_id)
    report = validate(wf, allow_cycles=True, known_types=registered_types())
    return {"summary": wf.summary().model_dump(), "validation": report.model_dump()}


@router.post("/api/workflows/register-inline")
async def register_inline(body: InlinePayload, request: Request) -> dict[str, Any]:
    """Register a workflow definition in the in-process inline cache.

    Once registered, ``sub_flow`` nodes (in any other workflow) can launch
    this workflow by its ``workflow_id`` without needing it in Mongo or the
    metadata API. Idempotent: re-registering replaces the cached entry.
    """
    rm = _manager(request)
    wf = WorkflowLoader.load_inline(
        body.payload, workflow_id=body.workflow_id, version=body.version
    )
    report = validate(wf, allow_cycles=body.allow_cycles)
    if not report.is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "workflow validation failed",
                "errors": [i.model_dump() for i in report.errors],
            },
        )
    rm.register_inline_workflow(wf)
    return {
        "summary": wf.summary().model_dump(),
        "validation": report.model_dump(),
        "registered": True,
    }
