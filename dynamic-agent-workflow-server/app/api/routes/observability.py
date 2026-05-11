"""Observability route — Langfuse trace link (spec §13)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.observability.langfuse_client import LangfuseClient

router = APIRouter(tags=["observability"])


@router.get("/api/runs/{run_id}/trace-link")
async def trace_link(run_id: str, request: Request) -> dict[str, Any]:
    lf: LangfuseClient | None = getattr(request.app.state, "langfuse", None)
    if lf is None or not lf.enabled:
        raise HTTPException(
            status_code=404, detail="Langfuse is not enabled or not configured"
        )
    return {"run_id": run_id, "url": lf.trace_url(run_id)}
