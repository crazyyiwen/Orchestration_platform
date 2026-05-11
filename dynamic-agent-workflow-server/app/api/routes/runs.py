"""Run lifecycle endpoints + SSE event stream (spec §13/§14)."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.core.errors import RunStateConflictError
from app.repositories.event_repository import EventRepository
from app.repositories.run_repository import RunRepository
from app.runtime.run_manager import RunManager
from app.workflow.loader import WorkflowLoader

router = APIRouter(tags=["runs"])


def _manager(request: Request) -> RunManager:
    rm = getattr(request.app.state, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="run manager not initialized")
    return rm


# ----- request bodies -----------------------------------------------------


class StartRunBody(BaseModel):
    input: dict[str, Any] | None = None
    version: int | None = None


class InlineRunBody(BaseModel):
    workflow_id: str | None = None
    version: int | None = None
    payload: dict[str, Any]
    input: dict[str, Any] | None = None
    wait: bool = False


class HumanInputBody(BaseModel):
    input: Any


class ApprovalBody(BaseModel):
    decision: str  # "approved" | "rejected"
    reason: str | None = None


# ----- routes --------------------------------------------------------------


@router.post("/api/workflows/{workflow_id}/runs")
async def create_and_start(workflow_id: str, body: StartRunBody, request: Request) -> dict[str, Any]:
    rm = _manager(request)
    run = await rm.create_run(
        workflow_id=workflow_id, input=body.input, version=body.version
    )
    await rm.start_run(
        run["run_id"],
        definition=run["definition"],
        initial_state=run["state"],
        wait=False,
    )
    return {"run_id": run["run_id"], "status": "running"}


@router.post("/api/workflows/run-inline")
async def run_inline(body: InlineRunBody, request: Request) -> dict[str, Any]:
    rm = _manager(request)
    wf = WorkflowLoader.load_inline(
        body.payload, workflow_id=body.workflow_id, version=body.version
    )
    run = await rm.create_run(
        workflow_id=wf.workflow_id, input=body.input, inline_definition=wf
    )
    if body.wait:
        final = await rm.start_run(
            run["run_id"], definition=run["definition"], initial_state=run["state"], wait=True
        )
        return {
            "run_id": run["run_id"],
            "status": _resolve_status(final),
            "final_output": final.get("final_output") if isinstance(final, dict) else None,
            "pause": _extract_pause(final) if isinstance(final, dict) else None,
        }
    await rm.start_run(
        run["run_id"], definition=run["definition"], initial_state=run["state"], wait=False
    )
    return {"run_id": run["run_id"], "status": "running"}


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    rm = _manager(request)
    row = await rm.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return _serialize_run(row)


@router.get("/api/runs/{run_id}/state")
async def get_run_state(run_id: str, request: Request) -> dict[str, Any]:
    rm = _manager(request)
    row = await rm.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return {
        "run_id": run_id,
        "status": row["status"],
        "state": row.get("state") or {},
        "final_output": row.get("final_output"),
        "pause": (row.get("state") or {}).get("pause"),
    }


@router.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
    rm = _manager(request)
    await rm.cancel_run(run_id)
    return {"run_id": run_id, "status": "cancelled"}


@router.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, body: dict[str, Any], request: Request) -> dict[str, Any]:
    rm = _manager(request)
    try:
        return await rm.resume_run(run_id, body.get("input"), wait=True)
    except RunStateConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/api/runs/{run_id}/human-input")
async def submit_human_input(
    run_id: str, body: HumanInputBody, request: Request
) -> dict[str, Any]:
    rm = _manager(request)
    try:
        return await rm.resume_run(run_id, body.input, wait=True)
    except RunStateConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/api/runs/{run_id}/approval")
async def submit_approval(
    run_id: str, body: ApprovalBody, request: Request
) -> dict[str, Any]:
    rm = _manager(request)
    try:
        return await rm.resume_run(
            run_id, {"decision": body.decision, "reason": body.reason}, wait=True
        )
    except RunStateConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/api/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    since: int | None = Query(default=None, description="resume from this sequence"),
) -> EventSourceResponse:
    """SSE stream of run events. Replays from Mongo (since=N) then live."""
    rm = _manager(request)
    event_repo: EventRepository | None = getattr(request.app.state, "event_repo", None)

    async def gen() -> AsyncIterator[dict[str, Any]]:
        replayed_max = since or 0
        if event_repo is not None:
            for ev in await event_repo.list_for_run(run_id, since=since):
                yield _sse_event(ev)
                replayed_max = max(replayed_max, int(ev.get("sequence") or 0))
        # Live tail.
        async for ev in rm.event_bus.stream(run_id):
            seq = int(ev.get("sequence") or 0)
            if seq and seq <= replayed_max:
                continue
            if await request.is_disconnected():
                break
            yield _sse_event(ev)

    return EventSourceResponse(gen())


# ----- helpers ------------------------------------------------------------


def _sse_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Format an event for sse-starlette (id/event/data fields)."""
    seq = ev.get("sequence")
    return {
        "id": str(seq) if seq is not None else None,
        "event": ev.get("type", "event"),
        "data": json.dumps(_json_safe(ev), ensure_ascii=False, default=str),
    }


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if hasattr(v, "isoformat"):  # datetime
        return v.isoformat()
    return v


def _serialize_run(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("_id", None)
    return _json_safe(out)


def _resolve_status(final: Any) -> str:
    if not isinstance(final, dict):
        return "completed"
    if final.get("__interrupt__"):
        return "paused"
    return final.get("status", "completed")


def _extract_pause(final: dict[str, Any]) -> Any:
    interrupts = final.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", None)
