"""Run document persistence (workflow_runs collection).

Documents follow spec §12 plus a few runtime additions:
    run_id, workflow_id, workflow_version, status, input, state,
    final_output, error, started_at, completed_at, created_at, updated_at,
    created_by, parent_run_id, langfuse_trace_id, event_seq

`event_seq` is a monotonic per-run counter allocated atomically via $inc;
event_repository.append() relies on it for ordered, gap-free SSE replay.
"""
from __future__ import annotations

from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.db import collections as col
from app.repositories.base import utc_now


class RunRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self._c = db[col.WORKFLOW_RUNS]

    async def create(
        self,
        *,
        run_id: str,
        workflow_id: str,
        workflow_version: int,
        input: dict[str, Any] | None = None,
        created_by: str | None = None,
        parent_run_id: str | None = None,
        session_id: str | None = None,
        langfuse_trace_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        doc: dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "workflow_version": workflow_version,
            "status": "pending",
            "input": input or {},
            "state": initial_state or {},
            "final_output": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "created_at": now,
            "updated_at": now,
            "created_by": created_by,
            "parent_run_id": parent_run_id,
            "session_id": session_id,
            "langfuse_trace_id": langfuse_trace_id,
            "event_seq": 0,
        }
        await self._c.insert_one(doc)
        return doc

    async def latest_for_session(
        self, workflow_id: str, session_id: str, *, exclude_run_id: str | None = None
    ) -> dict[str, Any] | None:
        """Most recent run for a (workflow_id, session_id), newest first.

        Used to carry conversational ``flow``/``thread`` state from one turn
        into the next. Prefers a completed run but falls back to the latest
        of any status so a paused/failed prior turn still hands over its state.
        """
        query: dict[str, Any] = {
            "workflow_id": workflow_id,
            "session_id": session_id,
        }
        if exclude_run_id:
            query["run_id"] = {"$ne": exclude_run_id}
        # Prefer the latest completed turn.
        doc = await self._c.find_one(
            {**query, "status": "completed"}, sort=[("created_at", -1)]
        )
        if doc is not None:
            return doc
        return await self._c.find_one(query, sort=[("created_at", -1)])

    async def get(self, run_id: str) -> dict[str, Any] | None:
        return await self._c.find_one({"run_id": run_id})

    async def update_state(
        self,
        run_id: str,
        *,
        state: dict[str, Any] | None = None,
        status: str | None = None,
        final_output: Any = ...,
        error: dict[str, Any] | None = ...,
        started_at_now: bool = False,
        completed_at_now: bool = False,
    ) -> bool:
        """Patch the run document. Returns True if a doc was matched."""
        update: dict[str, Any] = {"updated_at": utc_now()}
        if state is not None:
            update["state"] = state
        if status is not None:
            update["status"] = status
        if final_output is not ...:
            update["final_output"] = final_output
        if error is not ...:
            update["error"] = error
        if started_at_now:
            update["started_at"] = utc_now()
        if completed_at_now:
            update["completed_at"] = utc_now()
        result = await self._c.update_one({"run_id": run_id}, {"$set": update})
        return result.matched_count == 1

    async def transition_status(
        self,
        run_id: str,
        *,
        from_status: str,
        to_status: str,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Optimistic state transition. Returns True iff status was updated.

        Used to guard pause→running (resume race), running→cancelled, etc.
        """
        update: dict[str, Any] = {"status": to_status, "updated_at": utc_now()}
        if extra:
            update.update(extra)
        result = await self._c.update_one(
            {"run_id": run_id, "status": from_status},
            {"$set": update},
        )
        return result.matched_count == 1

    async def allocate_event_sequence(self, run_id: str) -> int:
        """Atomically increment and return the next event sequence for a run.

        Returns the **post-increment** value (i.e., 1 for the first event).
        """
        doc = await self._c.find_one_and_update(
            {"run_id": run_id},
            {"$inc": {"event_seq": 1}},
            projection={"event_seq": 1, "_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            from app.core.errors import RunNotFoundError

            raise RunNotFoundError(f"run not found: {run_id}", details={"run_id": run_id})
        return int(doc["event_seq"])

    async def list_runs(
        self,
        *,
        workflow_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if workflow_id is not None:
            query["workflow_id"] = workflow_id
        if status is not None:
            query["status"] = status
        cursor = self._c.find(query).sort("created_at", -1).skip(skip).limit(limit)
        return [doc async for doc in cursor]
