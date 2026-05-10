"""Workflow run event log (workflow_run_events).

Append-only. Events are tagged with a per-run monotonic ``sequence`` allocated
by ``RunRepository.allocate_event_sequence`` so that SSE clients can resume
with ``Last-Event-ID`` and never miss an event during reconnect.

Documents (spec §12 + small extensions):
    event_id (ObjectId), run_id, workflow_id, sequence, type,
    node_id?, node_name?, node_type?, payload, created_at
"""
from __future__ import annotations

from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.db import collections as col
from app.repositories.base import utc_now


class EventRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self._c = db[col.WORKFLOW_RUN_EVENTS]

    async def append(
        self,
        *,
        run_id: str,
        workflow_id: str,
        sequence: int,
        type: str,
        payload: dict[str, Any] | None = None,
        node_id: str | None = None,
        node_name: str | None = None,
        node_type: str | None = None,
    ) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "sequence": sequence,
            "type": type,
            "node_id": node_id,
            "node_name": node_name,
            "node_type": node_type,
            "payload": payload or {},
            "created_at": utc_now(),
        }
        result = await self._c.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    async def list_for_run(
        self,
        run_id: str,
        *,
        since: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return events for a run ordered by sequence ascending.

        ``since`` is exclusive: pass the last sequence the client received and
        you'll get only events after it.
        """
        query: dict[str, Any] = {"run_id": run_id}
        if since is not None:
            query["sequence"] = {"$gt": since}
        cursor = self._c.find(query).sort("sequence", 1).limit(limit)
        return [doc async for doc in cursor]

    async def latest_sequence(self, run_id: str) -> int:
        doc = await self._c.find_one(
            {"run_id": run_id},
            sort=[("sequence", -1)],
            projection={"sequence": 1, "_id": 0},
        )
        return int(doc["sequence"]) if doc else 0
