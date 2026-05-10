"""Workflow checkpoint persistence.

This is the *domain-level* checkpoint repo (per spec §12). The LangGraph
``BaseCheckpointSaver`` adapter that backs the engine itself is layered on
top of this in Phase 8 (langgraph_runtime/checkpointing.py) — keeping the
repo plain Mongo CRUD here means tests don't need LangGraph imports.

Documents:
    checkpoint_id (ObjectId), run_id, workflow_id,
    node_id?, node_name?, state, created_at
"""
from __future__ import annotations

from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.db import collections as col
from app.repositories.base import utc_now


class CheckpointRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self._c = db[col.WORKFLOW_CHECKPOINTS]

    async def save(
        self,
        *,
        run_id: str,
        workflow_id: str,
        state: dict[str, Any],
        node_id: str | None = None,
        node_name: str | None = None,
    ) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "node_id": node_id,
            "node_name": node_name,
            "state": state,
            "created_at": utc_now(),
        }
        result = await self._c.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    async def latest_for_run(self, run_id: str) -> dict[str, Any] | None:
        return await self._c.find_one({"run_id": run_id}, sort=[("created_at", -1)])

    async def list_for_run(self, run_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self._c.find({"run_id": run_id}).sort("created_at", 1).limit(limit)
        return [doc async for doc in cursor]

    async def delete_for_run(self, run_id: str) -> int:
        result = await self._c.delete_many({"run_id": run_id})
        return result.deleted_count
