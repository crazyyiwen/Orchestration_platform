"""Compile metadata cache (workflow_compiled_cache).

We never persist the compiled LangGraph object itself (it's an in-memory
graph). This collection records *that* a definition was compiled, with its
content hash and lightweight metadata, so we can detect drift, observe
compile rates, and invalidate the in-process cache when a workflow changes.
"""
from __future__ import annotations

from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.db import collections as col
from app.repositories.base import utc_now


class CompileCacheRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self._c = db[col.WORKFLOW_COMPILED_CACHE]

    async def get(self, workflow_id: str, workflow_version: int) -> dict[str, Any] | None:
        return await self._c.find_one(
            {"workflow_id": workflow_id, "workflow_version": workflow_version}
        )

    async def upsert(
        self,
        *,
        workflow_id: str,
        workflow_version: int,
        definition_hash: str,
        compiled_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        result = await self._c.find_one_and_update(
            {"workflow_id": workflow_id, "workflow_version": workflow_version},
            {
                "$set": {
                    "definition_hash": definition_hash,
                    "compiled_metadata": compiled_metadata or {},
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "workflow_id": workflow_id,
                    "workflow_version": workflow_version,
                    "created_at": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return result  # type: ignore[return-value]

    async def invalidate(self, workflow_id: str, workflow_version: int | None = None) -> int:
        query: dict[str, Any] = {"workflow_id": workflow_id}
        if workflow_version is not None:
            query["workflow_version"] = workflow_version
        result = await self._c.delete_many(query)
        return result.deleted_count
