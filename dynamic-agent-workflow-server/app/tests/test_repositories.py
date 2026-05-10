"""Integration tests for the four runtime repositories.

These exercise real Mongo CRUD; they auto-skip via the ``mongo_db`` fixture
when no MongoDB instance is reachable from the test environment.
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.errors import RunNotFoundError
from app.db import collections as col
from app.db.indexes import ensure_indexes
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.compile_cache_repository import CompileCacheRepository
from app.repositories.event_repository import EventRepository
from app.repositories.run_repository import RunRepository


pytestmark = pytest.mark.asyncio


async def test_indexes_are_created_idempotently(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    # Second call must not raise.
    await ensure_indexes(mongo_db)
    info = await mongo_db[col.WORKFLOW_RUNS].index_information()
    assert "run_id_unique" in info
    assert info["run_id_unique"].get("unique") is True


async def test_run_lifecycle_create_get_update_transition(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    repo = RunRepository(mongo_db)

    doc = await repo.create(
        run_id="run-1",
        workflow_id="wf-A",
        workflow_version=1,
        input={"q": "hi"},
    )
    assert doc["status"] == "pending"
    assert doc["event_seq"] == 0

    fetched = await repo.get("run-1")
    assert fetched is not None
    assert fetched["workflow_id"] == "wf-A"

    ok = await repo.update_state(
        "run-1",
        state={"variables": {"system": {"userQuery": "hi"}}},
        status="running",
        started_at_now=True,
    )
    assert ok is True
    after = await repo.get("run-1")
    assert after["status"] == "running"
    assert after["started_at"] is not None

    # Optimistic transition: only succeeds from the matching prior status.
    transitioned = await repo.transition_status(
        "run-1", from_status="running", to_status="paused"
    )
    assert transitioned is True
    refused = await repo.transition_status(
        "run-1", from_status="running", to_status="completed"
    )
    assert refused is False  # status is now "paused", not "running"


async def test_event_sequence_is_monotonic_under_concurrency(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    runs = RunRepository(mongo_db)
    events = EventRepository(mongo_db)

    await runs.create(run_id="run-2", workflow_id="wf-B", workflow_version=1)

    async def emit(i: int) -> int:
        seq = await runs.allocate_event_sequence("run-2")
        await events.append(
            run_id="run-2",
            workflow_id="wf-B",
            sequence=seq,
            type="node_completed",
            payload={"i": i},
        )
        return seq

    # Concurrent allocations must produce a strictly increasing, gap-free 1..N.
    sequences = await asyncio.gather(*(emit(i) for i in range(20)))
    assert sorted(sequences) == list(range(1, 21))

    listed = await events.list_for_run("run-2")
    assert [e["sequence"] for e in listed] == list(range(1, 21))

    after_5 = await events.list_for_run("run-2", since=5)
    assert [e["sequence"] for e in after_5] == list(range(6, 21))


async def test_allocate_sequence_for_missing_run_raises(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    runs = RunRepository(mongo_db)
    with pytest.raises(RunNotFoundError):
        await runs.allocate_event_sequence("does-not-exist")


async def test_checkpoint_save_and_latest(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    repo = CheckpointRepository(mongo_db)
    await repo.save(run_id="run-3", workflow_id="wf-C", state={"step": 1}, node_name="a")
    await repo.save(run_id="run-3", workflow_id="wf-C", state={"step": 2}, node_name="b")
    latest = await repo.latest_for_run("run-3")
    assert latest is not None
    assert latest["state"]["step"] == 2
    listed = await repo.list_for_run("run-3")
    assert len(listed) == 2


async def test_compile_cache_upsert_and_invalidate(mongo_db) -> None:
    await ensure_indexes(mongo_db)
    repo = CompileCacheRepository(mongo_db)
    first = await repo.upsert(
        workflow_id="wf-D",
        workflow_version=1,
        definition_hash="abc123",
        compiled_metadata={"nodes": 3},
    )
    assert first["definition_hash"] == "abc123"
    second = await repo.upsert(
        workflow_id="wf-D",
        workflow_version=1,
        definition_hash="def456",
        compiled_metadata={"nodes": 4},
    )
    assert second["definition_hash"] == "def456"
    # created_at remains stable across upserts; updated_at changes.
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]

    deleted = await repo.invalidate("wf-D")
    assert deleted == 1
    assert await repo.get("wf-D", 1) is None
