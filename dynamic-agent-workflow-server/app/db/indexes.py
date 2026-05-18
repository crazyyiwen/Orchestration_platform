"""Idempotent index provisioning for the runtime collections (spec §12)."""
from __future__ import annotations

import logging

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.asynchronous.database import AsyncDatabase

from app.db import collections as col

log = logging.getLogger(__name__)


# Index specs are also exported as plain data so they can be unit-tested
# without a live database.
RUN_INDEXES: list[IndexModel] = [
    IndexModel([("run_id", ASCENDING)], name="run_id_unique", unique=True),
    IndexModel([("workflow_id", ASCENDING)], name="workflow_id"),
    IndexModel([("status", ASCENDING)], name="status"),
    IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
    IndexModel([("parent_run_id", ASCENDING)], name="parent_run_id_sparse", sparse=True),
    IndexModel(
        [("workflow_id", ASCENDING), ("session_id", ASCENDING), ("created_at", DESCENDING)],
        name="workflow_session_recent",
        sparse=True,
    ),
]

EVENT_INDEXES: list[IndexModel] = [
    IndexModel(
        [("run_id", ASCENDING), ("sequence", ASCENDING)],
        name="run_id_sequence_unique",
        unique=True,
    ),
    IndexModel([("workflow_id", ASCENDING)], name="workflow_id"),
    IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
]

CHECKPOINT_INDEXES: list[IndexModel] = [
    IndexModel([("run_id", ASCENDING)], name="run_id"),
    IndexModel(
        [("run_id", ASCENDING), ("created_at", ASCENDING)],
        name="run_id_created_at",
    ),
]

COMPILE_CACHE_INDEXES: list[IndexModel] = [
    IndexModel(
        [("workflow_id", ASCENDING), ("workflow_version", ASCENDING)],
        name="workflow_id_version_unique",
        unique=True,
    ),
    IndexModel([("definition_hash", ASCENDING)], name="definition_hash"),
]


async def ensure_indexes(db: AsyncDatabase) -> None:
    """Create indexes for all runtime collections. Safe to call repeatedly."""
    await db[col.WORKFLOW_RUNS].create_indexes(RUN_INDEXES)
    await db[col.WORKFLOW_RUN_EVENTS].create_indexes(EVENT_INDEXES)
    await db[col.WORKFLOW_CHECKPOINTS].create_indexes(CHECKPOINT_INDEXES)
    await db[col.WORKFLOW_COMPILED_CACHE].create_indexes(COMPILE_CACHE_INDEXES)
    log.info("mongo indexes ensured collections=%d", len(col.ALL))
