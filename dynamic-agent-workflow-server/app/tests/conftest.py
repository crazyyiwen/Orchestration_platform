"""Shared pytest fixtures.

Integration tests that need MongoDB use the ``mongo_db`` fixture, which:
  * connects to ``$MONGODB_URI`` (default localhost:27017),
  * pings to confirm reachability,
  * yields a per-test database with a unique name,
  * drops the database on teardown.

If Mongo is unreachable, the fixture raises ``pytest.skip(...)`` so the test is
reported as skipped rather than failing — keeping CI green on machines without
a Mongo daemon while still verifying behavior locally and in proper CI.
"""
from __future__ import annotations

import os
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")


@pytest_asyncio.fixture
async def mongo_db() -> AsyncIterator[AsyncDatabase]:
    client = AsyncMongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    try:
        await client.admin.command("ping")
    except (PyMongoError, OSError) as e:
        await client.close()
        pytest.skip(f"MongoDB unreachable at {MONGO_URI}: {e}")
    db_name = f"test_runtime_{uuid.uuid4().hex[:12]}"
    db = client[db_name]
    try:
        yield db
    finally:
        try:
            await client.drop_database(db_name)
        finally:
            await client.close()
