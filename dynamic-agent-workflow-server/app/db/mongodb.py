"""PyMongo Async client wrapper.

The connection is lazy-pinged at startup (see app.main lifespan). A failure to
reach Mongo is logged but does **not** prevent the app from starting — `/health`
reports `mongo: "unavailable"` until it comes back. Repository methods will
raise ``ConfigurationError`` if called while the database is down so callers
get a clear, sanitizable error instead of a leaky driver exception.
"""
from __future__ import annotations

import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.core.errors import ConfigurationError

log = logging.getLogger(__name__)


class MongoDB:
    def __init__(self, uri: str, database: str) -> None:
        self._uri = uri
        self._database_name = database
        self._client: AsyncMongoClient | None = None
        self._db: AsyncDatabase | None = None
        self._available: bool = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def database_name(self) -> str:
        return self._database_name

    @property
    def db(self) -> AsyncDatabase:
        if not self._available or self._db is None:
            raise ConfigurationError("MongoDB is not available")
        return self._db

    async def connect(self) -> None:
        """Attempt to connect and ping. Failure is logged but does not raise."""
        try:
            self._client = AsyncMongoClient(self._uri, serverSelectionTimeoutMS=2000)
            self._db = self._client[self._database_name]
            await self._client.admin.command("ping")
            self._available = True
            log.info("mongo connected database=%s", self._database_name)
        except (PyMongoError, OSError) as e:
            self._available = False
            log.warning("mongo unreachable at %s: %s", _safe_uri(self._uri), e)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None
                self._db = None
                self._available = False


def _safe_uri(uri: str) -> str:
    """Strip credentials from a Mongo URI for safe logging."""
    if "@" in uri and "://" in uri:
        scheme, _, rest = uri.partition("://")
        _, _, host_part = rest.partition("@")
        return f"{scheme}://***@{host_part}"
    return uri
