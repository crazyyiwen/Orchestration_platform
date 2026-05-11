"""In-process event bus — async fan-out from run_manager → SSE subscribers.

Each subscriber gets its own :class:`asyncio.Queue`. Events are *also*
persisted to MongoDB before publish so reconnecting clients can replay via
``GET /api/runs/{id}/events?since=N``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)

_SENTINEL: object = object()


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, run_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subs.get(run_id, ()))
        for q in queues:
            # ``put_nowait`` because subscribers should drain promptly; if a
            # queue is full we drop (slow consumer protection).
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event bus dropped event for run_id=%s (queue full)", run_id)

    async def subscribe(self, run_id: str, *, maxsize: int = 1024) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subs.setdefault(run_id, set()).add(q)
        return q

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subs.get(run_id)
            if subs and queue in subs:
                subs.discard(queue)
                if not subs:
                    self._subs.pop(run_id, None)

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Generator that yields events for ``run_id`` until ``close(run_id)``."""
        q = await self.subscribe(run_id)
        try:
            while True:
                event = await q.get()
                if event is _SENTINEL:
                    return
                yield event
        finally:
            await self.unsubscribe(run_id, q)

    async def close(self, run_id: str) -> None:
        """Signal all subscribers for ``run_id`` to terminate cleanly."""
        async with self._lock:
            queues = list(self._subs.get(run_id, ()))
        for q in queues:
            try:
                q.put_nowait(_SENTINEL)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass
