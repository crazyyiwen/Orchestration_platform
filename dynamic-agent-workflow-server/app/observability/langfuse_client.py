"""Langfuse wrapper.

The wrapper exposes a stable surface and **no-ops cleanly** when Langfuse is
disabled or its keys are missing. Callers don't branch on enabled/disabled —
they just call methods. Per the plan, keys are never returned to clients;
``trace_url(run_id)`` constructs a public URL using only the host + run_id.
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings

log = logging.getLogger(__name__)


class LangfuseClient:
    def __init__(self, settings: Settings) -> None:
        self._enabled = (
            settings.LANGFUSE_ENABLED
            and bool(settings.LANGFUSE_PUBLIC_KEY)
            and bool(settings.LANGFUSE_SECRET_KEY)
        )
        self._host = settings.LANGFUSE_HOST.rstrip("/")
        self._client: Any | None = None
        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                log.info("langfuse client initialized host=%s", settings.LANGFUSE_HOST)
            except Exception as e:  # noqa: BLE001
                log.warning("langfuse init failed: %s — disabling tracing", e)
                self._enabled = False
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace_url(self, run_id: str, project_id: str | None = None) -> str | None:
        """Public URL for a run's trace. Returns None when disabled."""
        if not self._enabled:
            return None
        # Langfuse cloud / self-hosted both support /trace/<id> paths.
        return f"{self._host}/trace/{run_id}"

    def event(
        self,
        run_id: str,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort event ingestion. Failures are logged, never raised."""
        if not self._enabled or self._client is None:
            return
        try:
            # The langfuse SDK has changed shape across major versions; we use
            # the most stable surface (``trace`` + ``event``).
            client = self._client
            trace = client.trace(id=run_id, name=run_id)
            trace.event(name=name, input=input, output=output, metadata=metadata)
        except Exception:  # noqa: BLE001
            log.exception("langfuse event failed run_id=%s name=%s", run_id, name)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                log.exception("langfuse flush failed")
