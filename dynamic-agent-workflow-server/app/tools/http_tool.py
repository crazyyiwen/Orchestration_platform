"""HTTP-backed tool — invokes an external endpoint with the LLM's arguments.

Configurable via constructor. Honors the global ``ALLOW_EXTERNAL_HTTP`` gate.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import ConfigurationError, WorkflowServerError
from app.tools.base import BaseTool


class HttpTool(BaseTool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        url: str,
        method: str = "POST",
        parameters: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient,
        allow_external: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not allow_external:
            raise ConfigurationError(
                f"http tool {name!r} disabled by ALLOW_EXTERNAL_HTTP=false"
            )
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self._url = url
        self._method = method.upper()
        self._headers = headers or {}
        self._http = http_client
        self._timeout = timeout_seconds

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.request(
                self._method,
                self._url,
                json=args if self._method != "GET" else None,
                params=args if self._method == "GET" else None,
                headers=self._headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as e:
            raise WorkflowServerError(
                f"http tool {self.name!r} failed: {e}",
                details={"tool": self.name, "url": self._url},
            ) from e
        try:
            body = resp.json()
        except ValueError:
            body = {"text": resp.text}
        return {"status": resp.status_code, "body": body}
