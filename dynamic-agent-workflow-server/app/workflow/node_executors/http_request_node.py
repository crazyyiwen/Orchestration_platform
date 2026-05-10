"""HTTP Request node — async httpx call with config-driven URL/headers/body (spec §7)."""
from __future__ import annotations

from typing import Any

import httpx

from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.variables import VariableResolver


@register("http_request")
class HttpRequestNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        if not ctx.settings.ALLOW_EXTERNAL_HTTP:
            return _failed(node, "external HTTP disabled by ALLOW_EXTERNAL_HTTP=false")
        cfg = node.config or {}
        resolver = VariableResolver(state.get("variables", {}))

        url = resolver.resolve_string(cfg.get("url") or "")
        if not url:
            return _failed(node, "http_request missing url")
        method = (cfg.get("method") or "GET").upper()
        headers = resolver.resolve_value(cfg.get("headers") or {}) or {}
        params = resolver.resolve_value(cfg.get("query") or cfg.get("params") or {}) or {}
        body = resolver.resolve_value(cfg.get("body") or cfg.get("json") or None)
        timeout = float(cfg.get("timeout_seconds") or ctx.settings.NODE_TIMEOUT_SECONDS)

        try:
            resp = await ctx.http_client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=body if body is not None and method != "GET" else None,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            return _failed(node, f"HTTP error: {e}")

        try:
            json_body = resp.json()
        except ValueError:
            json_body = None

        result = {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "text": resp.text,
            "json": json_body,
        }
        next_h = "out" if resp.status_code < 400 else "error"
        return NodeExecutionResult(
            status="success" if next_h == "out" else "failed",
            output=result,
            next_handle=next_h,
            events=[
                {
                    "type": "http_request_completed",
                    "payload": {"status": resp.status_code, "url": url, "method": method},
                }
            ],
        )


def _failed(node: Node, message: str) -> NodeExecutionResult:
    return NodeExecutionResult(
        status="failed",
        output={"error": message},
        next_handle="error",
        error={"message": message, "node_id": node.id, "node_type": node.type},
    )
