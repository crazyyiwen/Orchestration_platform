"""External Agent node — POSTs an input payload to a configured endpoint."""
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


@register("external_agent")
class ExternalAgentNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        if not ctx.settings.ALLOW_EXTERNAL_HTTP:
            return _failed(node, "external HTTP disabled")
        cfg = node.config or {}
        resolver = VariableResolver(state.get("variables", {}))
        url = resolver.resolve_string(cfg.get("url") or "")
        if not url:
            return _failed(node, "external_agent requires config.url")
        headers = resolver.resolve_value(cfg.get("headers") or {}) or {}
        body = resolver.resolve_value(cfg.get("input") or cfg.get("body") or {})
        timeout = float(cfg.get("timeout_seconds") or ctx.settings.NODE_TIMEOUT_SECONDS)
        try:
            resp = await ctx.http_client.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            return _failed(node, f"HTTP error: {e}")
        if resp.status_code >= 400:
            return _failed(node, f"external agent returned {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            data = {"text": resp.text}
        return NodeExecutionResult(
            status="success",
            output=data,
            next_handle="out",
            events=[
                {
                    "type": "external_agent_completed",
                    "payload": {"url": url, "status": resp.status_code},
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
