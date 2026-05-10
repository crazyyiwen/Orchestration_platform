"""Script node — DISABLED by default for security (spec §16).

Honors ``ENABLE_SCRIPT_NODE``. When enabled, this is *still* a mock-only
executor: it returns a canned dict from ``config.mock_output``. Real Python
execution requires a sandbox worker and is out of scope for v1.

TODO(future): off-process sandbox (gVisor / firecracker / WASM) — see plan.
"""
from __future__ import annotations

from typing import Any

from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)


@register("script")
class ScriptNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        if not ctx.settings.ENABLE_SCRIPT_NODE:
            return NodeExecutionResult(
                status="failed",
                output={"error": "script node disabled"},
                next_handle="error",
                error={
                    "message": "script node disabled by ENABLE_SCRIPT_NODE=false",
                    "node_id": node.id,
                },
            )
        # Mock-only behavior even when enabled.
        mock_output = node.config.get("mock_output", {"executed": True})
        return NodeExecutionResult(
            status="success",
            output=mock_output,
            next_handle="out",
            events=[{"type": "script_executed_mock", "payload": {"node_id": node.id}}],
        )
