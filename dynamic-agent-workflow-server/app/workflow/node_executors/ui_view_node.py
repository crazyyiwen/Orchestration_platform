"""uiView node — a frontend "render HTML" step.

At runtime it's a no-op pass-through: it resolves the HTML template against the
current variables and returns the rendered string in its ``output``. The
frontend is what actually displays the HTML; the server just produces it so
later nodes (or the output node's mappings) can reference it.
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
from app.workflow.variables import VariableResolver


@register("uiView")
class UIViewNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        cfg = node.config or {}
        resolver = VariableResolver(state.get("variables", {}))
        html = resolver.resolve_string(cfg.get("html") or "")
        return NodeExecutionResult(
            status="success",
            output={"html": html, "sanitize": bool(cfg.get("sanitize", True))},
            next_handle="out",
        )
