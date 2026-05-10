"""Output node — terminal. Resolves output mappings into ``final_output``."""
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


@register("output")
class OutputNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        resolver = VariableResolver(state.get("variables", {}))
        mappings = node.config.get("outputMappings") or node.config.get("output") or {}
        if isinstance(mappings, dict):
            final = resolver.resolve_value(mappings)
        elif isinstance(mappings, str):
            final = resolver.resolve_string(mappings)
        else:
            # No mappings defined — fall back to the most recent node result.
            final = state.get("variables", {}).get("nodes", {})

        return NodeExecutionResult(
            status="success",
            output=final,
            next_handle=None,  # terminal
            state_updates={"final_output": final, "status": "completed"},
        )
