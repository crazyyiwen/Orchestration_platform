"""Sub Flow node — invokes another workflow synchronously (spec §7).

Calls the sub_flow_launcher injected by Phase 9's run_manager. Enforces
``MAX_SUBFLOW_DEPTH`` so recursive sub-flows can't run away.
"""
from __future__ import annotations

from typing import Any

from app.core.errors import ConfigurationError, ExecutionLimitExceeded
from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.variables import VariableResolver


@register("sub_flow")
class SubFlowNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        cfg = node.config or {}
        sub_id = cfg.get("workflow_id") or cfg.get("sub_workflow_id")
        if not sub_id:
            return _failed(node, "sub_flow node requires config.workflow_id")
        if ctx.depth + 1 > ctx.settings.MAX_SUBFLOW_DEPTH:
            raise ExecutionLimitExceeded(
                f"sub_flow depth {ctx.depth + 1} exceeds MAX_SUBFLOW_DEPTH={ctx.settings.MAX_SUBFLOW_DEPTH}"
            )
        if ctx.sub_flow_launcher is None:
            raise ConfigurationError("sub_flow_launcher not configured (run_manager missing)")

        resolver = VariableResolver(state.get("variables", {}))
        inputs = resolver.resolve_value(cfg.get("input") or cfg.get("inputs") or {})

        sub_result = await ctx.sub_flow_launcher(sub_id, inputs, ctx.depth + 1, ctx.run_id)
        return NodeExecutionResult(
            status="success",
            output=sub_result,
            next_handle="out",
            events=[
                {
                    "type": "subflow_completed",
                    "payload": {
                        "sub_workflow_id": sub_id,
                        "sub_run_id": sub_result.get("run_id") if isinstance(sub_result, dict) else None,
                    },
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
