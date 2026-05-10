"""Start node — initializes runtime variables and exits via ``out``."""
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


@register("start")
class StartNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        # Some workflows declare ``variableUpdates`` on the start node to seed
        # ``flow.*`` / ``thread.*`` / ``system.*`` from the run's input. We
        # honor the spec §7 contract (just exit "out") and apply only those
        # updates that don't reference unresolvable inputs.
        updates: dict[str, Any] = {}
        runtime_meta = {
            "workflowMetaData": {
                "workflowId": ctx.workflow_id,
                "agentName": node.name,
                "runId": ctx.run_id,
            }
        }
        updates.setdefault("variables", {}).update({"runtime": runtime_meta})

        return NodeExecutionResult(
            status="success",
            output={"started": True},
            next_handle="out",
            state_updates=updates,
        )
