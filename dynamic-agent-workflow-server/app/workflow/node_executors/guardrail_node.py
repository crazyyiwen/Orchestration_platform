"""Guardrail node — same condition DSL as Rule, routes ``allow`` / ``block``.

Config shape::

    {
      "input": "{{system.userQuery}}",
      "rules": [
        {"operator": "regex", "value": "credit card|ssn", "block_reason": "PII"}
      ]
    }
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
from app.workflow.node_executors.rule_node import _apply_operator
from app.workflow.variables import VariableResolver


@register("guardrail")
class GuardrailNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        cfg = node.config or {}
        resolver = VariableResolver(state.get("variables", {}))
        target = resolver.resolve_string(cfg.get("input") or "")
        rules = cfg.get("rules") or []

        for r in rules:
            op = r.get("operator") or "contains"
            val = resolver.resolve_value(r.get("value"))
            if _apply_operator(target, op, val):
                reason = r.get("block_reason") or r.get("reason") or "guardrail blocked"
                return NodeExecutionResult(
                    status="success",
                    output={"allowed": False, "reason": reason, "rule": r},
                    next_handle="block",
                    events=[
                        {
                            "type": "guardrail_blocked",
                            "payload": {"reason": reason, "rule": r},
                        }
                    ],
                )

        return NodeExecutionResult(
            status="success",
            output={"allowed": True},
            next_handle="allow",
            events=[{"type": "guardrail_allowed", "payload": {}}],
        )
