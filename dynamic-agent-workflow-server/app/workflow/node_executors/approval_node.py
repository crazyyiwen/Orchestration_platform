"""Approval node — pauses for human approve/reject decision (spec §7).

Resume input shape: ``{"decision": "approved" | "rejected", "reason": "..."}``
or simply the string ``"approved"`` / ``"rejected"``. The result's
``next_handle`` matches the decision so the dynamic router branches accordingly.
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
from app.workflow.state import set_path_inplace
from app.workflow.variables import VariableResolver


@register("approval")
class ApprovalNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        resume_input = state.get("_resume_input")
        save_to = node.config.get("save_to") or "system.approval"

        if resume_input is None:
            resolver = VariableResolver(state.get("variables", {}))
            payload = {
                "type": "approval_required",
                "node_id": node.id,
                "node_name": node.name,
                "summary": resolver.resolve_string(node.config.get("summary") or ""),
                "details": resolver.resolve_value(node.config.get("details") or {}),
                "save_to": save_to,
            }
            return NodeExecutionResult(
                status="paused",
                output=None,
                next_handle=None,
                pause_payload=payload,
                events=[{"type": "approval_required", "payload": payload}],
            )

        # Normalize the decision.
        decision: str
        if isinstance(resume_input, str):
            decision = resume_input.lower()
            full = {"decision": decision}
        elif isinstance(resume_input, dict):
            decision = str(resume_input.get("decision", "")).lower()
            full = resume_input
        else:
            decision = "rejected"
            full = {"decision": "rejected", "raw": resume_input}

        if decision not in {"approved", "rejected"}:
            decision = "rejected"
            full["decision"] = "rejected"

        new_vars = _deepcopy(state.get("variables", {}))
        set_path_inplace(new_vars, save_to, full)
        return NodeExecutionResult(
            status="success",
            output={"decision": decision, "saved_to": save_to},
            next_handle=decision,
            state_updates={"variables": new_vars, "_resume_input": None, "pause": None},
            events=[{"type": "approval_resolved", "payload": {"decision": decision}}],
        )


def _deepcopy(v: Any) -> Any:
    import copy

    return copy.deepcopy(v)
