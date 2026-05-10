"""Human Input node — pauses execution and resumes via REST endpoint (spec §7).

Pause/resume protocol:

  * On first execution: returns ``status="paused"`` with a ``pause_payload``
    describing what's needed. The compiler's wrapper persists the payload and
    calls LangGraph's ``interrupt(...)`` so the run halts cleanly.
  * On resume: the same node runs again with a ``_resume_input`` value in
    state (set by the run manager). We persist it to ``save_to`` and return
    ``next_handle="out"``.

Config shape::

    {
      "question": "What is your name?",
      "input_type": "text",     # text | choice | file ...
      "save_to": "system.humanInput",   # path within state.variables
      "choices": ["A", "B"]      # optional
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
from app.workflow.state import set_path_inplace
from app.workflow.variables import VariableResolver


@register("human_input")
class HumanInputNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        resume_input = state.get("_resume_input")
        save_to = node.config.get("save_to") or "system.humanInput"

        if resume_input is None:
            # Pause path.
            resolver = VariableResolver(state.get("variables", {}))
            payload = {
                "type": "human_input_required",
                "node_id": node.id,
                "node_name": node.name,
                "question": resolver.resolve_string(node.config.get("question") or ""),
                "input_type": node.config.get("input_type", "text"),
                "choices": node.config.get("choices"),
                "save_to": save_to,
            }
            return NodeExecutionResult(
                status="paused",
                output=None,
                next_handle=None,
                pause_payload=payload,
                events=[{"type": "human_input_required", "payload": payload}],
            )

        # Resume path: persist input, clear the resume slot, exit "out".
        new_vars = _deepcopy(state.get("variables", {}))
        set_path_inplace(new_vars, save_to, resume_input)
        return NodeExecutionResult(
            status="success",
            output={"received": resume_input, "saved_to": save_to},
            next_handle="out",
            state_updates={"variables": new_vars, "_resume_input": None, "pause": None},
            events=[{"type": "human_input_resumed", "payload": {"saved_to": save_to}}],
        )


def _deepcopy(v: Any) -> Any:
    import copy

    return copy.deepcopy(v)
