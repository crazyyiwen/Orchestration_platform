"""Agent node — LLM + tool-call loop, optional handoff (spec §7).

Loop bounded by ``ctx.settings.MAX_AGENT_ITERATIONS``.

Config shape::

    {
      "model_id": "gpt-4o",
      "system_prompt": "You are a helpful agent.",
      "user_template": "{{system.userQuery}}",   # what to send as the first user msg
      "tools": ["search", "calculator"],          # tool names, looked up in ToolRegistry
      "handoff_handles": ["specialist_a", "specialist_b"],   # optional
      "max_iterations": 6                         # overrides global limit downward
    }

If the LLM emits a tool call whose name matches one of the configured
``handoff_handles``, the executor returns ``next_handle=<that name>`` instead
of executing it as a tool — that's the handoff pattern.
"""
from __future__ import annotations

from typing import Any

from app.core.errors import ConfigurationError, WorkflowServerError
from app.llm.types import LLMRequest, Message, ToolSpec
from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.variables import VariableResolver


@register("agent")
class AgentNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        cfg = node.config or {}
        model_id = cfg.get("model_id") or ctx.settings.DEFAULT_MODEL_ID
        if not model_id:
            return _failed(node, "model_id is missing")

        resolver = VariableResolver(state.get("variables", {}))
        user_msg = resolver.resolve_string(
            cfg.get("user_template") or cfg.get("input") or "{{system.userQuery}}"
        )
        system_prompt = resolver.resolve_string(cfg.get("system_prompt") or "")

        max_iter = min(
            int(cfg.get("max_iterations") or ctx.settings.MAX_AGENT_ITERATIONS),
            ctx.settings.MAX_AGENT_ITERATIONS,
        )
        tool_names: list[str] = list(cfg.get("tools") or [])
        handoff_handles: set[str] = set(cfg.get("handoff_handles") or [])
        tool_specs: list[ToolSpec] = ctx.tool_registry.specs_for(tool_names)

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        if user_msg is not None:
            messages.append(Message(role="user", content=user_msg if isinstance(user_msg, str) else str(user_msg)))

        intermediate_steps: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for step in range(max_iter):
            try:
                resp = await ctx.llm_service.invoke(
                    model_id,
                    LLMRequest(messages=messages, tools=tool_specs or None),
                )
            except (ConfigurationError, WorkflowServerError) as e:
                return _failed(node, str(e), error_handle="error")

            events.append(
                {
                    "type": "agent_iteration",
                    "payload": {
                        "step": step,
                        "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
                        "finish_reason": resp.finish_reason,
                    },
                }
            )

            # Check for handoff first (before executing as a tool).
            if resp.tool_calls and handoff_handles:
                first_tc = resp.tool_calls[0]
                if first_tc.name in handoff_handles:
                    intermediate_steps.append({"handoff": first_tc.name, "args": first_tc.arguments})
                    return NodeExecutionResult(
                        status="success",
                        output={
                            "answer": None,
                            "handoff": first_tc.name,
                            "intermediate_steps": intermediate_steps,
                            "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
                        },
                        next_handle=first_tc.name,
                        events=events,
                    )

            if not resp.tool_calls:
                # Final answer reached.
                return NodeExecutionResult(
                    status="success",
                    output={
                        "answer": resp.content,
                        "content": resp.content,
                        "intermediate_steps": intermediate_steps,
                        "usage": resp.usage.model_dump(),
                    },
                    next_handle="out",
                    events=events,
                )

            # Append assistant turn (carrying tool_calls) and execute each tool.
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )
            for tc in resp.tool_calls:
                if not ctx.tool_registry.has(tc.name):
                    tool_result = {"error": f"unknown tool {tc.name!r}"}
                else:
                    tool = ctx.tool_registry.get(tc.name)
                    try:
                        tool_result = await tool.execute(tc.arguments)
                    except Exception as e:  # noqa: BLE001 — tool errors are logical, not fatal
                        tool_result = {"error": str(e)}
                intermediate_steps.append(
                    {"tool": tc.name, "args": tc.arguments, "result": tool_result}
                )
                events.append(
                    {"type": "tool_completed", "payload": {"tool": tc.name, "result": tool_result}}
                )
                import json as _json

                messages.append(
                    Message(
                        role="tool",
                        content=_json.dumps(tool_result, ensure_ascii=False),
                        tool_call_id=tc.id,
                    )
                )

        # Loop limit reached without a final answer.
        return NodeExecutionResult(
            status="failed",
            output={
                "error": "MAX_AGENT_ITERATIONS exceeded",
                "intermediate_steps": intermediate_steps,
            },
            next_handle="error",
            error={"message": "agent did not converge within max_iterations"},
            events=events,
        )


def _failed(node: Node, message: str, *, error_handle: str = "error") -> NodeExecutionResult:
    return NodeExecutionResult(
        status="failed",
        output={"error": message},
        next_handle=error_handle,
        error={"message": message, "node_id": node.id, "node_type": node.type},
    )
