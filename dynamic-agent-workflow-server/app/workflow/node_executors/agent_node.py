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
        # Frontend uses ``config.model`` (display name); legacy uses ``model_id``.
        # We accept either, falling back to settings.DEFAULT_MODEL_ID. If the
        # chosen id isn't in the registry, fall back to DEFAULT_MODEL_ID rather
        # than aborting the agent — keeps dev iterating without registry edits.
        requested_model = cfg.get("model_id") or cfg.get("model")
        model_id = requested_model or ctx.settings.DEFAULT_MODEL_ID
        if not model_id:
            return _failed(node, "model_id is missing")
        if not ctx.llm_service.models.has(model_id):
            import logging as _lg

            _lg.getLogger(__name__).warning(
                "agent node %r: model %r not in registry; falling back to %r",
                node.id, model_id, ctx.settings.DEFAULT_MODEL_ID,
            )
            model_id = ctx.settings.DEFAULT_MODEL_ID

        resolver = VariableResolver(state.get("variables", {}))
        user_msg = resolver.resolve_string(
            cfg.get("user_template") or cfg.get("userQuery") or cfg.get("input") or "{{system.userQuery}}"
        )
        system_prompt = resolver.resolve_string(
            cfg.get("system_prompt") or cfg.get("instructions") or ""
        )

        max_iter = min(
            int(cfg.get("max_iterations") or ctx.settings.MAX_AGENT_ITERATIONS),
            ctx.settings.MAX_AGENT_ITERATIONS,
        )
        tool_names: list[str] = list(cfg.get("tools") or [])

        # Two handoff config shapes are supported:
        #   1) ``handoff_handles: ["id1", "id2"]``         — flat ids (legacy)
        #   2) ``handoffs: [{"id": "...", "name": "..."}]`` — frontend shape
        # We build a name→id map. If present, the agent uses **text-mode**
        # handoff: it asks the LLM for JSON of ``{next_handle: <name>}`` and
        # maps the chosen name back to the edge's sourceHandle (the id).
        handoffs_cfg = cfg.get("handoffs") or []
        handoff_name_to_id: dict[str, str] = {}
        for h in handoffs_cfg:
            if isinstance(h, dict) and h.get("name") and h.get("id"):
                handoff_name_to_id[h["name"]] = h["id"]
        handoff_ids: set[str] = (
            set(handoff_name_to_id.values()) | set(cfg.get("handoff_handles") or [])
        )
        tool_specs: list[ToolSpec] = ctx.tool_registry.specs_for(tool_names)

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        if user_msg is not None:
            messages.append(Message(role="user", content=user_msg if isinstance(user_msg, str) else str(user_msg)))

        intermediate_steps: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        # If handoffs are declared, ask the LLM for JSON so we can parse
        # ``next_handle`` deterministically (text-mode handoff).
        use_text_handoff = bool(handoff_name_to_id) and not tool_specs
        request_format = "json" if use_text_handoff else "text"

        for step in range(max_iter):
            try:
                resp = await ctx.llm_service.invoke(
                    model_id,
                    LLMRequest(
                        messages=messages,
                        tools=tool_specs or None,
                        response_format=request_format,
                    ),
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

            # Tool-call-based handoff (model emitted a function call whose name
            # matches a declared handoff).
            if resp.tool_calls and handoff_ids:
                first_tc = resp.tool_calls[0]
                if first_tc.name in handoff_ids:
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

            # Text-mode handoff (JSON response with a ``next_handle`` field
            # whose value is either a handoff name OR id).
            chosen_id = _extract_text_handoff(
                resp, handoff_name_to_id=handoff_name_to_id, handoff_ids=handoff_ids
            )
            if chosen_id is not None:
                intermediate_steps.append({"handoff": chosen_id, "from": "text"})
                return NodeExecutionResult(
                    status="success",
                    output={
                        "answer": resp.content,
                        "handoff": chosen_id,
                        "parsed_json": resp.parsed_json,
                        "intermediate_steps": intermediate_steps,
                        "usage": resp.usage.model_dump(),
                    },
                    next_handle=chosen_id,
                    events=events,
                )

            if not resp.tool_calls:
                # Final answer reached.
                return NodeExecutionResult(
                    status="success",
                    output={
                        "answer": resp.content,
                        "content": resp.content,
                        "parsed_json": resp.parsed_json,
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


def _extract_text_handoff(
    resp: Any,
    *,
    handoff_name_to_id: dict[str, str],
    handoff_ids: set[str],
) -> str | None:
    """Inspect a JSON-mode LLM response for a ``next_handle`` field.

    Returns the resolved handoff **id** (matching the outgoing edge's
    ``sourceHandle``), or None if the response doesn't request a handoff.

    Accepts the value as either a handoff name (mapped via
    ``handoff_name_to_id``) or an id directly. Also tolerates the
    LLM-wrapped-in-text case by attempting to JSON-parse ``resp.content``
    when ``parsed_json`` is None.
    """
    payload = resp.parsed_json
    if payload is None and isinstance(resp.content, str):
        # The provider didn't parse it for us — try ourselves, leniently.
        import json as _json

        text = resp.content.strip()
        # Strip code fences the model may have emitted.
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            payload = _json.loads(text)
        except (ValueError, TypeError):
            payload = None
    if not isinstance(payload, dict):
        return None
    requested = payload.get("next_handle") or payload.get("handoff")
    if not isinstance(requested, str):
        return None
    if requested in handoff_name_to_id:
        return handoff_name_to_id[requested]
    if requested in handoff_ids:
        return requested
    return None


def _failed(node: Node, message: str, *, error_handle: str = "error") -> NodeExecutionResult:
    return NodeExecutionResult(
        status="failed",
        output={"error": message},
        next_handle=error_handle,
        error={"message": message, "node_id": node.id, "node_type": node.type},
    )
