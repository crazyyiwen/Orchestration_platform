"""LLM node — invokes ``LLMService.invoke`` with resolved messages (spec §7).

Config shape::

    {
      "model_id": "gpt-4o" | "mock-fast" | ...,
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "{{system.userQuery}}"}
      ],
      "response_format": "text" | "json",
      "temperature": 0.0,
      "max_tokens": 1024,
      "json_schema": {...}     # optional structured output
    }
"""
from __future__ import annotations

from typing import Any

from app.core.errors import ConfigurationError, WorkflowServerError
from app.llm.types import LLMRequest, Message
from app.schemas.node import Node
from app.schemas.validation import ValidationIssue, ValidationSeverity
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.variables import VariableResolver


@register("llm")
class LLMNodeExecutor(BaseNodeExecutor):
    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[ValidationIssue]:
        issues = []
        if not config.get("model_id"):
            issues.append(
                ValidationIssue(
                    code="llm_missing_model_id",
                    severity=ValidationSeverity.ERROR,
                    message="LLM node requires config.model_id",
                )
            )
        msgs = config.get("messages")
        if not isinstance(msgs, list) or not msgs:
            issues.append(
                ValidationIssue(
                    code="llm_missing_messages",
                    severity=ValidationSeverity.ERROR,
                    message="LLM node requires config.messages (non-empty list)",
                )
            )
        return issues

    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        cfg = node.config or {}
        # Frontend uses ``config.model``; legacy uses ``model_id``.
        requested_model = cfg.get("model_id") or cfg.get("model")
        model_id = requested_model or ctx.settings.DEFAULT_MODEL_ID
        if not model_id:
            return _failed(node, "model_id is missing")
        if not ctx.llm_service.models.has(model_id):
            import logging as _lg

            _lg.getLogger(__name__).warning(
                "llm node %r: model %r not in registry; falling back to %r",
                node.id, model_id, ctx.settings.DEFAULT_MODEL_ID,
            )
            model_id = ctx.settings.DEFAULT_MODEL_ID

        resolver = VariableResolver(state.get("variables", {}))
        try:
            messages = _resolve_messages(cfg.get("messages") or [], resolver)
        except ValueError as e:
            return _failed(node, f"messages malformed: {e}")

        # Frontend's ``config.instructions`` is the system prompt; if present
        # and no system message was already provided, prepend it.
        if cfg.get("instructions") and not any(m.role == "system" for m in messages):
            from app.llm.types import Message as _Msg

            messages.insert(
                0, _Msg(role="system", content=resolver.resolve_string(cfg["instructions"]))
            )

        # If the workflow declares ``outputVariables`` (frontend structured-output
        # schema), default response_format to json so the LLM emits a JSON doc.
        response_format = cfg.get("response_format")
        if not response_format and cfg.get("outputVariables"):
            response_format = "json"
        response_format = response_format or "text"

        request = LLMRequest(
            messages=messages,
            response_format=response_format,
            json_schema=cfg.get("json_schema"),
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            top_p=cfg.get("top_p"),
            stop=cfg.get("stop"),
        )

        try:
            resp = await ctx.llm_service.invoke(model_id, request)
        except (ConfigurationError, WorkflowServerError) as e:
            return _failed(node, str(e), error_handle="error")

        # Spec §7: store provider, model, content, parsed_json, usage.
        envelope = {
            "provider": resp.provider,
            "model": resp.model,
            "content": resp.content,
            "answer": resp.content,  # convenience alias used by templates
            "parsed_json": resp.parsed_json,
            "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
            "usage": resp.usage.model_dump(),
            "finish_reason": resp.finish_reason,
        }

        # Frontend structured-output contract: when the node declares
        # ``outputVariables`` and the model produced a JSON object, the node's
        # *result* IS that structured object. This makes the common pattern
        #   stateUpdates: [{key: "flow.intentResult", value: "{{nodes.llm.result}}"}]
        #   ... then {{flow.intentResult.aggregation}} downstream
        # work exactly as the workflow author expects. The raw envelope stays
        # available under ``result._llm`` for debugging / token accounting.
        if cfg.get("outputVariables") and isinstance(resp.parsed_json, dict):
            result_payload: dict[str, Any] = dict(resp.parsed_json)
            result_payload["_llm"] = envelope
        else:
            result_payload = envelope

        return NodeExecutionResult(
            status="success",
            output=result_payload,
            next_handle="out",
            events=[
                {
                    "type": "llm_completed",
                    "payload": {
                        "model": resp.model,
                        "provider": resp.provider,
                        "usage": resp.usage.model_dump(),
                        "finish_reason": resp.finish_reason,
                    },
                }
            ],
        )


def _resolve_messages(raw: list, resolver: VariableResolver) -> list[Message]:
    import json as _json

    out: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        resolved_content = (
            resolver.resolve_string(content) if isinstance(content, str) else content
        )

        # Frontend structured message input: when ``content`` is empty/blank
        # the real text lives in a ``fields: [{label, value}]`` array. Build
        # the message body as ``label: <resolved value>`` lines so the LLM
        # receives the user's actual input.
        fields = item.get("fields")
        if (not resolved_content) and isinstance(fields, list) and fields:
            lines: list[str] = []
            for f in fields:
                if not isinstance(f, dict):
                    continue
                label = f.get("label") or f.get("name") or ""
                val = resolver.resolve_value(f.get("value"))
                if not isinstance(val, str):
                    val = _json.dumps(val, ensure_ascii=False)
                lines.append(f"{label}: {val}" if label else val)
            resolved_content = "\n".join(lines)

        if not isinstance(resolved_content, (str, type(None))):
            # Stringify non-string resolved content (e.g. lists/dicts) for safety.
            resolved_content = _json.dumps(resolved_content, ensure_ascii=False)
        out.append(
            Message(
                role=role,
                content=resolved_content,
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
            )
        )
    return out


def _failed(node: Node, message: str, *, error_handle: str = "error") -> NodeExecutionResult:
    return NodeExecutionResult(
        status="failed",
        output=None,
        next_handle=error_handle,
        error={"message": message, "node_id": node.id, "node_type": node.type},
        events=[{"type": "node_failed", "payload": {"message": message}}],
    )
