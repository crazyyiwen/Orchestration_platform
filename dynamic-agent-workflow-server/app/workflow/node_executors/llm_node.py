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
        model_id = cfg.get("model_id") or ctx.settings.DEFAULT_MODEL_ID
        if not model_id:
            return _failed(node, "model_id is missing")

        resolver = VariableResolver(state.get("variables", {}))
        try:
            messages = _resolve_messages(cfg.get("messages") or [], resolver)
        except ValueError as e:
            return _failed(node, f"messages malformed: {e}")

        request = LLMRequest(
            messages=messages,
            response_format=cfg.get("response_format", "text"),
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
        result_payload = {
            "provider": resp.provider,
            "model": resp.model,
            "content": resp.content,
            "answer": resp.content,  # convenience alias used by templates
            "parsed_json": resp.parsed_json,
            "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
            "usage": resp.usage.model_dump(),
            "finish_reason": resp.finish_reason,
        }
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
    out: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        resolved_content = (
            resolver.resolve_string(content) if isinstance(content, str) else content
        )
        if not isinstance(resolved_content, (str, type(None))):
            # Stringify non-string resolved content (e.g. lists/dicts) for safety.
            import json as _json

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
