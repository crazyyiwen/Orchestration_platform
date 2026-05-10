"""Deterministic mock provider for tests, demos, and offline dev.

The output is a function of the input — no randomness — so tests can assert
exact strings. JSON-mode returns a parseable JSON document. Tool-use returns
the first declared tool with the most recent user message as its arguments
when the request opts in via metadata, and a plain text response otherwise.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from app.llm.providers.base import BaseLLMProvider
from app.llm.types import LLMRequest, LLMResponse, LLMUsage, ToolCall


class MockProvider(BaseLLMProvider):
    name = "mock"

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        return True

    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        last_user = _last_user_text(request)
        usage = LLMUsage(
            prompt_tokens=_estimate_tokens(_serialize_messages(request)),
            completion_tokens=0,
            total_tokens=0,
        )

        # Tool-call mode: when the caller passes tools and signals a tool use
        # via metadata.mock_tool_call=true, return a fake invocation of the
        # first tool.
        if request.tools and request.metadata.get("mock_tool_call"):
            tool = request.tools[0]
            tc = ToolCall(id="mock-tc-1", name=tool.name, arguments={"input": last_user})
            return LLMResponse(
                provider=self.name,
                model=model,
                content=None,
                tool_calls=[tc],
                usage=usage,
                finish_reason="tool_calls",
            )

        if request.response_format == "json":
            payload = {"echo": last_user, "model": model}
            content = json.dumps(payload, ensure_ascii=False)
            usage.completion_tokens = _estimate_tokens(content)
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return LLMResponse(
                provider=self.name,
                model=model,
                content=content,
                parsed_json=payload,
                usage=usage,
                finish_reason="stop",
            )

        text = f"[mock:{model}] {last_user}"
        usage.completion_tokens = _estimate_tokens(text)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return LLMResponse(
            provider=self.name,
            model=model,
            content=text,
            usage=usage,
            finish_reason="stop",
        )

    async def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[str]:
        last_user = _last_user_text(request)
        for chunk in (f"[mock:{model}] ", last_user):
            yield chunk


def _last_user_text(request: LLMRequest) -> str:
    for m in reversed(request.messages):
        if m.role == "user" and m.content is not None:
            return m.content
    return ""


def _serialize_messages(request: LLMRequest) -> str:
    return " ".join(m.content or "" for m in request.messages)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic: 4 chars per token. Good enough for usage telemetry."""
    return max(1, len(text) // 4) if text else 0
