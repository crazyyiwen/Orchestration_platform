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
            payload: dict = {"echo": last_user, "model": model}
            # If the caller explicitly requested a handoff via metadata, honor it.
            forced_handle = request.metadata.get("mock_next_handle")
            if forced_handle:
                payload = {"next_handle": forced_handle, "reason": "mock"}
            else:
                # Heuristic: if the system prompt enumerates valid ``next_handle``
                # values (the workflow author's agent prompt pattern), pick the
                # first one — that's enough to exercise the handoff routing in
                # dev/test without a real LLM. Picks the handle whose name
                # matches a keyword in the user's query when possible.
                candidates = _extract_handle_candidates(request)
                if candidates:
                    chosen = _pick_handle(candidates, last_user)
                    payload = {"next_handle": chosen, "reason": "mock"}
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


def _extract_handle_candidates(request: LLMRequest) -> list[str]:
    """Find ``next_handle`` candidate values in the system prompt.

    The workflow author's prompt typically lists valid values like:
        ``"next_handle": "fetching_data"`` or ``Valid next_handle values: fetching_data, summary, final_report``
    We extract all such tokens so MockProvider can deterministically pick one,
    enabling end-to-end handoff routing without a real LLM.
    """
    import re

    text = "\n".join(m.content or "" for m in request.messages if m.role == "system")
    if not text:
        return []
    # Match ``"next_handle": "value"`` patterns.
    quoted = re.findall(r'"next_handle"\s*:\s*"([a-zA-Z0-9_\-]+)"', text)
    if quoted:
        # Preserve order of first occurrence, drop duplicates.
        seen: list[str] = []
        for v in quoted:
            if v not in seen:
                seen.append(v)
        return seen
    return []


def _pick_handle(candidates: list[str], user_query: str) -> str:
    """Choose the candidate whose name best matches the user's intent.

    Pick the candidate whose name appears (case-insensitively, token-prefix
    match) in the user query — that's good enough for routing tests like
    'Fetch energy usage data' → 'fetching_data'. Falls back to the first
    candidate if no match.
    """
    if not user_query:
        return candidates[0]
    uq = user_query.lower()
    for cand in candidates:
        # Match on the first underscore-token (e.g. "fetching_data" → "fetch")
        # so verbs match: "Fetch energy" matches "fetching_data" via "fetch".
        head = cand.split("_", 1)[0].lower()
        if head and head in uq:
            return cand
    # Also try whole-name substring match.
    for cand in candidates:
        if cand.lower().replace("_", " ") in uq or cand.lower() in uq:
            return cand
    return candidates[0]


def _serialize_messages(request: LLMRequest) -> str:
    return " ".join(m.content or "" for m in request.messages)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic: 4 chars per token. Good enough for usage telemetry."""
    return max(1, len(text) // 4) if text else 0
