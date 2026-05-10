"""Anthropic Messages API provider.

Differences from OpenAI's chat-completions wire format:
  * ``system`` lives on a separate top-level field, not in ``messages``.
  * Response content is a list of content blocks (``text`` / ``tool_use``).
  * Tool calls live inside the content array as ``tool_use`` blocks.
  * ``max_tokens`` is required.
  * Uses ``x-api-key`` + ``anthropic-version`` headers, not ``Authorization``.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.errors import ConfigurationError, WorkflowServerError
from app.llm.providers.base import BaseLLMProvider
from app.llm.types import LLMRequest, LLMResponse, LLMUsage, Message, ToolCall

log = logging.getLogger(__name__)

ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://api.anthropic.com",
        api_version: str = ANTHROPIC_API_VERSION,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("anthropic provider requires ANTHROPIC_API_KEY")
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._timeout = timeout_seconds

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        url = f"{self._base_url}/v1/messages"
        payload = self._build_payload(request, model=model)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "Content-Type": "application/json",
        }
        try:
            resp = await self._http.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as e:
            raise WorkflowServerError(
                f"anthropic HTTP error: {e}",
                details={"provider": self.name, "model": model},
            ) from e
        if resp.status_code >= 400:
            raise WorkflowServerError(
                f"anthropic returned {resp.status_code}",
                details={
                    "provider": self.name,
                    "status": resp.status_code,
                    "body": _truncate(resp.text, 800),
                },
            )
        return self._parse_response(resp.json(), request=request, model=model)

    # ----- payload / response shaping -------------------------------------

    def _build_payload(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        system_content: str | None = None
        chat_messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                # Anthropic concatenates multiple system messages.
                if m.content:
                    system_content = (
                        m.content if system_content is None else f"{system_content}\n{m.content}"
                    )
                continue
            chat_messages.append(_message_to_anthropic(m))

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system_content is not None:
            payload["system"] = system_content
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]
        return payload

    def _parse_response(
        self, data: dict[str, Any], *, request: LLMRequest, model: str
    ) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or "",
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )

        content = "".join(text_parts) if text_parts else None
        parsed_json = None
        if request.response_format == "json" and isinstance(content, str):
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                parsed_json = None

        usage_data = data.get("usage") or {}
        prompt = int(usage_data.get("input_tokens", 0) or 0)
        completion = int(usage_data.get("output_tokens", 0) or 0)
        usage = LLMUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

        return LLMResponse(
            provider=self.name,
            model=model,
            content=content,
            parsed_json=parsed_json,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=data.get("stop_reason"),
            raw=data,
        )


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    """Convert one Message to Anthropic's content-block format."""
    if m.role == "tool":
        # Tool result message — Anthropic represents these as user messages
        # with a ``tool_result`` content block.
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content or "",
                }
            ],
        }
    if m.role == "assistant" and m.tool_calls:
        # Assistant turn that requests tool use.
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}
    return {"role": m.role, "content": m.content or ""}


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + "…"
