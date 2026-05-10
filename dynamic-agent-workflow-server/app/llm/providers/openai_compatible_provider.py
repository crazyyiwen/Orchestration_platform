"""OpenAI-compatible chat-completions provider.

Powers ``OpenAIProvider``, ``OllamaProvider`` (via ``/v1`` reverse-proxy) and
``VLLMProvider`` — they all speak the same wire format. Differences live in
defaults (base URL, auth header, optional max_tokens behavior).

Uses ``httpx.AsyncClient``; the client is injected so the lifespan can share
one pool across providers and tests can swap in ``httpx.MockTransport``.
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


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic ``/v1/chat/completions`` provider."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://api.openai.com",
        api_path: str = "/v1/chat/completions",
        require_api_key: bool = True,
        provider_name: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if require_api_key and not api_key:
            raise ConfigurationError(f"{provider_name or self.name} requires an API key")
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._api_path = api_path
        self._timeout = timeout_seconds
        self._require_api_key = require_api_key
        if provider_name:
            self.name = provider_name  # type: ignore[misc]

    @property
    def supports_chat(self) -> bool:
        return True

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
        return (not self._require_api_key) or bool(self._api_key)

    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        url = f"{self._base_url}{self._api_path}"
        payload = self._build_payload(request, model=model)
        headers = self._build_headers()
        try:
            resp = await self._http.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as e:
            raise WorkflowServerError(
                f"{self.name} HTTP error: {e}",
                details={"provider": self.name, "model": model},
            ) from e
        if resp.status_code >= 400:
            raise WorkflowServerError(
                f"{self.name} returned {resp.status_code}",
                details={
                    "provider": self.name,
                    "status": resp.status_code,
                    "body": _truncate(resp.text, 800),
                },
            )
        data = resp.json()
        return self._parse_response(data, request=request, model=model)

    # ----- payload / response shaping -------------------------------------

    def _build_payload(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_openai(m) for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        if request.stream:
            payload["stream"] = True
        if request.response_format == "json":
            if request.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.json_schema,
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _parse_response(
        self, data: dict[str, Any], *, request: LLMRequest, model: str
    ) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(
                ToolCall(id=tc.get("id") or "", name=fn.get("name") or "", arguments=args or {})
            )

        parsed_json = None
        if request.response_format == "json" and isinstance(content, str):
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                parsed_json = None

        usage_data = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_data.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_data.get("total_tokens", 0) or 0),
        )

        return LLMResponse(
            provider=self.name,
            model=model,
            content=content,
            parsed_json=parsed_json,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )


def _message_to_openai(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        out["content"] = m.content
    if m.name is not None:
        out["name"] = m.name
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    return out


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + "…"
