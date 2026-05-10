"""OpenAI + Anthropic provider tests using httpx.MockTransport.

We assert two things per provider:
  1. Outgoing payload is shaped correctly (the API contract we maintain).
  2. The native API response is normalized into our :class:`LLMResponse`.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.core.errors import WorkflowServerError
from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.types import LLMRequest, Message, ToolSpec


pytestmark = pytest.mark.asyncio


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- OpenAI --------------------------------------------------------------


async def test_openai_chat_sends_expected_payload_and_parses_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi there!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    async with _client(handler) as http:
        provider = OpenAIProvider(api_key="sk-test", http_client=http)
        resp = await provider.chat(
            LLMRequest(messages=[Message(role="user", content="Hi")]),
            model="gpt-4o-mini",
        )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert resp.content == "Hi there!"
    assert resp.usage.total_tokens == 8
    assert resp.finish_reason == "stop"


async def test_openai_json_mode_sets_response_format_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"k": "v"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async with _client(handler) as http:
        provider = OpenAIProvider(api_key="sk-test", http_client=http)
        resp = await provider.chat(
            LLMRequest(
                messages=[Message(role="user", content="JSON please")],
                response_format="json",
            ),
            model="gpt-4o",
        )

    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert resp.parsed_json == {"k": "v"}


async def test_openai_tool_calls_are_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"q": "cats"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async with _client(handler) as http:
        provider = OpenAIProvider(api_key="sk-test", http_client=http)
        resp = await provider.chat(
            LLMRequest(
                messages=[Message(role="user", content="x")],
                tools=[ToolSpec(name="search", description="d", parameters={})],
            ),
            model="gpt-4o",
        )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].arguments == {"q": "cats"}


async def test_openai_5xx_raises_workflow_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream is down")

    async with _client(handler) as http:
        provider = OpenAIProvider(api_key="sk-test", http_client=http)
        with pytest.raises(WorkflowServerError, match="returned 503"):
            await provider.chat(
                LLMRequest(messages=[Message(role="user", content="x")]),
                model="gpt-4o",
            )


# --- Anthropic -----------------------------------------------------------


async def test_anthropic_chat_extracts_system_into_top_level_field() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "content": [{"type": "text", "text": "Hello!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        )

    async with _client(handler) as http:
        provider = AnthropicProvider(api_key="sk-ant-test", http_client=http)
        resp = await provider.chat(
            LLMRequest(
                messages=[
                    Message(role="system", content="Be concise."),
                    Message(role="user", content="Hi"),
                ],
                max_tokens=100,
            ),
            model="claude-sonnet-4-6",
        )

    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"]
    body = captured["body"]
    assert body["system"] == "Be concise."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["max_tokens"] == 100
    assert resp.content == "Hello!"
    assert resp.usage.prompt_tokens == 7
    assert resp.usage.completion_tokens == 2
    assert resp.usage.total_tokens == 9
    assert resp.finish_reason == "end_turn"


async def test_anthropic_tool_use_blocks_are_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Looking up..."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "search",
                        "input": {"q": "cats"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async with _client(handler) as http:
        provider = AnthropicProvider(api_key="sk-ant-test", http_client=http)
        resp = await provider.chat(
            LLMRequest(
                messages=[Message(role="user", content="search cats")],
                tools=[ToolSpec(name="search", description="", parameters={})],
            ),
            model="claude-sonnet-4-6",
        )
    assert resp.content == "Looking up..."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].arguments == {"q": "cats"}


async def test_anthropic_default_max_tokens_when_omitted() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async with _client(handler) as http:
        provider = AnthropicProvider(api_key="sk-ant-test", http_client=http)
        await provider.chat(
            LLMRequest(messages=[Message(role="user", content="hi")]),
            model="claude-haiku-4-5",
        )
    assert captured["body"]["max_tokens"] >= 256  # has a sensible default
