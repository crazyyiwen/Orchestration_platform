"""MockProvider: deterministic outputs for tests + offline dev."""
from __future__ import annotations

import pytest

from app.llm.providers.mock_provider import MockProvider
from app.llm.types import LLMRequest, Message, ToolSpec


pytestmark = pytest.mark.asyncio


async def test_mock_text_completion_echoes_user_message() -> None:
    p = MockProvider()
    req = LLMRequest(messages=[Message(role="user", content="hello world")])
    resp = await p.chat(req, model="mock")
    assert resp.provider == "mock"
    assert resp.model == "mock"
    assert "hello world" in (resp.content or "")
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens > 0


async def test_mock_json_mode_returns_valid_json() -> None:
    p = MockProvider()
    req = LLMRequest(
        messages=[Message(role="user", content="produce JSON")],
        response_format="json",
    )
    resp = await p.chat(req, model="mock")
    assert resp.parsed_json is not None
    assert resp.parsed_json["echo"] == "produce JSON"


async def test_mock_tool_call_when_metadata_signals() -> None:
    p = MockProvider()
    req = LLMRequest(
        messages=[Message(role="user", content="search for cats")],
        tools=[ToolSpec(name="search", description="Search the web", parameters={})],
        metadata={"mock_tool_call": True},
    )
    resp = await p.chat(req, model="mock")
    assert resp.content is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].arguments == {"input": "search for cats"}
    assert resp.finish_reason == "tool_calls"


async def test_mock_capabilities_reported() -> None:
    p = MockProvider()
    assert p.supports_chat
    assert p.supports_json_mode
    assert p.supports_tools
    assert p.supports_streaming
    assert p.available
