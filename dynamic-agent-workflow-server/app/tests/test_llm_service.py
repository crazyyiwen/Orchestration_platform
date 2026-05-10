"""LLMService dispatch + capability gating + JSON-mode fallback."""
from __future__ import annotations

import pytest

from app.core.errors import ConfigurationError
from app.llm.providers.base import BaseLLMProvider
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry
from app.llm.service import LLMService, _inject_json_instruction
from app.llm.types import LLMRequest, LLMResponse, LLMUsage, Message, ToolSpec


# Async detection is automatic (pyproject asyncio_mode=auto); no module-level mark needed.


# --- A controllable provider that records what it received ---------------


class _RecordingProvider(BaseLLMProvider):
    name = "rec"

    def __init__(self, *, content: str = "ok", parsed: object | None = None) -> None:
        self.last_request: LLMRequest | None = None
        self.last_model: str | None = None
        self._content = content
        self._parsed = parsed

    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        self.last_request = request
        self.last_model = model
        return LLMResponse(
            provider=self.name,
            model=model,
            content=self._content,
            parsed_json=self._parsed,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
        )


def _service(entry: ModelEntry, provider: BaseLLMProvider) -> LLMService:
    return LLMService(
        models=ModelRegistry([entry]),
        providers=ProviderRegistry({provider.name: provider}),
    )


# --- dispatch + tagging --------------------------------------------------


async def test_invoke_routes_to_provider_and_passes_native_model_name() -> None:
    entry = ModelEntry(
        id="my-model", provider="rec", model="rec-internal-v3", capabilities=frozenset({"chat"})
    )
    rec = _RecordingProvider()
    svc = _service(entry, rec)

    resp = await svc.invoke("my-model", LLMRequest(messages=[Message(role="user", content="hi")]))

    # Provider gets the *internal* model name; response gets re-tagged with user-facing id.
    assert rec.last_model == "rec-internal-v3"
    assert resp.model == "my-model"
    assert resp.provider == "rec"


async def test_invoke_unknown_model_id_raises() -> None:
    svc = _service(
        ModelEntry(id="x", provider="rec", model="x", capabilities=frozenset({"chat"})),
        _RecordingProvider(),
    )
    with pytest.raises(ConfigurationError, match="unknown model_id"):
        await svc.invoke("nope", LLMRequest(messages=[Message(role="user", content="hi")]))


# --- capability gating ---------------------------------------------------


async def test_invoke_rejects_when_chat_capability_missing() -> None:
    entry = ModelEntry(id="x", provider="rec", model="x", capabilities=frozenset())
    svc = _service(entry, _RecordingProvider())
    with pytest.raises(ConfigurationError, match="'chat'"):
        await svc.invoke("x", LLMRequest(messages=[Message(role="user", content="hi")]))


async def test_invoke_rejects_tools_when_model_lacks_tools_capability() -> None:
    entry = ModelEntry(id="x", provider="rec", model="x", capabilities=frozenset({"chat"}))
    svc = _service(entry, _RecordingProvider())
    req = LLMRequest(
        messages=[Message(role="user", content="hi")],
        tools=[ToolSpec(name="t", description="d", parameters={})],
    )
    with pytest.raises(ConfigurationError, match="does not support tools"):
        await svc.invoke("x", req)


async def test_invoke_rejects_streaming_when_unsupported() -> None:
    entry = ModelEntry(id="x", provider="rec", model="x", capabilities=frozenset({"chat"}))
    svc = _service(entry, _RecordingProvider())
    req = LLMRequest(messages=[Message(role="user", content="hi")], stream=True)
    with pytest.raises(ConfigurationError, match="streaming"):
        await svc.invoke("x", req)


# --- JSON mode fallback --------------------------------------------------


async def test_json_mode_passes_through_when_model_supports_it() -> None:
    entry = ModelEntry(
        id="json-native",
        provider="rec",
        model="rec",
        capabilities=frozenset({"chat", "json_mode"}),
    )
    rec = _RecordingProvider(content='{"x": 1}', parsed={"x": 1})
    svc = _service(entry, rec)
    req = LLMRequest(
        messages=[Message(role="user", content="give me JSON")],
        response_format="json",
    )
    resp = await svc.invoke("json-native", req)

    # No injection — response_format remains "json" and no system message added.
    assert rec.last_request is not None
    assert rec.last_request.response_format == "json"
    assert all(m.role != "system" for m in rec.last_request.messages)
    assert resp.parsed_json == {"x": 1}


async def test_json_mode_fallback_injects_instruction_and_parses_text() -> None:
    entry = ModelEntry(
        id="no-json", provider="rec", model="rec", capabilities=frozenset({"chat"})
    )
    rec = _RecordingProvider(content='{"x": 1}')
    svc = _service(entry, rec)
    req = LLMRequest(
        messages=[Message(role="user", content="give me JSON")],
        response_format="json",
    )
    resp = await svc.invoke("no-json", req)

    # Provider was called in text mode with a system instruction prepended.
    assert rec.last_request is not None
    assert rec.last_request.response_format == "text"
    assert rec.last_request.messages[0].role == "system"
    assert "JSON" in (rec.last_request.messages[0].content or "")
    # Service parsed the text response back into parsed_json.
    assert resp.parsed_json == {"x": 1}


async def test_json_fallback_strips_code_fences() -> None:
    entry = ModelEntry(
        id="no-json", provider="rec", model="rec", capabilities=frozenset({"chat"})
    )
    rec = _RecordingProvider(content='```json\n{"y": 2}\n```')
    svc = _service(entry, rec)
    req = LLMRequest(
        messages=[Message(role="user", content="x")],
        response_format="json",
    )
    resp = await svc.invoke("no-json", req)
    assert resp.parsed_json == {"y": 2}


async def test_json_fallback_unparseable_leaves_parsed_json_none_without_raising() -> None:
    entry = ModelEntry(
        id="no-json", provider="rec", model="rec", capabilities=frozenset({"chat"})
    )
    rec = _RecordingProvider(content="this is not JSON at all")
    svc = _service(entry, rec)
    resp = await svc.invoke(
        "no-json",
        LLMRequest(
            messages=[Message(role="user", content="x")],
            response_format="json",
        ),
    )
    assert resp.parsed_json is None
    assert resp.content == "this is not JSON at all"


# --- _inject_json_instruction direct unit test --------------------------


def test_inject_json_instruction_appends_to_existing_system_message() -> None:
    req = LLMRequest(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hi"),
        ],
        response_format="json",
    )
    out = _inject_json_instruction(req)
    assert out.messages[0].role == "system"
    assert "You are helpful." in (out.messages[0].content or "")
    assert "JSON" in (out.messages[0].content or "")
    assert out.response_format == "text"
    # Original was not mutated.
    assert req.response_format == "json"


def test_inject_json_instruction_prepends_when_no_system_message() -> None:
    req = LLMRequest(
        messages=[Message(role="user", content="hi")], response_format="json"
    )
    out = _inject_json_instruction(req)
    assert out.messages[0].role == "system"
    assert "JSON" in (out.messages[0].content or "")
    assert out.messages[1].role == "user"
