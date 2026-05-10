"""ModelRegistry + ProviderRegistry tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.llm.providers.mock_provider import MockProvider
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry


CONFIG_YAML = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def test_model_registry_loads_yaml() -> None:
    reg = ModelRegistry.from_yaml(CONFIG_YAML)
    ids = {e.id for e in reg.list()}
    assert "mock-fast" in ids
    assert reg.has("mock-fast")
    entry = reg.get("mock-fast")
    assert entry.provider == "mock"
    assert "chat" in entry.capabilities


def test_model_registry_unknown_raises_clean_error() -> None:
    reg = ModelRegistry.from_yaml(CONFIG_YAML)
    with pytest.raises(ConfigurationError, match="unknown model_id"):
        reg.get("does-not-exist")


def test_model_registry_rejects_duplicates() -> None:
    with pytest.raises(ConfigurationError, match="duplicate model id"):
        ModelRegistry(
            [
                ModelEntry(id="x", provider="mock", model="x"),
                ModelEntry(id="x", provider="openai", model="gpt"),
            ]
        )


def test_model_registry_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        ModelRegistry.from_yaml(Path("/nonexistent/models.yaml"))


def test_provider_registry_from_settings_always_includes_mock() -> None:
    settings = Settings(OPENAI_API_KEY="", ANTHROPIC_API_KEY="")
    reg = ProviderRegistry.from_settings(settings, http_client=None)
    assert reg.has("mock")
    assert isinstance(reg.get("mock"), MockProvider)


def test_provider_registry_omits_keyed_providers_without_keys() -> None:
    settings = Settings(OPENAI_API_KEY="", ANTHROPIC_API_KEY="")
    reg = ProviderRegistry.from_settings(settings, http_client=None)
    # No http_client means no key-bearing providers either, but mock + huggingface are still there.
    assert reg.has("mock")
    assert not reg.has("openai")
    assert not reg.has("anthropic")


@pytest.mark.asyncio
async def test_provider_registry_includes_keyed_providers_when_configured() -> None:
    import httpx

    settings = Settings(OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="sk-ant-test")
    async with httpx.AsyncClient() as http:
        reg = ProviderRegistry.from_settings(settings, http_client=http)
    assert reg.has("openai")
    assert reg.has("anthropic")
    assert reg.has("ollama")
    assert reg.has("vllm")


def test_provider_registry_unknown_raises() -> None:
    reg = ProviderRegistry({"mock": MockProvider()})
    with pytest.raises(ConfigurationError, match="not registered"):
        reg.get("nonexistent")
