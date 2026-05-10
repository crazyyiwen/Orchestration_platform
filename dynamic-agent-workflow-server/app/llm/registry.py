"""Provider + model registries.

The :class:`ModelRegistry` is loaded from ``config/models.yaml`` (decision
captured in the architecture plan) and maps a user-facing ``model_id`` to its
provider name + provider-specific model string + declared capabilities. The
:class:`ProviderRegistry` holds instantiated providers keyed by name.

Both are built once at startup. Hot-reload is out of scope for v1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.base import BaseLLMProvider
from app.llm.providers.huggingface_provider import HuggingFaceProvider
from app.llm.providers.mock_provider import MockProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.vllm_provider import VLLMProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    provider: str
    model: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    label: str | None = None
    description: str | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "capabilities": sorted(self.capabilities),
            "label": self.label,
            "description": self.description,
        }


# ----- ModelRegistry ------------------------------------------------------


class ModelRegistry:
    def __init__(self, entries: Iterable[ModelEntry]) -> None:
        self._by_id: dict[str, ModelEntry] = {}
        for e in entries:
            if e.id in self._by_id:
                raise ConfigurationError(f"duplicate model id in registry: {e.id!r}")
            self._by_id[e.id] = e

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self._by_id[model_id]
        except KeyError as e:
            raise ConfigurationError(
                f"unknown model_id {model_id!r}", details={"model_id": model_id}
            ) from e

    def has(self, model_id: str) -> bool:
        return model_id in self._by_id

    def list(self) -> list[ModelEntry]:
        return list(self._by_id.values())

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRegistry":
        if not path.exists():
            raise ConfigurationError(f"model registry file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_entries = data.get("models") or []
        entries: list[ModelEntry] = []
        for i, raw in enumerate(raw_entries):
            if not isinstance(raw, dict):
                raise ConfigurationError(f"model registry entry {i} must be a mapping")
            try:
                entries.append(
                    ModelEntry(
                        id=raw["id"],
                        provider=raw["provider"],
                        model=raw["model"],
                        capabilities=frozenset(raw.get("capabilities") or []),
                        label=raw.get("label"),
                        description=raw.get("description"),
                    )
                )
            except KeyError as e:
                raise ConfigurationError(
                    f"model registry entry {i} missing required field: {e}"
                ) from e
        return cls(entries)


# ----- ProviderRegistry ---------------------------------------------------


class ProviderRegistry:
    def __init__(self, providers: dict[str, BaseLLMProvider]) -> None:
        self._providers = dict(providers)

    def get(self, name: str) -> BaseLLMProvider:
        try:
            return self._providers[name]
        except KeyError as e:
            raise ConfigurationError(
                f"provider {name!r} is not registered or not configured",
                details={"provider": name, "registered": sorted(self._providers)},
            ) from e

    def has(self, name: str) -> bool:
        return name in self._providers

    def list(self) -> list[BaseLLMProvider]:
        return list(self._providers.values())

    def names(self) -> list[str]:
        return sorted(self._providers)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> "ProviderRegistry":
        """Instantiate every provider that's *configurable* under the current
        settings. A provider with a missing API key is omitted entirely (so
        ``has(name)`` is False); the route handler / service surfaces that
        clearly via 'unknown provider'."""
        providers: dict[str, BaseLLMProvider] = {}
        # Mock is always on — it doesn't talk to anything external.
        providers["mock"] = MockProvider()

        # Providers below all need the shared httpx client to actually run.
        if http_client is not None:
            if settings.OPENAI_API_KEY:
                providers["openai"] = OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY, http_client=http_client
                )
            if settings.ANTHROPIC_API_KEY:
                providers["anthropic"] = AnthropicProvider(
                    api_key=settings.ANTHROPIC_API_KEY, http_client=http_client
                )
            # Ollama and vLLM don't require keys — they're "available" iff a
            # local daemon answers, but we register them so workflows can opt in.
            providers["ollama"] = OllamaProvider(http_client=http_client)
            providers["vllm"] = VLLMProvider(http_client=http_client)

        # HuggingFace placeholder — registered so config that targets it
        # produces a clear error when invoked, not "unknown provider".
        providers["huggingface"] = HuggingFaceProvider(api_key=settings.HUGGINGFACE_API_KEY)
        log.info("providers registered: %s", sorted(providers))
        return cls(providers)

    def register(self, provider: BaseLLMProvider) -> None:
        """Test hook: allow swapping a provider in (e.g. with a MockTransport)."""
        self._providers[provider.name] = provider
