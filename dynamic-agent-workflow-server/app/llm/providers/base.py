"""Base interface for all LLM providers.

A provider knows how to translate ``LLMRequest`` → its native API and back.
The :class:`LLMService` is responsible for selecting which provider to use
(based on the ``ModelEntry.provider`` field), so providers themselves are
free of model-routing logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar

from app.llm.types import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    name: ClassVar[str] = ""

    @abstractmethod
    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        """Single-shot completion. Returns a normalized :class:`LLMResponse`."""

    async def stream(
        self, request: LLMRequest, *, model: str
    ) -> AsyncIterator[str]:
        """Streaming completion (chunks of generated text).

        Default implementation refuses; providers that support streaming
        override this. The ``async generator`` form lets callers ``async for``
        chunks with normal cancellation semantics.
        """
        raise NotImplementedError(f"streaming not supported by provider {self.name!r}")
        yield  # pragma: no cover  — make this an async generator

    # Capability hints. Per-model capabilities still come from the registry
    # (a provider may serve a model that doesn't support tools); these are the
    # provider-wide upper bounds.
    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def available(self) -> bool:
        """Whether this provider is configured to accept requests right now
        (e.g., its API key is set). Default: True."""
        return True
