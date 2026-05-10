"""HuggingFace provider — placeholder per spec §8.

The HF inference API surface differs from OpenAI's chat completions; a real
implementation needs per-model handling for text-generation vs.
chat-completion endpoints. Until that's prioritized, this provider raises a
clear error so config that *would* have used it fails fast and informatively
instead of pretending to work.
"""
from __future__ import annotations

from app.core.errors import ConfigurationError
from app.llm.providers.base import BaseLLMProvider
from app.llm.types import LLMRequest, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    name = "huggingface"

    def __init__(self, *, api_key: str = "") -> None:
        self._api_key = api_key

    @property
    def available(self) -> bool:
        return False  # placeholder — never actually serves requests

    async def chat(self, request: LLMRequest, *, model: str) -> LLMResponse:
        raise ConfigurationError(
            "huggingface provider is a placeholder; configure another provider "
            "or implement HuggingFaceProvider.chat()",
            details={"provider": self.name, "model": model},
        )
