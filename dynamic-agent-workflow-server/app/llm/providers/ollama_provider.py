"""Ollama provider — uses its OpenAI-compatible /v1 endpoint.

No API key required. Default base URL points to local daemon.
"""
from __future__ import annotations

import httpx

from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,  # local models can be slower
    ) -> None:
        super().__init__(
            api_key="",
            http_client=http_client,
            base_url=base_url,
            api_path="/v1/chat/completions",
            require_api_key=False,
            provider_name="ollama",
            timeout_seconds=timeout_seconds,
        )
