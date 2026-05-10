"""vLLM provider — its OpenAI-compatible server speaks the same wire format."""
from __future__ import annotations

import httpx

from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    name = "vllm"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str = "http://localhost:8000",
        api_key: str = "",  # vLLM may run with or without an API key
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            http_client=http_client,
            base_url=base_url,
            api_path="/v1/chat/completions",
            require_api_key=False,
            provider_name="vllm",
            timeout_seconds=timeout_seconds,
        )
