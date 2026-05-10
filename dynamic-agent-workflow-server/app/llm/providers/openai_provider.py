"""OpenAI provider — thin specialization of OpenAICompatibleProvider."""
from __future__ import annotations

import httpx

from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://api.openai.com",
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            http_client=http_client,
            base_url=base_url,
            api_path="/v1/chat/completions",
            require_api_key=True,
            provider_name="openai",
            timeout_seconds=timeout_seconds,
        )
