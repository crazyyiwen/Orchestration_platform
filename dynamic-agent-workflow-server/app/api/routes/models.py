"""Models / providers API surface (spec §13)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.errors import ConfigurationError, WorkflowServerError
from app.llm.service import LLMService
from app.llm.types import LLMRequest, Message

router = APIRouter(tags=["models"])


def _llm(request: Request) -> LLMService:
    svc: LLMService | None = getattr(request.app.state, "llm", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return svc


@router.get("/api/models")
async def list_models(request: Request) -> dict[str, Any]:
    svc = _llm(request)
    providers = svc.providers
    out: list[dict[str, Any]] = []
    for entry in svc.models.list():
        provider = providers.get(entry.provider) if providers.has(entry.provider) else None
        available = bool(provider and provider.available)
        item = entry.to_public()
        item["available"] = available
        item["provider_registered"] = providers.has(entry.provider)
        out.append(item)
    return {"models": out}


@router.get("/api/providers")
async def list_providers(request: Request) -> dict[str, Any]:
    svc = _llm(request)
    return {
        "providers": [
            {
                "name": p.name,
                "available": p.available,
                "supports_chat": p.supports_chat,
                "supports_json_mode": p.supports_json_mode,
                "supports_tools": p.supports_tools,
                "supports_streaming": p.supports_streaming,
            }
            for p in svc.providers.list()
        ]
    }


class ModelTestRequest(BaseModel):
    model_id: str
    prompt: str = "Hello!"
    response_format: str = Field(default="text", pattern="^(text|json)$")


@router.post("/api/models/test")
async def test_model(request: Request, body: ModelTestRequest) -> dict[str, Any]:
    """Round-trip a single chat call through the configured provider.

    Useful as a smoke test from the frontend or curl. Errors are surfaced
    as domain errors (sanitized in the global handler).
    """
    svc = _llm(request)
    req = LLMRequest(
        messages=[Message(role="user", content=body.prompt)],
        response_format=body.response_format,  # type: ignore[arg-type]
    )
    try:
        resp = await svc.invoke(body.model_id, req)
    except (ConfigurationError, WorkflowServerError):
        raise
    return {
        "provider": resp.provider,
        "model": resp.model,
        "content": resp.content,
        "parsed_json": resp.parsed_json,
        "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
        "usage": resp.usage.model_dump(),
        "finish_reason": resp.finish_reason,
    }
