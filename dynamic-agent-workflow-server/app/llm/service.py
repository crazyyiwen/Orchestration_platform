"""LLMService — the single entry point used by node executors.

Responsibilities (per spec §8 + plan §7):

  1. Look up the ``ModelEntry`` by ``model_id``.
  2. Validate the request against the model's declared capabilities.
  3. Look up the provider instance.
  4. Apply a JSON-mode fallback when the model can't natively guarantee JSON
     (inject an instruction + parse the resulting text).
  5. Dispatch to the provider.
  6. Return a normalized :class:`LLMResponse` carrying the *requested*
     ``model_id`` (not just the provider's internal model name) so traces
     remain workflow-author-friendly.

Langfuse spans are attached in Phase 10; the service is structured to make
that a thin wrapper around ``invoke``.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy

from app.core.errors import ConfigurationError, WorkflowServerError
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry
from app.llm.types import (
    CAPABILITY_CHAT,
    CAPABILITY_JSON_MODE,
    CAPABILITY_STREAMING,
    CAPABILITY_TOOLS,
    LLMRequest,
    LLMResponse,
    Message,
)

log = logging.getLogger(__name__)


_JSON_FALLBACK_INSTRUCTION = (
    "Respond with a single valid JSON document and nothing else. "
    "Do not wrap it in code fences or include any commentary."
)


class LLMService:
    def __init__(
        self, *, models: ModelRegistry, providers: ProviderRegistry
    ) -> None:
        self._models = models
        self._providers = providers

    @property
    def models(self) -> ModelRegistry:
        return self._models

    @property
    def providers(self) -> ProviderRegistry:
        return self._providers

    async def invoke(self, model_id: str, request: LLMRequest) -> LLMResponse:
        entry = self._models.get(model_id)
        self._validate_capabilities(entry, request)
        provider = self._providers.get(entry.provider)

        actual_request = request
        needs_json_fallback = self._needs_json_fallback(entry, request)
        if needs_json_fallback:
            actual_request = _inject_json_instruction(request)

        response = await provider.chat(actual_request, model=entry.model)

        # Re-tag with the user-facing model_id so traces / events carry the
        # registry id (e.g., "gpt-4o") rather than the provider's internal
        # string (e.g., "gpt-4o-2024-08-06"). This keeps workflow JSON and
        # observability data aligned.
        response.model = model_id

        if needs_json_fallback:
            response = _attach_parsed_json(response)

        return response

    # ----- internals ------------------------------------------------------

    def _validate_capabilities(self, entry: ModelEntry, request: LLMRequest) -> None:
        caps = entry.capabilities
        if CAPABILITY_CHAT not in caps:
            raise ConfigurationError(
                f"model {entry.id!r} does not declare 'chat' capability",
                details={"model_id": entry.id, "capabilities": sorted(caps)},
            )
        if request.tools and CAPABILITY_TOOLS not in caps:
            raise ConfigurationError(
                f"model {entry.id!r} does not support tools but request includes {len(request.tools)} tool(s)",
                details={"model_id": entry.id, "capabilities": sorted(caps)},
            )
        if request.stream and CAPABILITY_STREAMING not in caps:
            raise ConfigurationError(
                f"model {entry.id!r} does not support streaming",
                details={"model_id": entry.id, "capabilities": sorted(caps)},
            )

    def _needs_json_fallback(self, entry: ModelEntry, request: LLMRequest) -> bool:
        if request.response_format != "json":
            return False
        return CAPABILITY_JSON_MODE not in entry.capabilities


# ----- helpers (module-level for testability) -----------------------------


def _inject_json_instruction(request: LLMRequest) -> LLMRequest:
    """Return a copy of ``request`` with a JSON-only instruction appended to
    the system message (or prepended as one) and ``response_format`` set to
    ``text`` so the provider isn't asked for native JSON it can't guarantee.
    """
    cloned = deepcopy(request)
    msgs = list(cloned.messages)
    if msgs and msgs[0].role == "system":
        existing = msgs[0].content or ""
        suffix = "" if existing.endswith("\n") else "\n"
        msgs[0] = Message(role="system", content=f"{existing}{suffix}{_JSON_FALLBACK_INSTRUCTION}")
    else:
        msgs.insert(0, Message(role="system", content=_JSON_FALLBACK_INSTRUCTION))
    cloned.messages = msgs
    cloned.response_format = "text"
    return cloned


def _attach_parsed_json(response: LLMResponse) -> LLMResponse:
    """After fallback, attempt to parse ``response.content`` as JSON.

    On success we populate ``parsed_json``. On failure we leave it ``None``
    and don't raise — the caller can decide whether that's fatal.
    """
    if response.parsed_json is not None or response.content is None:
        return response
    text = _strip_code_fences(response.content.strip())
    try:
        response.parsed_json = json.loads(text)
    except json.JSONDecodeError:
        log.info("JSON-fallback could not parse content; leaving parsed_json=None")
    return response


def _strip_code_fences(s: str) -> str:
    """If the model wrapped JSON in ```json ... ``` despite instructions, peel it."""
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) < 2:
        return s
    # Drop the opening fence (possibly with language tag) and any closing fence.
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
