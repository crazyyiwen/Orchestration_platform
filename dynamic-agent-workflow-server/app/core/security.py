from __future__ import annotations

import re
from typing import Any

from app.core.config import Settings

# Patterns of values we never want to leak in error responses or logs aimed at clients.
_SECRET_KEY_NAMES = re.compile(
    r"(?i)(api[_-]?key|secret|token|authorization|password|cookie|bearer)"
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),     # OpenAI-style
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), # Anthropic-style
    re.compile(r"hf_[A-Za-z0-9]{16,}"),        # HuggingFace
]


def _redact_value(v: Any) -> Any:
    if isinstance(v, str):
        s = v
        for pat in _SECRET_VALUE_PATTERNS:
            s = pat.sub("***", s)
        return s
    return v


def _redact_mapping(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(k, str) and _SECRET_KEY_NAMES.search(k):
            out[k] = "***"
        elif isinstance(v, dict):
            out[k] = _redact_mapping(v)
        elif isinstance(v, list):
            out[k] = [_redact_mapping(x) if isinstance(x, dict) else _redact_value(x) for x in v]
        else:
            out[k] = _redact_value(v)
    return out


def sanitize_error(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-like values from an error payload before returning it to clients."""
    return _redact_mapping(payload)


def known_provider_keys(settings: Settings) -> dict[str, bool]:
    """Boolean view of which provider credentials are configured. Never returns the values."""
    return {
        "openai": bool(settings.OPENAI_API_KEY),
        "anthropic": bool(settings.ANTHROPIC_API_KEY),
        "huggingface": bool(settings.HUGGINGFACE_API_KEY),
        "langfuse": bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY),
    }
