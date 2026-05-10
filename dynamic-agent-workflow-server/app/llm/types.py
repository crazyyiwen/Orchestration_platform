"""LLM request / response DTOs and capability constants.

Provider-agnostic. Each :class:`BaseLLMProvider` translates ``LLMRequest`` →
its native API and translates the response back into :class:`LLMResponse`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ----- capabilities -------------------------------------------------------

# Capability strings used by ModelEntry.capabilities. Free-form (a registry
# entry can declare any capability) but these are the well-known ones the
# service checks for.
CAPABILITY_CHAT = "chat"
CAPABILITY_JSON_MODE = "json_mode"
CAPABILITY_TOOLS = "tools"
CAPABILITY_STREAMING = "streaming"

KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {CAPABILITY_CHAT, CAPABILITY_JSON_MODE, CAPABILITY_TOOLS, CAPABILITY_STREAMING}
)


# ----- messages -----------------------------------------------------------


class ToolCall(BaseModel):
    """A single function/tool invocation requested by the model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


# ----- tool schema --------------------------------------------------------


class ToolSpec(BaseModel):
    """OpenAI-style function/tool description."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


# ----- request / response -------------------------------------------------


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[Message]
    response_format: Literal["text", "json"] = "text"
    json_schema: dict[str, Any] | None = None
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict[str, Any] | None = None  # "auto" | "none" | {tool: name}
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    model: str
    content: str | None
    parsed_json: Any | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
