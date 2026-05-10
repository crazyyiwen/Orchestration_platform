"""Base abstractions for the Tool Registry (spec §9).

Tools wrap arbitrary capabilities the Agent node can invoke. Each tool exposes
an OpenAI-style function schema so the LLM can choose to call it. Implementations
must be safe to execute server-side — no arbitrary Python evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.llm.types import ToolSpec


class BaseTool(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the tool with parsed arguments. Returns a JSON-serializable dict."""

    def to_spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)
