"""Tool registry — keyed lookup of available tools (spec §9)."""
from __future__ import annotations

from typing import Iterable

from app.core.errors import ConfigurationError
from app.llm.types import ToolSpec
from app.tools.base import BaseTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for t in tools or ():
            self.register(t)

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ConfigurationError("tool name is required")
        if tool.name in self._tools:
            raise ConfigurationError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise ConfigurationError(
                f"unknown tool {name!r}", details={"registered": sorted(self._tools)}
            ) from e

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def specs_for(self, names: Iterable[str]) -> list[ToolSpec]:
        out = []
        for n in names:
            if self.has(n):
                out.append(self.get(n).to_spec())
        return out
