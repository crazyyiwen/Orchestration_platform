"""Deterministic mock tools for tests + offline dev."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool


class EchoTool(BaseTool):
    """Returns its input verbatim — useful for tool-call wiring tests."""

    name = "echo"
    description = "Echo the input back."
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"echoed": args.get("input", "")}


class StaticAnswerTool(BaseTool):
    """A tool that always returns a fixed answer — handy for agent-loop tests."""

    name = "static_answer"
    description = "Return a canned answer."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, answer: str = "42") -> None:
        self._answer = answer

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"answer": self._answer}
