"""Generic Node model.

Permissive on purpose: real frontend output carries UI fields (position, data,
properties, variableUpdates, width, height, ...) that the runtime ignores but
must round-trip without rejection. Type values are not enum-constrained at the
schema layer — the validator (Phase 4) and executor registry (Phase 7) decide
which type names are runnable.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Node(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    type: str
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None

    @property
    def extras(self) -> dict[str, Any]:
        """Frontend-extra fields that flowed in via extras='allow'."""
        return self.__pydantic_extra__ or {}
