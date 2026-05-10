"""Top-level workflow definition (the parsed, normalized form of frontend JSON)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.edge import Edge
from app.schemas.node import Node


class WorkflowInterface(BaseModel):
    """Optional interface block describing inputs/outputs of the workflow.

    The frontend's shape is loose — sometimes a list of variable specs, sometimes
    a dict. We accept both and let downstream code interpret them.
    """

    model_config = ConfigDict(extra="allow")

    inputs: Any | None = None
    outputs: Any | None = None


class WorkflowDefinition(BaseModel):
    """A normalized, runtime-ready workflow definition."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    workflow_id: str
    workflow_version: int = 1
    name: str | None = None
    description: str | None = None

    nodes: list[Node]
    edges: list[Edge]

    interface: WorkflowInterface | None = None

    # Original payload kept for traceability (e.g. for debugging or hashing).
    # Excluded from API serialization to avoid duplication.
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    def summary(self) -> "WorkflowSummary":
        return WorkflowSummary(
            workflow_id=self.workflow_id,
            workflow_version=self.workflow_version,
            name=self.name,
            description=self.description,
            node_count=len(self.nodes),
            edge_count=len(self.edges),
        )


class WorkflowSummary(BaseModel):
    workflow_id: str
    workflow_version: int
    name: str | None = None
    description: str | None = None
    node_count: int
    edge_count: int
