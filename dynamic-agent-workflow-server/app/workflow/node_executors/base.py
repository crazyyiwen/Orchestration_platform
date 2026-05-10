"""Node executor interface, registry, and shared types (spec §6).

Each executor:
  * Declares its ``node_type`` (matched against ``Node.type``).
  * Implements ``execute(node, state, ctx)``.
  * Optionally implements ``validate_config(config)`` for static checks the
    Phase 4 validator can run.

The registry is populated via the ``@register("type_name")`` decorator at
import time. ``app.workflow.node_executors.__init__`` imports every executor
module so registration is automatic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Iterable, Literal

import httpx

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.llm.service import LLMService
from app.schemas.node import Node
from app.schemas.validation import ValidationIssue
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.runtime.event_bus import EventBus


# ----- result -------------------------------------------------------------


@dataclass
class NodeExecutionResult:
    """The contract every executor returns. The compiler's wrapper consumes it."""

    status: Literal["success", "failed", "paused", "skipped"] = "success"
    output: Any = None
    next_handle: str | None = "out"
    error: dict[str, Any] | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
    pause_payload: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


# ----- execution context --------------------------------------------------


# Type alias for the sub-flow launcher injected by Phase 9's run_manager.
SubFlowLauncher = Callable[[str, dict[str, Any], int, str], Awaitable[dict[str, Any]]]


@dataclass
class ExecutionContext:
    run_id: str
    workflow_id: str
    workflow_version: int
    settings: Settings
    llm_service: LLMService
    tool_registry: ToolRegistry
    http_client: httpx.AsyncClient
    depth: int = 0
    parent_run_id: str | None = None
    sub_flow_launcher: SubFlowLauncher | None = None
    event_bus: "EventBus | None" = None


# ----- base ABC -----------------------------------------------------------


class BaseNodeExecutor(ABC):
    node_type: ClassVar[str] = ""

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[ValidationIssue]:
        """Optional static-config check. Default: accept anything."""
        return []

    @abstractmethod
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        """Run the node and return a normalized result."""


# ----- registry singleton -------------------------------------------------

_REGISTRY: dict[str, type[BaseNodeExecutor]] = {}


def register(node_type: str) -> Callable[[type[BaseNodeExecutor]], type[BaseNodeExecutor]]:
    def _decorate(cls: type[BaseNodeExecutor]) -> type[BaseNodeExecutor]:
        if node_type in _REGISTRY:
            raise ConfigurationError(
                f"executor for node_type {node_type!r} already registered"
            )
        cls.node_type = node_type
        _REGISTRY[node_type] = cls
        return cls

    return _decorate


def get_executor(node_type: str) -> BaseNodeExecutor:
    cls = _REGISTRY.get(node_type)
    if cls is None:
        raise ConfigurationError(
            f"no executor registered for node_type {node_type!r}",
            details={"registered": sorted(_REGISTRY)},
        )
    return cls()


def registered_types() -> set[str]:
    return set(_REGISTRY)


def clear_registry_for_tests() -> None:
    _REGISTRY.clear()
