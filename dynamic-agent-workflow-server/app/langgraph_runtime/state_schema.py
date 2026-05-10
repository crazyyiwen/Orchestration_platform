"""LangGraph state schema (TypedDict).

Per the plan's tricky-detail #1, accumulator channels need explicit reducers
or last-write-wins overwrites them when running parallel paths. We use
``operator.add`` for ``messages`` and ``events`` so future parallel-branch
work composes cleanly. Everything else uses LangGraph's default
last-write-wins behavior, which is what we want for state replacement.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class WorkflowState(TypedDict, total=False):
    run_id: str
    workflow_id: str
    workflow_version: int
    status: Literal[
        "pending", "running", "paused", "completed", "failed", "cancelled"
    ]
    current_node_id: str | None
    current_node_name: str | None
    variables: dict[str, Any]
    messages: Annotated[list, operator.add]
    events: Annotated[list, operator.add]
    final_output: Any
    error: dict[str, Any] | None
    pause: dict[str, Any] | None
    step_count: int
    _next_handle: str | None
    _resume_input: Any
    langfuse_trace_id: str | None
