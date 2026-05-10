"""Canonical MongoDB collection names. The single source of truth for them."""
from __future__ import annotations

from typing import Final

WORKFLOW_RUNS: Final[str] = "workflow_runs"
WORKFLOW_RUN_EVENTS: Final[str] = "workflow_run_events"
WORKFLOW_CHECKPOINTS: Final[str] = "workflow_checkpoints"
WORKFLOW_COMPILED_CACHE: Final[str] = "workflow_compiled_cache"

ALL: Final[tuple[str, ...]] = (
    WORKFLOW_RUNS,
    WORKFLOW_RUN_EVENTS,
    WORKFLOW_CHECKPOINTS,
    WORKFLOW_COMPILED_CACHE,
)
