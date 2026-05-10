"""In-process compile cache keyed by ``definition_hash``.

Compiling a graph involves wiring up wrappers, edges, and routers — non-trivial
work we don't want to repeat per run. The key is the *content* hash so cosmetic
edits in the frontend don't bust the cache, and version bumps with no semantic
change still hit cache.
"""
from __future__ import annotations

import logging
from typing import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.langgraph_runtime.graph_builder import compile_workflow
from app.schemas.workflow import WorkflowDefinition
from app.workflow.hashing import definition_hash
from app.workflow.node_executors.base import ExecutionContext

log = logging.getLogger(__name__)


class CompileCache:
    def __init__(self) -> None:
        self._by_hash: dict[str, object] = {}

    def get_or_compile(
        self,
        definition: WorkflowDefinition,
        *,
        context_factory: Callable[[], ExecutionContext],
        checkpointer: BaseCheckpointSaver | None = None,
        on_node_event=None,
    ):
        key = definition_hash(definition)
        if key in self._by_hash:
            log.debug("compile cache hit %s for %s", key[:8], definition.workflow_id)
            return self._by_hash[key]
        compiled = compile_workflow(
            definition,
            context_factory=context_factory,
            checkpointer=checkpointer,
            on_node_event=on_node_event,
        )
        self._by_hash[key] = compiled
        log.info(
            "compile cache miss %s — compiled workflow=%s nodes=%d",
            key[:8],
            definition.workflow_id,
            len(definition.nodes),
        )
        return compiled

    def invalidate(self, definition: WorkflowDefinition) -> bool:
        return self._by_hash.pop(definition_hash(definition), None) is not None

    def clear(self) -> None:
        self._by_hash.clear()
