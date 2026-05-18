"""Dynamic LangGraph compiler.

Turns a :class:`WorkflowDefinition` into an executable ``CompiledGraph``. The
core invariants:

  * **One LangGraph node per workflow node**, all bound to the same generic
    wrapper. Zero ``if node.type == ...`` branches in the runtime.
  * Routing comes from ``state._next_handle`` matched against each edge's
    ``sourceHandle`` — no hard-coded handles, no per-workflow routing.
  * Output nodes are terminal (edge to END).
  * Pause-capable nodes return ``status="paused"`` and the wrapper raises
    LangGraph's ``GraphInterrupt`` (via ``interrupt(...)``) so a checkpointer
    can persist state and the run halts cleanly.

The wrapper applies state updates with the documented merge semantics
(:func:`merge_updates`) and writes ``variables.nodes[<name>].result`` so
later nodes can reference ``{{nodes.<name>.result.*}}``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.core.errors import ExecutionLimitExceeded
from app.langgraph_runtime.dynamic_router import make_router
from app.langgraph_runtime.state_schema import WorkflowState
from app.schemas.node import Node
from app.schemas.workflow import WorkflowDefinition
from app.workflow.graph_utils import adjacency, find_start_node
from app.workflow.node_executors.base import (
    ExecutionContext,
    NodeExecutionResult,
    get_executor,
)
from app.workflow.state import get_path, merge_updates, set_path_inplace
from app.workflow.variables import VariableResolver

log = logging.getLogger(__name__)


# ----- public API ---------------------------------------------------------


def compile_workflow(
    definition: WorkflowDefinition,
    *,
    context_factory: Callable[[], ExecutionContext],
    checkpointer: BaseCheckpointSaver | None = None,
    on_node_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
):
    """Build a CompiledGraph from a workflow definition.

    Parameters
    ----------
    definition
        Already-validated workflow definition.
    context_factory
        Callable returning a fresh :class:`ExecutionContext` per node call —
        Phase 9's run_manager constructs this with run-scoped state.
    checkpointer
        LangGraph checkpointer. Defaults to ``InMemorySaver`` so pause/resume
        works within one process. Phase-tickets cover a Mongo-backed
        ``BaseCheckpointSaver`` for cross-process resume.
    on_node_event
        Optional async callback invoked for each event the wrapper emits.
        The run_manager wires this to ``EventBus.publish`` + repository writes.
    """
    g = StateGraph(WorkflowState)
    adj = adjacency(definition)

    for node in definition.nodes:
        g.add_node(
            node.id,
            _make_node_runner(
                node,
                context_factory=context_factory,
                on_node_event=on_node_event,
            ),
        )

    start = find_start_node(definition)
    if start is None:
        # The validator should have caught this, but be defensive.
        from app.core.errors import CompilationError

        raise CompilationError("workflow has no detectable start node")
    g.add_edge(START, start.id)

    for node in definition.nodes:
        outs = adj.get(node.id, [])
        if node.type == "output" or not outs:
            g.add_edge(node.id, END)
            continue

        # Group outgoing edges by sourceHandle (None → "out").
        handle_to_target: dict[str | None, str] = {}
        for e in outs:
            key = e.sourceHandle  # None preserved; router normalizes
            # If an agent node has multiple edges with the same handoff handle,
            # the first one wins for compile-time mapping; the agent executor
            # should have returned a more specific next_handle. The validator
            # already warned via ambiguous_handle_routing.
            if key in handle_to_target:
                continue
            handle_to_target[key] = e.target

        if len(handle_to_target) == 1 and (None in handle_to_target or "out" in handle_to_target):
            # Plain pass-through.
            target = next(iter(handle_to_target.values()))
            g.add_edge(node.id, target)
        else:
            # No `mapping` arg: the router returns target node ids (or END)
            # directly. We pass the candidate set so LangGraph knows which
            # downstream nodes this branch can reach — END is included for
            # the unmapped-handle fallback case.
            candidates = list(set(handle_to_target.values()))
            candidates.append(END)
            g.add_conditional_edges(
                node.id,
                make_router(source_node_id=node.id, handle_to_target=handle_to_target),
                candidates,
            )

    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ----- generic node wrapper ----------------------------------------------


def _make_node_runner(
    node: Node,
    *,
    context_factory: Callable[[], ExecutionContext],
    on_node_event: Callable[[dict[str, Any]], Awaitable[None]] | None,
):
    async def runner(state: dict[str, Any]) -> dict[str, Any]:
        ctx = context_factory()

        # Step-limit guard (plan tricky-detail).
        step_count = int(state.get("step_count") or 0) + 1
        if step_count > ctx.settings.MAX_WORKFLOW_STEPS:
            raise ExecutionLimitExceeded(
                f"workflow exceeded MAX_WORKFLOW_STEPS={ctx.settings.MAX_WORKFLOW_STEPS}"
            )

        executor = get_executor(node.type)
        try:
            result = await asyncio.wait_for(
                executor.execute(node, state, ctx),
                timeout=ctx.settings.NODE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await _emit(
                on_node_event,
                _make_event(node, "node_failed", {"reason": "timeout"}, state),
            )
            return {
                "status": "failed",
                "error": {"message": f"node {node.id!r} timed out", "node_id": node.id},
                "current_node_id": node.id,
                "current_node_name": node.name,
                "step_count": step_count,
                "_next_handle": "error",
            }

        # Build the state delta. The reducers on `messages` and `events` mean
        # ``operator.add`` lists merge cleanly when LangGraph applies them.
        delta: dict[str, Any] = {
            "current_node_id": node.id,
            "current_node_name": node.name,
            "step_count": step_count,
        }

        if result.state_updates:
            delta.update(_merge_state_updates(state, result.state_updates))

        # Always record the node's result under variables.nodes.<name>.result
        # so downstream {{nodes.<name>.result.*}} works without each executor
        # having to do it itself.
        new_vars = delta.get("variables")
        if new_vars is None:
            new_vars = _deepcopy(state.get("variables", {}))
        set_path_inplace(new_vars, f"nodes.{node.name}.result", _to_json_safe(result.output))

        # Generic post-execution config.stateUpdates pass — mirrors the
        # frontend executor's applyConfigStateUpdates. EVERY node may carry a
        # "State Update" list ({key, value, operation}); each row resolves
        # against the *current* variables (which now include this node's
        # result, written just above) and writes to `key`. Honors
        # `stateUpdatesRunOnlyWhen` (skip the list if it resolves falsy).
        if result.status not in ("paused", "failed"):
            _apply_config_state_updates(node, new_vars)

        delta["variables"] = new_vars

        delta["_next_handle"] = result.next_handle

        # Append a node-level event and any executor-emitted events.
        events_out: list[dict[str, Any]] = []
        if result.status == "success":
            events_out.append(_make_event(node, "node_completed", {"output": _to_json_safe(result.output)}, state))
        elif result.status == "failed":
            events_out.append(_make_event(node, "node_failed", {"error": result.error}, state))
            if not delta.get("error"):
                delta["error"] = result.error
        elif result.status == "skipped":
            events_out.append(_make_event(node, "node_skipped", {}, state))
        for ev in result.events or []:
            events_out.append(_decorate_event(ev, node, state))

        for ev in events_out:
            await _emit(on_node_event, ev)
        delta["events"] = events_out

        if result.status == "paused":
            # Persist pause payload + status before we hand control to LangGraph.
            delta["status"] = "paused"
            delta["pause"] = result.pause_payload or {}
            await _emit(on_node_event, _make_event(node, "run_paused", delta["pause"], state))
            # ``interrupt(...)`` raises GraphInterrupt; LangGraph catches it,
            # snapshots state via the checkpointer, and exits ``ainvoke``.
            interrupt(delta["pause"])

        return delta

    return runner


# ----- helpers ------------------------------------------------------------


def _merge_state_updates(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply executor-supplied state updates with the documented semantics.

    Returns a dict suitable for direct inclusion in the LangGraph state delta.
    `variables` is deep-merged via :func:`merge_updates`; everything else
    overwrites by key (LangGraph's default channel behavior).
    """
    out: dict[str, Any] = {}
    for k, v in updates.items():
        if k == "variables" and isinstance(v, dict):
            out["variables"] = merge_updates(state.get("variables") or {}, v)
        else:
            out[k] = v
    return out


def _apply_config_state_updates(node: Node, variables: dict[str, Any]) -> None:
    """Apply a node's declarative ``config.stateUpdates`` list in place.

    Mirrors the frontend executor's ``applyConfigStateUpdates``. Every
    non-system node may carry a "State Update" section. Each row is
    ``{key, value, operation}``:

      * ``value`` is resolved against the *current* ``variables`` (which by
        now include this node's own ``nodes.<name>.result``), so a row like
        ``{key: "flow.intentResult", value: "{{nodes.llm.result}}"}`` works.
      * ``key`` is a dotted path written into ``variables``.
      * ``operation`` supports set / append / merge / increment / remove
        (same semantics as the variable_update node).

    The whole list is skipped when ``config.stateUpdatesRunOnlyWhen`` is
    present and resolves falsy. Both ``stateUpdates`` and ``variableUpdates``
    keys are honored (the Start node uses the latter).
    """
    cfg = node.config if isinstance(node.config, dict) else {}
    rows = cfg.get("stateUpdates") or cfg.get("variableUpdates") or []
    if not isinstance(rows, list) or not rows:
        return

    resolver = VariableResolver(variables)

    gate = str(cfg.get("stateUpdatesRunOnlyWhen") or "").strip()
    if gate:
        if not resolver.resolve_value(gate):
            return

    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key") or row.get("fieldName") or row.get("path")
        if not key:
            continue
        key = str(key).strip().strip("{}").strip()
        op = (row.get("operation") or "set").lower()
        value = resolver.resolve_value(row.get("value"))

        if op == "set":
            set_path_inplace(variables, key, value)
        elif op == "append":
            existing = get_path(variables, key, default=None)
            if not isinstance(existing, list):
                existing = []
            existing = list(existing)
            existing.append(value)
            set_path_inplace(variables, key, existing)
        elif op == "merge":
            existing = get_path(variables, key, default=None)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged = {**existing, **value}
            else:
                merged = value if isinstance(value, dict) else {}
            set_path_inplace(variables, key, merged)
        elif op == "increment":
            existing = get_path(variables, key, default=0) or 0
            try:
                set_path_inplace(variables, key, existing + (value or 1))
            except TypeError:
                set_path_inplace(variables, key, value)
        elif op == "remove":
            _delete_path(variables, key)
        else:
            set_path_inplace(variables, key, value)


def _delete_path(target: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    cur: Any = target
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _make_event(
    node: Node, event_type: str, payload: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": event_type,
        "node_id": node.id,
        "node_name": node.name,
        "node_type": node.type,
        "payload": payload,
    }


def _decorate_event(ev: dict[str, Any], node: Node, state: dict[str, Any]) -> dict[str, Any]:
    out = dict(ev)
    out.setdefault("node_id", node.id)
    out.setdefault("node_name", node.name)
    out.setdefault("node_type", node.type)
    return out


async def _emit(
    callback: Callable[[dict[str, Any]], Awaitable[None]] | None, event: dict[str, Any]
) -> None:
    if callback is None:
        return
    try:
        await callback(event)
    except Exception:  # noqa: BLE001
        log.exception("on_node_event callback failed")


def _to_json_safe(v: Any) -> Any:
    """Best-effort JSON normalization for state persistence.

    Pydantic models become dicts; everything else passes through.
    """
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        return v
    if isinstance(v, BaseModel):
        return v.model_dump()
    if isinstance(v, dict):
        return {k: _to_json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_json_safe(x) for x in v]
    return v


def _deepcopy(v: Any) -> Any:
    import copy

    return copy.deepcopy(v)
