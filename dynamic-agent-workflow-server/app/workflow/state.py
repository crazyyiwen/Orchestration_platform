"""State merge semantics and path utilities (spec §5 + plan tricky-detail #6).

Codified merge rule for ``state_updates`` returned by node executors:

* dict + dict → **deep merge** (deeper key wins on scalar conflicts)
* list  + any → **replace** (no concatenation)
* scalar     → **overwrite**

These functions never mutate the input ``state``; callers must use the returned
value. ``set_path_inplace`` is the one exception — it mutates by design and is
named accordingly.
"""
from __future__ import annotations

import copy
from typing import Any


def merge_updates(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``updates`` into ``state``. Returns a new dict.

    Top-level shape: dicts deep-merge recursively, everything else overwrites
    (lists are replaced, not concatenated; scalars overwrite).
    """
    if not isinstance(state, dict) or not isinstance(updates, dict):
        raise TypeError("merge_updates expects two dicts")

    out: dict[str, Any] = copy.deepcopy(state)
    for key, val in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = merge_updates(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def get_path(target: Any, path: str, default: Any = None) -> Any:
    """Walk a dotted path through nested dicts. Returns ``default`` on miss.

    Empty path returns ``target`` itself. Non-dict intermediates return default.
    """
    if path == "":
        return target
    cur: Any = target
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def set_path_inplace(target: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Set ``value`` at dotted path, creating intermediate dicts as needed.

    **Mutates** ``target``. If an intermediate slot exists but isn't a dict,
    it's overwritten with a new dict.
    """
    if not path:
        raise ValueError("set_path_inplace: path must be non-empty")
    keys = path.split(".")
    cur: dict[str, Any] = target
    for key in keys[:-1]:
        if not isinstance(cur.get(key), dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value
    return target


def set_path(target: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Pure variant of ``set_path_inplace``: returns a new dict, leaves input alone."""
    out = copy.deepcopy(target)
    set_path_inplace(out, path, value)
    return out


def has_path(target: Any, path: str) -> bool:
    """True iff a value (including None) is reachable at the dotted path."""
    sentinel = object()
    return get_path(target, path, default=sentinel) is not sentinel


def empty_runtime_state(
    *,
    run_id: str,
    workflow_id: str,
    workflow_version: int,
) -> dict[str, Any]:
    """Build the canonical empty runtime state used by Phase 8/9.

    The shape mirrors spec §4 — Phase 8 will register this as the LangGraph
    StateGraph's initial state and Phase 9's run_manager will mutate it
    through the wrapper.
    """
    return {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow_version": workflow_version,
        "status": "pending",
        "current_node_id": None,
        "current_node_name": None,
        "variables": {"system": {}, "runtime": {}, "nodes": {}},
        "messages": [],
        "events": [],
        "final_output": None,
        "error": None,
        "pause": None,
        "step_count": 0,
        "_next_handle": None,
        "_resume_input": None,
        "langfuse_trace_id": None,
    }
