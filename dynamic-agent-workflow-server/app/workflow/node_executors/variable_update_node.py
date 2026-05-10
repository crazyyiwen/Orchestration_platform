"""Variable Update node — applies set/append/merge/increment/remove ops (spec §7).

Config shape::

    {
      "updates": [
        {"path": "system.userQuery", "operation": "set",       "value": "{{...}}"},
        {"path": "thread.messages", "operation": "append",    "value": "{{...}}"},
        {"path": "flow.config",     "operation": "merge",     "value": {"k": 1}},
        {"path": "flow.counter",    "operation": "increment", "value": 1},
        {"path": "flow.tmp",        "operation": "remove"}
      ]
    }
"""
from __future__ import annotations

from typing import Any

from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.state import get_path, set_path_inplace
from app.workflow.variables import VariableResolver


@register("variable_update")
class VariableUpdateNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        resolver = VariableResolver(state.get("variables", {}))
        updates = node.config.get("updates") or node.config.get("variableUpdates") or []
        # We mutate a working copy of variables and return it as a state_update.
        new_vars = _deepcopy(state.get("variables", {}))
        applied: list[dict[str, Any]] = []

        for u in updates:
            path = u.get("path") or u.get("fieldName")
            op = (u.get("operation") or "set").lower()
            value = resolver.resolve_value(u.get("value"))
            if not path:
                continue
            # Tolerate frontends that wrap path in {{ }} (treat literally).
            path = path.strip("{} ").strip()
            if op == "set":
                set_path_inplace(new_vars, path, value)
            elif op == "append":
                existing = get_path(new_vars, path, default=None)
                if not isinstance(existing, list):
                    existing = []
                existing.append(value)
                set_path_inplace(new_vars, path, existing)
            elif op == "merge":
                existing = get_path(new_vars, path, default=None)
                if isinstance(existing, dict) and isinstance(value, dict):
                    existing = {**existing, **value}
                else:
                    existing = value if isinstance(value, dict) else {}
                set_path_inplace(new_vars, path, existing)
            elif op == "increment":
                existing = get_path(new_vars, path, default=0) or 0
                set_path_inplace(new_vars, path, existing + (value or 1))
            elif op == "remove":
                _delete_path(new_vars, path)
            applied.append({"path": path, "operation": op})

        return NodeExecutionResult(
            status="success",
            output={"updates_applied": applied},
            next_handle="out",
            state_updates={"variables": new_vars},
        )


def _deepcopy(v: Any) -> Any:
    import copy

    return copy.deepcopy(v)


def _delete_path(target: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    cur = target
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
