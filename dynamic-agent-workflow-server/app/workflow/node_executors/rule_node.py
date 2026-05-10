"""Rule (branch) node — evaluates conditions and returns a branch handle (spec §7).

Config shape (canonical):

    {
      "branches": [
        {
          "handle": "case_1",
          "logic": "AND",          # or "OR"; default AND
          "conditions": [
            {"field": "{{system.userQuery}}", "operator": "contains", "value": "weather"}
          ]
        },
        ...
      ],
      "default_handle": "else"
    }

Returns the *first* matching branch handle, or ``default_handle`` if none match.
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.variables import VariableResolver


_OPERATORS = {
    "equals", "==", "eq",
    "not_equals", "!=", "ne",
    "contains", "not_contains",
    "greater_than", ">", "gt",
    "less_than", "<", "lt",
    "greater_or_equal", ">=", "ge",
    "less_or_equal", "<=", "le",
    "exists", "not_exists",
    "empty", "not_empty",
    "is empty", "is not empty",
    "regex",
    "in", "not_in",
}


@register("rule")
class RuleNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        resolver = VariableResolver(state.get("variables", {}))
        branches = node.config.get("branches") or []
        default_handle = node.config.get("default_handle") or "else"
        matched: dict[str, Any] = {}

        for branch in branches:
            handle = branch.get("handle")
            if not handle:
                continue
            conditions = branch.get("conditions") or []
            logic = (branch.get("logic") or branch.get("logicType") or "AND").upper()
            evaluator = all if logic == "AND" else any
            if not conditions:
                # Empty conditions evaluate truthy in AND, falsy in OR.
                if logic == "AND":
                    matched = {"handle": handle, "branch": branch}
                    break
                continue
            if evaluator(_evaluate_condition(c, resolver) for c in conditions):
                matched = {"handle": handle, "branch": branch}
                break

        chosen = matched.get("handle") or default_handle
        return NodeExecutionResult(
            status="success",
            output={"chosen_handle": chosen, "matched_branch": matched.get("branch")},
            next_handle=chosen,
        )


def _evaluate_condition(cond: dict[str, Any], resolver: VariableResolver) -> bool:
    raw_field = cond.get("field")
    operator = (cond.get("operator") or "equals").strip()
    raw_value = cond.get("value")
    field_val = resolver.resolve_string(raw_field) if isinstance(raw_field, str) else raw_field
    cmp_val = resolver.resolve_string(raw_value) if isinstance(raw_value, str) else raw_value
    return _apply_operator(field_val, operator, cmp_val)


def _apply_operator(left: Any, op: str, right: Any) -> bool:  # noqa: PLR0911 PLR0912
    op_norm = op.lower().replace(" ", "_")
    try:
        if op_norm in {"equals", "==", "eq"}:
            return left == right
        if op_norm in {"not_equals", "!=", "ne"}:
            return left != right
        if op_norm == "contains":
            return _contains(left, right)
        if op_norm == "not_contains":
            return not _contains(left, right)
        if op_norm in {"greater_than", ">", "gt"}:
            return _coerce_num(left) > _coerce_num(right)
        if op_norm in {"less_than", "<", "lt"}:
            return _coerce_num(left) < _coerce_num(right)
        if op_norm in {"greater_or_equal", ">=", "ge"}:
            return _coerce_num(left) >= _coerce_num(right)
        if op_norm in {"less_or_equal", "<=", "le"}:
            return _coerce_num(left) <= _coerce_num(right)
        if op_norm in {"exists"}:
            return left is not None
        if op_norm in {"not_exists"}:
            return left is None
        if op_norm in {"empty", "is_empty"}:
            return _is_empty(left)
        if op_norm in {"not_empty", "is_not_empty"}:
            return not _is_empty(left)
        if op_norm == "regex":
            if not isinstance(right, str) or left is None:
                return False
            return re.search(right, str(left)) is not None
        if op_norm == "in":
            return left in (right or [])
        if op_norm == "not_in":
            return left not in (right or [])
    except (TypeError, ValueError):
        return False
    # Unknown operator: treat as no-match (validator can flag separately).
    return False


def _contains(haystack: Any, needle: Any) -> bool:
    if haystack is None:
        return False
    if isinstance(haystack, (list, tuple, set)):
        return needle in haystack
    return str(needle) in str(haystack)


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (str, list, tuple, dict, set)):
        return len(v) == 0
    return False


def _coerce_num(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v))
