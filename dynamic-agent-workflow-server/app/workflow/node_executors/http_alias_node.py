"""http node — alias for ``http_request`` that handles the React Flow shape.

The frontend's HTTP node uses list-of-pairs for headers/query/body
(``[{name, value, type, required}, ...]``). This executor converts those into
the dict form our :class:`HttpRequestNodeExecutor` expects, then delegates.
"""
from __future__ import annotations

import copy
from typing import Any

from app.schemas.node import Node
from app.workflow.node_executors.base import (
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    register,
)
from app.workflow.node_executors.http_request_node import HttpRequestNodeExecutor


@register("http")
class HttpAliasNodeExecutor(BaseNodeExecutor):
    async def execute(
        self, node: Node, state: dict[str, Any], ctx: ExecutionContext
    ) -> NodeExecutionResult:
        normalized = copy.deepcopy(node)
        cfg = normalized.config or {}
        cfg["headers"] = _pairs_to_dict(cfg.get("headers"))
        cfg["query"] = _pairs_to_dict(cfg.get("query"))
        body = cfg.get("body")
        if isinstance(body, list):
            cfg["body"] = _pairs_to_dict(body)
        normalized.config = cfg
        return await HttpRequestNodeExecutor().execute(normalized, state, ctx)


def _pairs_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {item["name"]: item.get("value") for item in value if isinstance(item, dict) and "name" in item}
    return {}
