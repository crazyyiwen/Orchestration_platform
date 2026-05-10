"""Dynamic router factory — translates ``state._next_handle`` → target node id.

Per the plan, the router falls back through ``"out"`` → ``"else"`` → END if a
specific handle isn't mapped, so unknown handles end the run with a clean
event rather than raising. The compiler builds one router per non-trivial
branch node.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END

log = logging.getLogger(__name__)


def make_router(
    *,
    source_node_id: str,
    handle_to_target: dict[str | None, str],
) -> Callable[[dict[str, Any]], str]:
    """Build a router function for ``add_conditional_edges``.

    ``handle_to_target`` maps each outgoing edge's ``sourceHandle`` (None
    counted as ``"out"``) to the target node id. The router reads
    ``state._next_handle`` and looks it up.
    """
    # Normalize None → "out" for lookup convenience.
    mapping: dict[str, str] = {
        (k or "out"): v for k, v in handle_to_target.items()
    }

    def router(state: dict[str, Any]) -> str:
        handle = state.get("_next_handle") or "out"
        if handle in mapping:
            return mapping[handle]
        if "else" in mapping:
            log.info(
                "router %s: unmatched handle %r, falling back to 'else'",
                source_node_id,
                handle,
            )
            return mapping["else"]
        if "out" in mapping:
            log.info(
                "router %s: unmatched handle %r, falling back to 'out'",
                source_node_id,
                handle,
            )
            return mapping["out"]
        log.warning(
            "router %s: unmatched handle %r and no fallback edge; ending run",
            source_node_id,
            handle,
        )
        return END

    return router
