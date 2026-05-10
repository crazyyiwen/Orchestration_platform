"""Graph helpers used by both the validator (Phase 4) and the compiler (Phase 8).

All functions are pure — they read a :class:`WorkflowDefinition` and return
plain dicts/sets/lists. No I/O, no mutation.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

from app.schemas.edge import Edge
from app.schemas.node import Node
from app.schemas.workflow import WorkflowDefinition


# ---------- adjacency ------------------------------------------------------


def adjacency(definition: WorkflowDefinition) -> dict[str, list[Edge]]:
    """node_id -> list of outgoing Edge objects (preserves source order)."""
    out: dict[str, list[Edge]] = {n.id: [] for n in definition.nodes}
    for e in definition.edges:
        out.setdefault(e.source, []).append(e)
    return out


def incoming(definition: WorkflowDefinition) -> dict[str, list[Edge]]:
    """node_id -> list of incoming Edge objects."""
    inc: dict[str, list[Edge]] = {n.id: [] for n in definition.nodes}
    for e in definition.edges:
        inc.setdefault(e.target, []).append(e)
    return inc


def adjacency_ids(definition: WorkflowDefinition) -> dict[str, list[str]]:
    return {src: [e.target for e in edges] for src, edges in adjacency(definition).items()}


# ---------- start / output -------------------------------------------------


def find_start_node(definition: WorkflowDefinition) -> Node | None:
    """Return the explicit ``type='start'`` node, or — if absent — infer the
    unique node with no incoming edges. Returns None if neither rule applies.
    """
    explicit = [n for n in definition.nodes if n.type == "start"]
    if explicit:
        return explicit[0]
    inc = incoming(definition)
    candidates = [n for n in definition.nodes if not inc.get(n.id)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def find_output_nodes(definition: WorkflowDefinition) -> list[Node]:
    return [n for n in definition.nodes if n.type == "output"]


# ---------- topological order / cycles ------------------------------------


def topological_order(
    definition: WorkflowDefinition,
) -> tuple[list[str], set[str]]:
    """Kahn's algorithm.

    Returns ``(sorted_ids, cyclic_ids)``: ids that *did* topologically sort and
    the remainder which sit in (or feed into) cycles. If ``cyclic_ids`` is
    empty the graph is a DAG.
    """
    adj = adjacency_ids(definition)
    all_ids = [n.id for n in definition.nodes]
    in_degree: dict[str, int] = {nid: 0 for nid in all_ids}
    for u in all_ids:
        for v in adj.get(u, []):
            if v in in_degree:
                in_degree[v] += 1

    queue: deque[str] = deque(nid for nid, d in in_degree.items() if d == 0)
    sorted_ids: list[str] = []
    while queue:
        u = queue.popleft()
        sorted_ids.append(u)
        for v in adj.get(u, []):
            if v not in in_degree:
                continue
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    cyclic = {nid for nid in all_ids if nid not in sorted_ids}
    return sorted_ids, cyclic


def has_cycle(definition: WorkflowDefinition) -> bool:
    _, cyclic = topological_order(definition)
    return bool(cyclic)


# ---------- reachability ---------------------------------------------------


def reachable_from(definition: WorkflowDefinition, start_id: str) -> set[str]:
    adj = adjacency_ids(definition)
    seen: set[str] = set()
    queue: deque[str] = deque([start_id])
    while queue:
        u = queue.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj.get(u, []):
            if v not in seen:
                queue.append(v)
    return seen


def unreachable_node_ids(
    definition: WorkflowDefinition, start_id: str
) -> set[str]:
    seen = reachable_from(definition, start_id)
    return {n.id for n in definition.nodes} - seen


# ---------- handle helpers -------------------------------------------------


def outgoing_handles(definition: WorkflowDefinition) -> dict[str, set[str | None]]:
    """node_id -> set of distinct sourceHandles on its outgoing edges."""
    by_node: dict[str, set[str | None]] = {n.id: set() for n in definition.nodes}
    for e in definition.edges:
        by_node.setdefault(e.source, set()).add(e.sourceHandle)
    return by_node


def find_duplicate_edges(definition: WorkflowDefinition) -> list[tuple[Edge, Edge]]:
    """Return pairs of edges with the same (source, target, sourceHandle)."""
    seen: dict[tuple[str, str, str | None], Edge] = {}
    dups: list[tuple[Edge, Edge]] = []
    for e in definition.edges:
        key = (e.source, e.target, e.sourceHandle)
        if key in seen:
            dups.append((seen[key], e))
        else:
            seen[key] = e
    return dups


def find_duplicate_ids(items: Iterable) -> set[str]:
    """Return ids that appear more than once across an iterable of objects with .id."""
    seen: set[str] = set()
    dups: set[str] = set()
    for it in items:
        oid = it.id  # type: ignore[attr-defined]
        if oid in seen:
            dups.add(oid)
        else:
            seen.add(oid)
    return dups


def find_duplicate_names(nodes: Iterable[Node]) -> set[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for n in nodes:
        if n.name in seen:
            dups.add(n.name)
        else:
            seen.add(n.name)
    return dups
