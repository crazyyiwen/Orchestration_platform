"""Workflow validator (spec §2).

``validate()`` is a pure function: it takes a :class:`WorkflowDefinition` and
returns a :class:`ValidationReport`. The checks are tiny composable functions
that each yield :class:`ValidationIssue` objects.

Phase 7 will register an executor registry and pass its known node-type set
into ``validate(known_types=...)`` to tighten unknown-type checking. Until
then, unknown types are reported as **warnings** so production frontend JSON
(which uses extra type names like ``tool``/``variable``) does not fail
validation prematurely.
"""
from __future__ import annotations

from typing import Iterable

from app.schemas.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from app.schemas.workflow import WorkflowDefinition
from app.workflow.graph_utils import (
    adjacency,
    find_duplicate_edges,
    find_duplicate_ids,
    find_duplicate_names,
    find_output_nodes,
    find_start_node,
    incoming,
    outgoing_handles,
    topological_order,
    unreachable_node_ids,
)

# The 13 canonical types from spec §6. Frontend may emit additional names;
# unknown types become warnings unless caller passes a stricter set.
CANONICAL_NODE_TYPES: frozenset[str] = frozenset(
    {
        "start",
        "llm",
        "agent",
        "rule",
        "script",
        "human_input",
        "output",
        "sub_flow",
        "variable_update",
        "http_request",
        "guardrail",
        "approval",
        "external_agent",
    }
)


def _err(code: str, message: str, **kw) -> ValidationIssue:
    return ValidationIssue(code=code, severity=ValidationSeverity.ERROR, message=message, **kw)


def _warn(code: str, message: str, **kw) -> ValidationIssue:
    return ValidationIssue(code=code, severity=ValidationSeverity.WARNING, message=message, **kw)


def _info(code: str, message: str, **kw) -> ValidationIssue:
    return ValidationIssue(code=code, severity=ValidationSeverity.INFO, message=message, **kw)


def validate(
    definition: WorkflowDefinition,
    *,
    known_types: Iterable[str] | None = None,
    allow_cycles: bool = False,
) -> ValidationReport:
    """Run all structural + semantic checks. Returns a partitioned report."""
    issues: list[ValidationIssue] = []
    issues += _check_basic_structure(definition)
    issues += _check_unique_ids(definition)
    issues += _check_unique_names(definition)
    issues += _check_edges_reference_real_nodes(definition)
    issues += _check_no_duplicate_edges(definition)
    issues += _check_node_types(definition, known_types)
    issues += _check_start_and_output(definition)
    issues += _check_outgoing_edges_per_node_type(definition)
    issues += _check_required_handles(definition)
    issues += _check_no_fan_out(definition)
    issues += _check_cycles(definition, allow_cycles=allow_cycles)
    issues += _check_reachability(definition)
    return ValidationReport.from_issues(issues)


# ---------- individual checks ---------------------------------------------


def _check_basic_structure(d: WorkflowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not d.nodes:
        issues.append(_err("workflow_empty_nodes", "workflow has no nodes"))
    if not d.edges and len(d.nodes) > 1:
        issues.append(
            _err(
                "workflow_empty_edges",
                f"workflow has {len(d.nodes)} nodes but no edges connecting them",
            )
        )
    return issues


def _check_unique_ids(d: WorkflowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    dup_node_ids = find_duplicate_ids(d.nodes)
    for nid in dup_node_ids:
        issues.append(
            _err("duplicate_node_id", f"duplicate node id: {nid!r}", node_id=nid)
        )
    dup_edge_ids = find_duplicate_ids(d.edges)
    for eid in dup_edge_ids:
        issues.append(
            _err("duplicate_edge_id", f"duplicate edge id: {eid!r}", edge_id=eid)
        )
    return issues


def _check_unique_names(d: WorkflowDefinition) -> list[ValidationIssue]:
    """Node names must be unique because runtime variables key off them
    (``{{nodes.<name>.result}}``)."""
    issues: list[ValidationIssue] = []
    for name in find_duplicate_names(d.nodes):
        # Tag every offender so the frontend can highlight all of them.
        offenders = [n.id for n in d.nodes if n.name == name]
        for nid in offenders:
            issues.append(
                _err(
                    "duplicate_node_name",
                    f"duplicate node name: {name!r}",
                    node_id=nid,
                    details={"name": name, "node_ids": offenders},
                )
            )
    return issues


def _check_edges_reference_real_nodes(d: WorkflowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_ids = {n.id for n in d.nodes}
    for e in d.edges:
        if e.source not in node_ids:
            issues.append(
                _err(
                    "edge_source_missing",
                    f"edge {e.id!r} source {e.source!r} does not reference a node",
                    edge_id=e.id,
                    details={"source": e.source},
                )
            )
        if e.target not in node_ids:
            issues.append(
                _err(
                    "edge_target_missing",
                    f"edge {e.id!r} target {e.target!r} does not reference a node",
                    edge_id=e.id,
                    details={"target": e.target},
                )
            )
    return issues


def _check_no_duplicate_edges(d: WorkflowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for first, dup in find_duplicate_edges(d):
        issues.append(
            _err(
                "duplicate_edge",
                f"duplicate edge: ({dup.source!r} → {dup.target!r}, "
                f"sourceHandle={dup.sourceHandle!r}) — already on edge {first.id!r}",
                edge_id=dup.id,
                details={
                    "source": dup.source,
                    "target": dup.target,
                    "sourceHandle": dup.sourceHandle,
                    "duplicate_of": first.id,
                },
            )
        )
    return issues


def _check_node_types(
    d: WorkflowDefinition, known_types: Iterable[str] | None
) -> list[ValidationIssue]:
    """Unknown node types are flagged.

    * If caller provides ``known_types`` (Phase 7+: the executor registry),
      unknown types are **errors** — they cannot run.
    * Otherwise, unknown types fall outside the canonical 13 and are reported
      as **warnings** so existing frontend JSON does not fail prematurely.
    """
    issues: list[ValidationIssue] = []
    if known_types is None:
        gate = CANONICAL_NODE_TYPES
        severity_factory = _warn
    else:
        gate = frozenset(known_types)
        severity_factory = _err
    for n in d.nodes:
        if n.type not in gate:
            issues.append(
                severity_factory(
                    "unknown_node_type",
                    f"node {n.id!r} has unknown type {n.type!r}",
                    node_id=n.id,
                    details={"type": n.type, "allowed": sorted(gate)},
                )
            )
    return issues


def _check_start_and_output(d: WorkflowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    explicit_starts = [n for n in d.nodes if n.type == "start"]
    if len(explicit_starts) > 1:
        for n in explicit_starts:
            issues.append(
                _err(
                    "multiple_start_nodes",
                    f"workflow has {len(explicit_starts)} explicit start nodes; only one is allowed",
                    node_id=n.id,
                )
            )
    if not explicit_starts:
        # Try to infer: a single node with no incoming edges.
        inferred = find_start_node(d)
        if inferred is None and d.nodes:
            issues.append(
                _err(
                    "missing_start_node",
                    "workflow has no node with type='start' and no unique entry node could be inferred",
                )
            )

    outputs = find_output_nodes(d)
    if not outputs:
        issues.append(_err("missing_output_node", "workflow has no output node"))
    return issues


def _check_outgoing_edges_per_node_type(
    d: WorkflowDefinition,
) -> list[ValidationIssue]:
    """Non-output nodes should have at least one outgoing edge.

    Output nodes must have *zero* outgoing edges (they're terminal).
    """
    issues: list[ValidationIssue] = []
    adj = adjacency(d)
    for n in d.nodes:
        outs = adj.get(n.id, [])
        if n.type == "output":
            if outs:
                issues.append(
                    _err(
                        "output_node_has_outgoing",
                        f"output node {n.id!r} must be terminal but has {len(outs)} outgoing edge(s)",
                        node_id=n.id,
                    )
                )
        else:
            if not outs:
                # Warning, not error — sub_flow / external_agent / agent with handoff
                # may legitimately fan into a follow-up that is dynamically chosen, and
                # the frontend may have a draft state that should still parse.
                issues.append(
                    _warn(
                        "node_has_no_outgoing",
                        f"node {n.id!r} (type={n.type!r}) has no outgoing edges; run will end here",
                        node_id=n.id,
                    )
                )
    return issues


def _check_required_handles(d: WorkflowDefinition) -> list[ValidationIssue]:
    """Generic check: if a node's config declares ``required_handles: [...]``,
    every entry must correspond to an outgoing edge with that ``sourceHandle``.

    Per-node-type executors can also encode per-config requirements; those are
    delegated to ``executor.validate_config()`` in Phase 7. This check covers
    the generic case so that Approval and Rule nodes already enjoy a sane
    default (their configs typically include this field).
    """
    issues: list[ValidationIssue] = []
    handles_by_node = outgoing_handles(d)
    for n in d.nodes:
        cfg = n.config if isinstance(n.config, dict) else {}
        required = cfg.get("required_handles")
        if not isinstance(required, list):
            continue
        present = handles_by_node.get(n.id, set())
        for handle in required:
            if handle not in present:
                issues.append(
                    _err(
                        "missing_required_handle",
                        f"node {n.id!r} (type={n.type!r}) requires outgoing handle "
                        f"{handle!r} but no edge has sourceHandle={handle!r}",
                        node_id=n.id,
                        details={"required_handle": handle, "present": sorted(map(str, present))},
                    )
                )
    return issues


def _check_no_fan_out(d: WorkflowDefinition) -> list[ValidationIssue]:
    """Multiple outgoing edges with the same ``sourceHandle`` from one node.

    This is **not** a structural error in this runtime — it's the agent-handoff
    pattern: an agent node returns ``next_handle="handoff"`` and the router
    must pick one of N candidates based on the executor's output payload.
    The Phase 8 compiler defines the tie-break rule.

    Reported as a **warning** so dev tools surface it (it's still a useful
    smell when you didn't intend handoff-style routing). Genuine parallel
    branches — which require reducer-aware state — remain out of scope and
    will be flagged separately when the compiler encounters them.
    """
    issues: list[ValidationIssue] = []
    seen: dict[tuple[str, str | None], str] = {}  # (source, handle) -> first edge id
    for e in d.edges:
        key = (e.source, e.sourceHandle)
        if key in seen:
            issues.append(
                _warn(
                    "ambiguous_handle_routing",
                    f"node {e.source!r} has multiple outgoing edges with "
                    f"sourceHandle={e.sourceHandle!r}; the executor must "
                    f"return enough info for the router to pick one (handoff pattern)",
                    edge_id=e.id,
                    details={
                        "source": e.source,
                        "sourceHandle": e.sourceHandle,
                        "first_edge": seen[key],
                    },
                )
            )
        else:
            seen[key] = e.id
    return issues


def _check_cycles(
    d: WorkflowDefinition, *, allow_cycles: bool
) -> list[ValidationIssue]:
    _, cyclic = topological_order(d)
    if not cyclic:
        return []
    if allow_cycles:
        return [
            _info(
                "cycle_present",
                f"workflow contains a cycle through {sorted(cyclic)} (allowed by configuration)",
                details={"cyclic_nodes": sorted(cyclic)},
            )
        ]
    return [
        _err(
            "cycle_detected",
            f"workflow contains a cycle involving nodes: {sorted(cyclic)}",
            details={"cyclic_nodes": sorted(cyclic)},
        )
    ]


def _check_reachability(d: WorkflowDefinition) -> list[ValidationIssue]:
    """Disconnected nodes are reported as warnings (per spec §2)."""
    start = find_start_node(d)
    if start is None:
        return []  # already handled by missing_start_node
    unreachable = unreachable_node_ids(d, start.id)
    if not unreachable:
        return []
    issues: list[ValidationIssue] = []
    for nid in sorted(unreachable):
        issues.append(
            _warn(
                "unreachable_node",
                f"node {nid!r} is not reachable from the start node",
                node_id=nid,
            )
        )
    return issues
