"""Unit tests for the graph helper functions."""
from __future__ import annotations

from app.schemas.workflow import WorkflowDefinition
from app.workflow.graph_utils import (
    adjacency,
    adjacency_ids,
    find_duplicate_edges,
    find_output_nodes,
    find_start_node,
    has_cycle,
    incoming,
    outgoing_handles,
    reachable_from,
    topological_order,
    unreachable_node_ids,
)


def _make(nodes: list[dict], edges: list[dict]) -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {"workflow_id": "wf-test", "nodes": nodes, "edges": edges}
    )


def test_adjacency_and_incoming_round_trip() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "c", "type": "output", "name": "c"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "c"},
        ],
    )
    adj = adjacency(wf)
    assert [e.target for e in adj["a"]] == ["b"]
    assert [e.target for e in adj["b"]] == ["c"]
    assert adj["c"] == []
    inc = incoming(wf)
    assert inc["a"] == []
    assert [e.source for e in inc["b"]] == ["a"]
    assert [e.source for e in inc["c"]] == ["b"]


def test_find_start_node_explicit_takes_precedence() -> None:
    wf = _make(
        [
            {"id": "x", "type": "start", "name": "x"},
            {"id": "y", "type": "llm", "name": "y"},  # also has 0 incoming
        ],
        [],
    )
    assert find_start_node(wf).id == "x"


def test_find_start_node_inferred_when_unique() -> None:
    wf = _make(
        [
            {"id": "x", "type": "llm", "name": "x"},
            {"id": "y", "type": "output", "name": "y"},
        ],
        [{"id": "e", "source": "x", "target": "y"}],
    )
    assert find_start_node(wf).id == "x"


def test_find_start_node_none_when_ambiguous() -> None:
    wf = _make(
        [
            {"id": "x", "type": "llm", "name": "x"},
            {"id": "y", "type": "llm", "name": "y"},
        ],
        [],
    )
    # Two zero-in-degree nodes, no explicit start → ambiguous.
    assert find_start_node(wf) is None


def test_find_output_nodes() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "output", "name": "b"},
            {"id": "c", "type": "output", "name": "c"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "a", "target": "c", "sourceHandle": "alt"},
        ],
    )
    assert {n.id for n in find_output_nodes(wf)} == {"b", "c"}


def test_topological_order_dag() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "c", "type": "llm", "name": "c"},
            {"id": "d", "type": "output", "name": "d"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "a", "target": "c", "sourceHandle": "alt"},
            {"id": "e3", "source": "b", "target": "d"},
            {"id": "e4", "source": "c", "target": "d", "sourceHandle": "alt2"},
        ],
    )
    sorted_ids, cyclic = topological_order(wf)
    assert set(sorted_ids) == {"a", "b", "c", "d"}
    assert cyclic == set()
    assert sorted_ids.index("a") < sorted_ids.index("b")
    assert sorted_ids.index("a") < sorted_ids.index("c")
    assert sorted_ids.index("b") < sorted_ids.index("d")


def test_topological_order_detects_cycle_members() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "c", "type": "llm", "name": "c"},
            {"id": "d", "type": "output", "name": "d"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "c"},
            {"id": "e3", "source": "c", "target": "b"},  # creates b<->c cycle
            {"id": "e4", "source": "b", "target": "d", "sourceHandle": "exit"},
        ],
    )
    sorted_ids, cyclic = topological_order(wf)
    assert "a" in sorted_ids and "d" not in sorted_ids
    # b and c are both in (or feed into) the cycle, and d is downstream of it.
    assert {"b", "c"}.issubset(cyclic)
    assert has_cycle(wf) is True


def test_self_loop_is_a_cycle() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "c", "type": "output", "name": "c"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "b", "sourceHandle": "again"},
            {"id": "e3", "source": "b", "target": "c", "sourceHandle": "done"},
        ],
    )
    _, cyclic = topological_order(wf)
    assert "b" in cyclic
    assert "c" in cyclic  # downstream of the cycle


def test_reachability_and_unreachable_set() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "orphan", "type": "llm", "name": "orphan"},
            {"id": "c", "type": "output", "name": "c"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "c"},
        ],
    )
    seen = reachable_from(wf, "a")
    assert seen == {"a", "b", "c"}
    assert unreachable_node_ids(wf, "a") == {"orphan"}


def test_outgoing_handles_distinct_per_node() -> None:
    wf = _make(
        [
            {"id": "rule", "type": "rule", "name": "rule"},
            {"id": "x", "type": "output", "name": "x"},
            {"id": "y", "type": "output", "name": "y"},
            {"id": "z", "type": "output", "name": "z"},
        ],
        [
            {"id": "e1", "source": "rule", "target": "x", "sourceHandle": "case_1"},
            {"id": "e2", "source": "rule", "target": "y", "sourceHandle": "case_2"},
            {"id": "e3", "source": "rule", "target": "z", "sourceHandle": "else"},
        ],
    )
    assert outgoing_handles(wf)["rule"] == {"case_1", "case_2", "else"}


def test_find_duplicate_edges_keys_on_source_target_handle() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "output", "name": "b"},
        ],
        [
            {"id": "e1", "source": "a", "target": "b", "sourceHandle": "out"},
            # Same source/target/handle as e1 → duplicate.
            {"id": "e2", "source": "a", "target": "b", "sourceHandle": "out"},
            # Same source/target but different handle → NOT a duplicate.
            {"id": "e3", "source": "a", "target": "b", "sourceHandle": "alt"},
        ],
    )
    dups = find_duplicate_edges(wf)
    assert len(dups) == 1
    assert dups[0][1].id == "e2"


def test_adjacency_ids_is_target_only_view() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a"},
            {"id": "b", "type": "output", "name": "b"},
        ],
        [{"id": "e1", "source": "a", "target": "b"}],
    )
    assert adjacency_ids(wf) == {"a": ["b"], "b": []}
