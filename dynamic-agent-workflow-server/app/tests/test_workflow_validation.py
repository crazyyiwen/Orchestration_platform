"""Tests for the workflow validator (spec §2)."""
from __future__ import annotations

import json
from pathlib import Path

from app.schemas.validation import ValidationSeverity
from app.schemas.workflow import WorkflowDefinition
from app.workflow.loader import WorkflowLoader
from app.workflow.validation import CANONICAL_NODE_TYPES, validate

FIXTURES = Path(__file__).parent / "fixtures"


def _make(nodes: list[dict], edges: list[dict]) -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {"workflow_id": "wf-test", "nodes": nodes, "edges": edges}
    )


def _codes(report, severity: ValidationSeverity) -> list[str]:
    items = {
        ValidationSeverity.ERROR.value: report.errors,
        ValidationSeverity.WARNING.value: report.warnings,
        ValidationSeverity.INFO.value: report.infos,
    }[severity.value]
    return [i.code for i in items]


def test_minimal_valid_workflow_is_valid() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [{"id": "e", "source": "s", "target": "o", "sourceHandle": "out"}],
    )
    report = validate(wf)
    assert report.is_valid, report.errors
    assert report.errors == []


def test_empty_nodes_rejected() -> None:
    wf = _make([], [])
    report = validate(wf)
    assert not report.is_valid
    assert "workflow_empty_nodes" in _codes(report, ValidationSeverity.ERROR)


def test_duplicate_node_ids_rejected() -> None:
    wf = _make(
        [
            {"id": "a", "type": "start", "name": "a1"},
            {"id": "a", "type": "output", "name": "a2"},
        ],
        [{"id": "e", "source": "a", "target": "a"}],
    )
    report = validate(wf)
    assert "duplicate_node_id" in _codes(report, ValidationSeverity.ERROR)


def test_duplicate_node_names_rejected() -> None:
    wf = _make(
        [
            {"id": "n1", "type": "start", "name": "same"},
            {"id": "n2", "type": "output", "name": "same"},
        ],
        [{"id": "e", "source": "n1", "target": "n2"}],
    )
    report = validate(wf)
    codes = _codes(report, ValidationSeverity.ERROR)
    assert codes.count("duplicate_node_name") >= 2  # tagged on every offender


def test_edge_referencing_missing_node_rejected() -> None:
    wf = _make(
        [{"id": "a", "type": "start", "name": "a"}, {"id": "b", "type": "output", "name": "b"}],
        [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "ghost", "target": "b"},
            {"id": "e3", "source": "a", "target": "phantom"},
        ],
    )
    report = validate(wf)
    codes = _codes(report, ValidationSeverity.ERROR)
    assert "edge_source_missing" in codes
    assert "edge_target_missing" in codes


def test_duplicate_edges_rejected() -> None:
    wf = _make(
        [{"id": "a", "type": "start", "name": "a"}, {"id": "b", "type": "output", "name": "b"}],
        [
            {"id": "e1", "source": "a", "target": "b", "sourceHandle": "x"},
            {"id": "e2", "source": "a", "target": "b", "sourceHandle": "x"},  # duplicate
        ],
    )
    report = validate(wf)
    assert "duplicate_edge" in _codes(report, ValidationSeverity.ERROR)


def test_missing_output_node_rejected() -> None:
    wf = _make(
        [{"id": "s", "type": "start", "name": "s"}, {"id": "l", "type": "llm", "name": "l"}],
        [{"id": "e", "source": "s", "target": "l"}],
    )
    report = validate(wf)
    assert "missing_output_node" in _codes(report, ValidationSeverity.ERROR)


def test_multiple_explicit_starts_rejected() -> None:
    wf = _make(
        [
            {"id": "s1", "type": "start", "name": "s1"},
            {"id": "s2", "type": "start", "name": "s2"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [
            {"id": "e1", "source": "s1", "target": "o"},
            {"id": "e2", "source": "s2", "target": "o", "sourceHandle": "alt"},
        ],
    )
    report = validate(wf)
    assert "multiple_start_nodes" in _codes(report, ValidationSeverity.ERROR)


def test_output_node_with_outgoing_edges_rejected() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "o", "type": "output", "name": "o"},
            {"id": "extra", "type": "llm", "name": "extra"},
        ],
        [
            {"id": "e1", "source": "s", "target": "o"},
            {"id": "e2", "source": "o", "target": "extra"},  # output is terminal
        ],
    )
    report = validate(wf)
    assert "output_node_has_outgoing" in _codes(report, ValidationSeverity.ERROR)


def test_unknown_node_type_warns_by_default() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "x", "type": "totally_made_up", "name": "x"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [
            {"id": "e1", "source": "s", "target": "x"},
            {"id": "e2", "source": "x", "target": "o"},
        ],
    )
    report = validate(wf)
    assert report.is_valid  # warnings only
    assert "unknown_node_type" in _codes(report, ValidationSeverity.WARNING)


def test_unknown_node_type_errors_when_known_types_passed() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "x", "type": "totally_made_up", "name": "x"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [
            {"id": "e1", "source": "s", "target": "x"},
            {"id": "e2", "source": "x", "target": "o"},
        ],
    )
    report = validate(wf, known_types=CANONICAL_NODE_TYPES)
    assert not report.is_valid
    assert "unknown_node_type" in _codes(report, ValidationSeverity.ERROR)


def test_required_handles_check() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "ap",
                "type": "approval",
                "name": "ap",
                "config": {"required_handles": ["approved", "rejected"]},
            },
            {"id": "o1", "type": "output", "name": "o1"},
        ],
        [
            {"id": "e1", "source": "s", "target": "ap"},
            {"id": "e2", "source": "ap", "target": "o1", "sourceHandle": "approved"},
            # missing the "rejected" branch
        ],
    )
    report = validate(wf)
    assert "missing_required_handle" in _codes(report, ValidationSeverity.ERROR)


def test_fan_out_with_same_handle_warns_not_errors() -> None:
    """Multiple edges with the same sourceHandle from one source is the agent
    handoff pattern; the validator surfaces it as a warning (Phase 8's compiler
    is the one that decides how to route)."""
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "o1", "type": "output", "name": "o1"},
            {"id": "o2", "type": "output", "name": "o2"},
        ],
        [
            {"id": "e1", "source": "s", "target": "o1", "sourceHandle": "out"},
            {"id": "e2", "source": "s", "target": "o2", "sourceHandle": "out"},
        ],
    )
    report = validate(wf)
    assert report.is_valid  # warning, not error
    assert "ambiguous_handle_routing" in _codes(report, ValidationSeverity.WARNING)


def test_cycle_rejected_by_default() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "a", "type": "llm", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [
            {"id": "e1", "source": "s", "target": "a"},
            {"id": "e2", "source": "a", "target": "b"},
            {"id": "e3", "source": "b", "target": "a", "sourceHandle": "loop"},
            {"id": "e4", "source": "a", "target": "o", "sourceHandle": "exit"},
        ],
    )
    report = validate(wf)
    assert not report.is_valid
    assert "cycle_detected" in _codes(report, ValidationSeverity.ERROR)


def test_cycle_allowed_when_flag_set_emits_info() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "a", "type": "rule", "name": "a"},
            {"id": "b", "type": "llm", "name": "b"},
            {"id": "o", "type": "output", "name": "o"},
        ],
        [
            {"id": "e1", "source": "s", "target": "a"},
            {"id": "e2", "source": "a", "target": "b", "sourceHandle": "loop"},
            {"id": "e3", "source": "b", "target": "a"},
            {"id": "e4", "source": "a", "target": "o", "sourceHandle": "exit"},
        ],
    )
    report = validate(wf, allow_cycles=True)
    assert report.is_valid
    assert "cycle_present" in _codes(report, ValidationSeverity.INFO)


def test_unreachable_node_warning() -> None:
    wf = _make(
        [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "o", "type": "output", "name": "o"},
            {"id": "orphan", "type": "llm", "name": "orphan"},
            {"id": "o2", "type": "output", "name": "o2"},
        ],
        [
            {"id": "e1", "source": "s", "target": "o"},
            {"id": "e2", "source": "orphan", "target": "o2"},  # orphaned subgraph
        ],
    )
    report = validate(wf)
    assert report.is_valid  # only warnings
    assert "unreachable_node" in _codes(report, ValidationSeverity.WARNING)


def test_real_frontend_workflow_validates_with_warnings_only() -> None:
    """The actual buying-channel-assistant.json must pass with no errors.

    It uses node types like ``tool`` and ``variable`` that aren't in the
    canonical 13 — those should be warnings, not blockers.
    """
    import subprocess

    proc = subprocess.run(
        ["git", "show", "6462ae2:feeding_files/buying-channel-assistant.json"],
        capture_output=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        # Bytes mode + explicit decode — the file contains characters that
        # cp1252 (Windows default) cannot represent.
    )
    if proc.returncode != 0 or not proc.stdout:
        # Not in a git repo (e.g. installed wheel) — skip.
        import pytest

        pytest.skip("git not available or feeding_files commit missing")
    raw = json.loads(proc.stdout.decode("utf-8"))
    wf = WorkflowLoader.load_inline(raw, workflow_id="buying-channel-assistant")
    # Real agent workflows have intentional cycles (agent ↔ tool loop) and
    # handoff fan-out. allow_cycles=True acknowledges the bounded agent loop.
    report = validate(wf, allow_cycles=True)
    assert report.is_valid, [f"{i.code}: {i.message}" for i in report.errors]
    warning_codes = _codes(report, ValidationSeverity.WARNING)
    assert "unknown_node_type" in warning_codes  # tool/variable
    assert "ambiguous_handle_routing" in warning_codes  # agent handoff fan-out
    info_codes = _codes(report, ValidationSeverity.INFO)
    assert "cycle_present" in info_codes  # bounded agent loop


def test_load_then_validate_minimal_fixture() -> None:
    raw = json.loads(
        (FIXTURES / "sample_minimal_workflow.json").read_text(encoding="utf-8")
    )
    wf = WorkflowLoader.load_inline(raw, workflow_id="wf-min")
    report = validate(wf)
    assert report.is_valid, report.errors
