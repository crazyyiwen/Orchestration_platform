"""Schema parsing tests — does the canonical model accept real frontend JSON?"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.workflow import WorkflowDefinition

FIXTURES = Path(__file__).parent / "fixtures"


def _load_minimal() -> dict:
    return json.loads((FIXTURES / "sample_minimal_workflow.json").read_text(encoding="utf-8"))


def test_minimal_workflow_parses_with_extras() -> None:
    raw = _load_minimal()
    raw["workflow_id"] = "wf-min"
    wf = WorkflowDefinition.model_validate(raw)
    assert wf.workflow_id == "wf-min"
    assert wf.workflow_version == 1
    assert len(wf.nodes) == 3
    assert {n.type for n in wf.nodes} == {"start", "llm", "output"}
    # extras (UI fields) must round-trip via Node.extras
    start = next(n for n in wf.nodes if n.id == "start")
    assert "position" in start.extras
    summary = wf.summary()
    assert summary.node_count == 3
    assert summary.edge_count == 2


def test_node_tolerates_unknown_type() -> None:
    raw = _load_minimal()
    raw["workflow_id"] = "wf-unknown"
    raw["nodes"][1]["type"] = "totally_new_node_type"
    wf = WorkflowDefinition.model_validate(raw)
    # Schema does not gate on node-type enum (validator + registry do that).
    assert any(n.type == "totally_new_node_type" for n in wf.nodes)


def test_edge_default_handles_are_none() -> None:
    raw = _load_minimal()
    raw["workflow_id"] = "wf-edges"
    # Strip handles entirely — defaults must be None.
    for e in raw["edges"]:
        e.pop("sourceHandle", None)
        e.pop("targetHandle", None)
    wf = WorkflowDefinition.model_validate(raw)
    assert all(e.sourceHandle is None for e in wf.edges)


def test_validation_report_partitions_issues() -> None:
    from app.schemas.validation import (
        ValidationIssue,
        ValidationReport,
        ValidationSeverity,
    )

    issues = [
        ValidationIssue(code="A", severity=ValidationSeverity.ERROR, message="boom"),
        ValidationIssue(code="B", severity=ValidationSeverity.WARNING, message="meh"),
        ValidationIssue(code="C", severity=ValidationSeverity.INFO, message="fyi"),
    ]
    report = ValidationReport.from_issues(issues)
    assert report.is_valid is False
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    assert len(report.infos) == 1


def test_validation_report_valid_when_no_errors() -> None:
    from app.schemas.validation import (
        ValidationIssue,
        ValidationReport,
        ValidationSeverity,
    )

    issues = [ValidationIssue(code="X", severity=ValidationSeverity.WARNING, message="ok-ish")]
    report = ValidationReport.from_issues(issues)
    assert report.is_valid is True
    assert len(report.warnings) == 1
