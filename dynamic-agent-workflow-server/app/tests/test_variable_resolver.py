"""Tests for the {{path.to.value}} template resolver (spec §5)."""
from __future__ import annotations

import pytest

from app.core.errors import WorkflowServerError
from app.workflow.variables import VariableResolver


def _root() -> dict:
    return {
        "system": {
            "userQuery": "What is the weather?",
            "attachments": ["a.png", "b.png"],
            "humanInput": None,
            "answers": {"first": "yes"},
        },
        "runtime": {
            "workflowMetaData": {
                "workflowId": "wf-42",
                "agentName": "main",
                "iteration": 3,
            }
        },
        "nodes": {
            "llm_1": {"result": {"answer": "It's sunny.", "tokens": 11}},
            "agent_1": {"result": {"toolResult": {"items": [1, 2, 3]}}},
        },
        # Real frontend uses extra namespaces beyond the spec list.
        "flow": {"channel_identified": True, "supplier_data": [{"id": "S1"}]},
    }


# ---- resolve_path --------------------------------------------------------


def test_resolve_path_walks_dotted_path() -> None:
    r = VariableResolver(_root())
    assert r.resolve_path("system.userQuery") == "What is the weather?"
    assert r.resolve_path("nodes.llm_1.result.answer") == "It's sunny."
    assert r.resolve_path("runtime.workflowMetaData.workflowId") == "wf-42"


def test_resolve_path_returns_none_on_miss() -> None:
    r = VariableResolver(_root())
    assert r.resolve_path("system.notExist") is None
    assert r.resolve_path("totally.unknown.namespace") is None


def test_resolve_path_returns_complex_object_intact() -> None:
    r = VariableResolver(_root())
    result = r.resolve_path("nodes.agent_1.result.toolResult")
    assert result == {"items": [1, 2, 3]}


# ---- resolve_string: single-ref preserves type --------------------------


def test_single_ref_string_preserves_dict_type() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("{{nodes.agent_1.result.toolResult}}")
    assert out == {"items": [1, 2, 3]}
    assert isinstance(out, dict)


def test_single_ref_string_preserves_list_type() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("{{system.attachments}}")
    assert out == ["a.png", "b.png"]
    assert isinstance(out, list)


def test_single_ref_string_preserves_int_type() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("{{runtime.workflowMetaData.iteration}}")
    assert out == 3
    assert isinstance(out, int)


def test_single_ref_string_preserves_none() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("{{system.humanInput}}")
    assert out is None


def test_single_ref_with_surrounding_whitespace_still_preserves_type() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("  {{ system.attachments }}  ")
    assert isinstance(out, list)
    assert out == ["a.png", "b.png"]


# ---- resolve_string: interpolation -------------------------------------


def test_string_with_text_interpolates_to_string() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("Q: {{system.userQuery}} (id={{runtime.workflowMetaData.workflowId}})")
    assert out == "Q: What is the weather? (id=wf-42)"


def test_multiple_refs_in_one_string_interpolate() -> None:
    r = VariableResolver(_root())
    # Two refs back-to-back is still string interpolation, not type-preserve.
    out = r.resolve_string("{{system.userQuery}}{{runtime.workflowMetaData.agentName}}")
    assert out == "What is the weather?main"


def test_interpolation_stringifies_collections_as_json() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("attachments={{system.attachments}}")
    # JSON-serialized list (deterministic dump for collections).
    assert out == 'attachments=["a.png", "b.png"]'


def test_interpolation_replaces_none_with_empty_string() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("input=({{system.humanInput}})")
    assert out == "input=()"


def test_interpolation_renders_bool_as_lowercase() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("identified={{flow.channel_identified}}")
    assert out == "identified=true"


def test_interpolation_handles_missing_path_as_empty() -> None:
    r = VariableResolver(_root())
    out = r.resolve_string("hello {{system.missing}} world")
    assert out == "hello  world"


# ---- resolve_value (recursive) -----------------------------------------


def test_resolve_value_walks_dicts_and_lists() -> None:
    r = VariableResolver(_root())
    config = {
        "messages": [
            {"role": "user", "content": "{{system.userQuery}}"},
            {"role": "assistant", "content": "OK; tokens={{nodes.llm_1.result.tokens}}"},
        ],
        "attachments": "{{system.attachments}}",  # whole-value ref → keep list
        "deeply": {"nested": {"id": "{{runtime.workflowMetaData.workflowId}}"}},
        "untouched_int": 42,
        "untouched_none": None,
    }
    out = r.resolve_value(config)
    assert out["messages"][0]["content"] == "What is the weather?"
    assert out["messages"][1]["content"] == "OK; tokens=11"
    assert out["attachments"] == ["a.png", "b.png"]  # type preserved
    assert out["deeply"]["nested"]["id"] == "wf-42"
    assert out["untouched_int"] == 42
    assert out["untouched_none"] is None


def test_resolve_value_does_not_mutate_input() -> None:
    r = VariableResolver(_root())
    config = {"x": "{{system.userQuery}}"}
    snapshot = dict(config)
    r.resolve_value(config)
    assert config == snapshot


def test_resolve_value_passes_non_string_scalars_through() -> None:
    r = VariableResolver(_root())
    assert r.resolve_value(42) == 42
    assert r.resolve_value(True) is True
    assert r.resolve_value(None) is None
    assert r.resolve_value(3.14) == 3.14


# ---- depth guard --------------------------------------------------------


def test_resolver_max_recursion_depth_guard() -> None:
    """Pathologically deep nesting must raise rather than silently overflow."""
    deeply_nested: dict = {"x": "leaf"}
    cur = deeply_nested
    # Build 60 levels of nesting; default max is 50.
    for _ in range(60):
        cur["x"] = {"x": cur["x"]}
        cur = cur["x"]
    r = VariableResolver({"system": {}}, max_recursion_depth=20)
    with pytest.raises(WorkflowServerError, match="recursion"):
        r.resolve_value(deeply_nested)


# ---- non-string sources go through resolve_string unchanged ------------


def test_resolve_string_passes_non_strings_through() -> None:
    r = VariableResolver(_root())
    # If a caller mistakenly hands a non-string in, we don't crash.
    assert r.resolve_string(42) == 42  # type: ignore[arg-type]
    assert r.resolve_string(None) is None  # type: ignore[arg-type]


def test_resolver_root_must_be_dict() -> None:
    with pytest.raises(TypeError):
        VariableResolver([1, 2, 3])  # type: ignore[arg-type]
