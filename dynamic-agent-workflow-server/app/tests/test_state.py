"""Tests for state merge semantics + path get/set."""
from __future__ import annotations

import pytest

from app.workflow.state import (
    empty_runtime_state,
    get_path,
    has_path,
    merge_updates,
    set_path,
    set_path_inplace,
)


# ---- merge_updates -------------------------------------------------------


def test_merge_dicts_deep_merge() -> None:
    base = {"a": {"b": 1, "c": 2}, "x": 10}
    upd = {"a": {"c": 99, "d": 3}, "y": 20}
    out = merge_updates(base, upd)
    assert out == {"a": {"b": 1, "c": 99, "d": 3}, "x": 10, "y": 20}


def test_merge_lists_replace_not_concatenate() -> None:
    base = {"items": [1, 2, 3]}
    upd = {"items": [9]}
    out = merge_updates(base, upd)
    assert out == {"items": [9]}


def test_merge_scalar_overwrites_dict() -> None:
    base = {"a": {"deep": True}}
    upd = {"a": "now-a-string"}
    out = merge_updates(base, upd)
    assert out == {"a": "now-a-string"}


def test_merge_dict_replaces_scalar() -> None:
    base = {"a": "scalar"}
    upd = {"a": {"deep": 1}}
    out = merge_updates(base, upd)
    assert out == {"a": {"deep": 1}}


def test_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}}
    upd = {"a": {"c": 2}}
    out = merge_updates(base, upd)
    assert base == {"a": {"b": 1}}
    assert upd == {"a": {"c": 2}}
    assert out == {"a": {"b": 1, "c": 2}}


def test_merge_deepcopies_so_aliasing_is_severed() -> None:
    nested = {"x": 1}
    base = {"a": {"nested": nested}}
    upd = {"a": {"other": 2}}
    out = merge_updates(base, upd)
    out["a"]["nested"]["x"] = 999
    assert nested == {"x": 1}  # original list/dict not affected


def test_merge_empty_updates_returns_copy_of_state() -> None:
    base = {"a": 1, "b": [1, 2]}
    out = merge_updates(base, {})
    assert out == base
    out["b"].append(99)
    assert base["b"] == [1, 2]


def test_merge_rejects_non_dict_inputs() -> None:
    with pytest.raises(TypeError):
        merge_updates("not a dict", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        merge_updates({}, "not a dict")  # type: ignore[arg-type]


# ---- get_path / has_path -------------------------------------------------


def test_get_path_simple_and_nested() -> None:
    state = {"variables": {"system": {"userQuery": "hi"}, "nodes": {}}}
    assert get_path(state, "variables.system.userQuery") == "hi"
    assert get_path(state, "variables.nodes") == {}


def test_get_path_missing_returns_default() -> None:
    state = {"a": {"b": 1}}
    assert get_path(state, "a.missing") is None
    assert get_path(state, "a.missing", default="fallback") == "fallback"
    assert get_path(state, "z.y.x") is None


def test_get_path_through_non_dict_returns_default() -> None:
    state = {"a": [1, 2, 3]}
    # Lists aren't walked; treat as miss.
    assert get_path(state, "a.0") is None


def test_get_path_empty_returns_target() -> None:
    state = {"a": 1}
    assert get_path(state, "") == state


def test_has_path_distinguishes_none_value_from_missing_key() -> None:
    state = {"a": None, "b": {"c": None}}
    assert has_path(state, "a") is True
    assert has_path(state, "b.c") is True
    assert has_path(state, "missing") is False
    assert has_path(state, "b.missing") is False


# ---- set_path / set_path_inplace -----------------------------------------


def test_set_path_creates_intermediate_dicts() -> None:
    state: dict = {}
    out = set_path(state, "a.b.c", 42)
    assert out == {"a": {"b": {"c": 42}}}
    # Original untouched.
    assert state == {}


def test_set_path_inplace_mutates() -> None:
    state: dict = {"existing": 1}
    set_path_inplace(state, "new.deep", "v")
    assert state == {"existing": 1, "new": {"deep": "v"}}


def test_set_path_overwrites_non_dict_intermediate() -> None:
    state: dict = {"a": "scalar"}
    set_path_inplace(state, "a.b", 1)
    assert state == {"a": {"b": 1}}


def test_set_path_writes_top_level_key() -> None:
    state: dict = {}
    set_path_inplace(state, "x", 42)
    assert state == {"x": 42}


def test_set_path_empty_path_raises() -> None:
    with pytest.raises(ValueError):
        set_path_inplace({}, "", 1)


# ---- empty_runtime_state -------------------------------------------------


def test_empty_runtime_state_has_canonical_shape() -> None:
    s = empty_runtime_state(run_id="r1", workflow_id="wf", workflow_version=2)
    assert s["run_id"] == "r1"
    assert s["status"] == "pending"
    assert s["step_count"] == 0
    assert s["variables"] == {"system": {}, "runtime": {}, "nodes": {}}
    assert s["messages"] == []
    assert s["events"] == []
    # Transient routing fields exist but are None.
    assert "_next_handle" in s and s["_next_handle"] is None
    assert "_resume_input" in s and s["_resume_input"] is None
