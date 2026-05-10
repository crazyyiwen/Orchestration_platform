"""Unit-level checks of the index specs (no DB needed)."""
from __future__ import annotations

from app.db.indexes import (
    CHECKPOINT_INDEXES,
    COMPILE_CACHE_INDEXES,
    EVENT_INDEXES,
    RUN_INDEXES,
)


def _names(indexes: list) -> set[str]:
    return {ix.document["name"] for ix in indexes}


def test_run_indexes_cover_required_fields() -> None:
    names = _names(RUN_INDEXES)
    assert "run_id_unique" in names
    assert "workflow_id" in names
    assert "status" in names
    assert "created_at_desc" in names
    # run_id index must be unique
    run_id_ix = next(ix for ix in RUN_INDEXES if ix.document["name"] == "run_id_unique")
    assert run_id_ix.document.get("unique") is True


def test_event_indexes_have_unique_run_sequence() -> None:
    names = _names(EVENT_INDEXES)
    assert "run_id_sequence_unique" in names
    ix = next(i for i in EVENT_INDEXES if i.document["name"] == "run_id_sequence_unique")
    assert ix.document.get("unique") is True


def test_checkpoint_indexes_have_run_lookup() -> None:
    names = _names(CHECKPOINT_INDEXES)
    assert "run_id" in names
    assert "run_id_created_at" in names


def test_compile_cache_has_unique_workflow_version() -> None:
    names = _names(COMPILE_CACHE_INDEXES)
    assert "workflow_id_version_unique" in names
    ix = next(
        i
        for i in COMPILE_CACHE_INDEXES
        if i.document["name"] == "workflow_id_version_unique"
    )
    assert ix.document.get("unique") is True
