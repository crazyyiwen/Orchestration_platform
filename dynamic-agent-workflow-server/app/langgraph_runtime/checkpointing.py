"""Checkpointer factory.

For v1 we use LangGraph's :class:`InMemorySaver`. Native interrupts +
``Command(resume=...)`` work cleanly within a single server process, which
satisfies the acceptance criteria. Domain-level state persistence (Mongo) is
handled separately by the run repository so a server restart can still report
historical run state — just not resume an in-flight pause.

Cross-process resume (Mongo-backed ``BaseCheckpointSaver``) is a follow-on:
the protocol exists, but a correct implementation needs careful serializer
work. The ``CheckpointStore`` interface here keeps the rest of the codebase
free of LangGraph-specific imports so swapping is a one-place change.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


def make_checkpointer() -> BaseCheckpointSaver:
    """Return the active checkpointer instance for compiled graphs."""
    return InMemorySaver()
