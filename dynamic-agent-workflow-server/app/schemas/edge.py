"""Generic Edge model.

The runtime cares only about ``id``, ``source``, ``target``, and ``sourceHandle``
(used by the dynamic router to match ``next_handle``). React Flow style/data
fields are tolerated via extras='allow'.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None
