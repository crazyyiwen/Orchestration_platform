"""Canonical hash for a WorkflowDefinition (used by the compile cache)."""
from __future__ import annotations

import hashlib
import json

from app.schemas.workflow import WorkflowDefinition


def definition_hash(definition: WorkflowDefinition) -> str:
    """Stable sha256 of the *semantically meaningful* parts of the definition.

    UI-only fields (positions, dragging state, etc.) flow through as extras
    on Node/Edge but are excluded from the hash so cosmetic changes don't
    invalidate the compile cache.
    """
    payload = {
        "workflow_id": definition.workflow_id,
        "workflow_version": definition.workflow_version,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "name": n.name,
                "config": n.config,
            }
            for n in definition.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "sourceHandle": e.sourceHandle,
                "targetHandle": e.targetHandle,
            }
            for e in definition.edges
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
