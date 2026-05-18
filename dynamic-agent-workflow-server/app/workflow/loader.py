"""Workflow loader — 3 backends behind a single facade.

Sources:
  * ``MetadataApiWorkflowSource`` — preferred when ``METADATA_API_ENABLED=true``;
    fetches via httpx from the existing FastAPI metadata API.
  * ``MongoWorkflowSource`` — used when METADATA_API_ENABLED=false; reads from
    a local ``workflow_definitions`` collection that the operator can seed.
  * ``InlineWorkflowSource`` — for ``/run-inline`` and tests; no external IO.

All three return a normalized :class:`WorkflowDefinition`. The loader tolerates
multiple wrapper shapes returned by metadata APIs:

  * bare definition: ``{nodes, edges, ...}``
  * wrapped: ``{workflow_id, version, name, definition: {...}}``
  * alt-wrapped: ``{workflow: {...}, ...}``
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import ValidationError as PydanticValidationError
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.errors import (
    CompilationError,
    ConfigurationError,
    WorkflowNotFoundError,
)
from app.db.mongodb import MongoDB
from app.schemas.workflow import WorkflowDefinition

log = logging.getLogger(__name__)

LOCAL_DEFINITIONS_COLLECTION = "workflow_definitions"


class BaseWorkflowSource(ABC):
    @abstractmethod
    async def fetch(self, workflow_id: str, version: int | None = None) -> dict[str, Any]:
        ...


class InlineWorkflowSource(BaseWorkflowSource):
    """Wraps a pre-supplied dict so the loader pipeline is uniform."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def fetch(self, workflow_id: str, version: int | None = None) -> dict[str, Any]:
        return self._payload


class MetadataApiWorkflowSource(BaseWorkflowSource):
    def __init__(self, base_url: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http_client or httpx.AsyncClient(timeout=10.0)

    async def fetch(self, workflow_id: str, version: int | None = None) -> dict[str, Any]:
        url = f"{self._base_url}/api/workflows/{workflow_id}"
        params = {"version": version} if version is not None else None
        try:
            resp = await self._http.get(url, params=params)
        except httpx.HTTPError as e:
            raise ConfigurationError(
                f"metadata API unreachable: {e}", details={"workflow_id": workflow_id}
            ) from e
        if resp.status_code == 404:
            raise WorkflowNotFoundError(
                f"workflow '{workflow_id}' not found in metadata API",
                details={"workflow_id": workflow_id, "version": version},
            )
        if resp.status_code >= 400:
            raise ConfigurationError(
                f"metadata API returned {resp.status_code}",
                details={"workflow_id": workflow_id, "status": resp.status_code},
            )
        return resp.json()


class MongoWorkflowSource(BaseWorkflowSource):
    """Reads from the local ``workflow_definitions`` collection.

    Used only when ``METADATA_API_ENABLED=false`` (per spec §12 — we don't
    duplicate workflow definitions otherwise).
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self._c = db[LOCAL_DEFINITIONS_COLLECTION]

    async def fetch(self, workflow_id: str, version: int | None = None) -> dict[str, Any]:
        query: dict[str, Any] = {"workflow_id": workflow_id}
        if version is not None:
            query["workflow_version"] = version
        sort = [("workflow_version", -1)]
        doc = await self._c.find_one(query, sort=sort)
        if doc is None:
            raise WorkflowNotFoundError(
                f"workflow '{workflow_id}' not found in local Mongo",
                details={"workflow_id": workflow_id, "version": version},
            )
        doc.pop("_id", None)
        return doc


class WorkflowLoader:
    """Facade that picks a source based on settings + Mongo availability."""

    def __init__(
        self,
        settings: Settings,
        mongo: MongoDB | None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._mongo = mongo
        self._http_client = http_client
        self._api_source: MetadataApiWorkflowSource | None = None
        self._mongo_source: MongoWorkflowSource | None = None
        if settings.METADATA_API_ENABLED:
            self._api_source = MetadataApiWorkflowSource(
                settings.METADATA_API_BASE_URL, http_client
            )
        elif mongo is not None and mongo.available:
            self._mongo_source = MongoWorkflowSource(mongo.db)

    async def load_by_id(
        self, workflow_id: str, version: int | None = None
    ) -> WorkflowDefinition:
        source: BaseWorkflowSource | None = self._api_source or self._mongo_source
        if source is None:
            raise ConfigurationError(
                "no workflow source configured (METADATA_API_ENABLED is off and Mongo "
                "is unavailable)"
            )
        payload = await source.fetch(workflow_id, version)
        # Can be optimized in the future
        return self.load_inline(payload, workflow_id=workflow_id, version=version)

    @staticmethod
    def load_inline(
        payload: dict[str, Any],
        *,
        workflow_id: str | None = None,
        version: int | None = None,
    ) -> WorkflowDefinition:
        """Normalize an arbitrary dict (any of the wrapper shapes) into a definition."""
        return _normalize(payload, workflow_id=workflow_id, version=version)


def _normalize(
    payload: dict[str, Any],
    *,
    workflow_id: str | None,
    version: int | None,
) -> WorkflowDefinition:
    if not isinstance(payload, dict):
        raise CompilationError("workflow payload must be a JSON object")

    definition_dict, meta = _extract_definition(payload)
    # Flatten React Flow's ``type: "dynamic"`` node wrappers before validation.
    definition_dict = _normalize_react_flow_nodes(definition_dict)

    merged: dict[str, Any] = dict(definition_dict)
    # The metadata API's `doc` carries the workflow id under ``id``, not
    # ``workflow_id`` — promote it so the schema validator finds it.
    if "workflow_id" not in merged and merged.get("id"):
        merged["workflow_id"] = merged["id"]
    # Apply metadata wrappers without overwriting fields already in the definition.
    if "workflow_id" not in merged:
        merged["workflow_id"] = (
            workflow_id or meta.get("workflow_id") or definition_dict.get("workflow_id")
        )
    if "workflow_version" not in merged:
        v = (
            version
            or meta.get("workflow_version")
            or meta.get("version")
            or definition_dict.get("workflow_version")
            or 1
        )
        try:
            merged["workflow_version"] = int(v)
        except (TypeError, ValueError) as e:
            raise CompilationError(f"invalid workflow_version: {v!r}") from e
    for key in ("name", "description", "interface"):
        if key not in merged and meta.get(key) is not None:
            merged[key] = meta[key]

    if not merged.get("workflow_id"):
        raise CompilationError(
            "workflow_id is required (not present in payload, metadata, or override)"
        )

    merged.setdefault("raw", payload)

    try:
        return WorkflowDefinition.model_validate(merged)
    except PydanticValidationError as e:
        raise CompilationError(
            "workflow definition failed schema validation",
            details={"errors": e.errors(include_url=False)},
        ) from e


def _extract_definition(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (definition_dict, wrapper_metadata).

    Tolerates four wrapper shapes commonly returned by metadata APIs:
      1. bare:        ``{nodes, edges, ...}``
      2. definition:  ``{definition: {...}, workflow_id, version, name}``
      3. workflow:    ``{workflow: {...}, ...}``
      4. meta+doc:    ``{meta: {workflow_id, current_version, ...}, doc: {...}}``
         (the FastAPI metadata-API project on port 8000 uses this)
    """
    if "nodes" in payload and "edges" in payload:
        return payload, {}
    if isinstance(payload.get("definition"), dict):
        meta = {k: v for k, v in payload.items() if k != "definition"}
        return payload["definition"], meta
    if isinstance(payload.get("workflow"), dict):
        meta = {k: v for k, v in payload.items() if k != "workflow"}
        return payload["workflow"], meta
    if isinstance(payload.get("doc"), dict) and "doc" in payload:
        # The metadata API's response shape. ``meta.current_version`` is the
        # latest published version; the definition's own ``version`` may lag.
        # We promote ``current_version`` to the meta dict so it takes precedence.
        raw_meta = payload.get("meta") or {}
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        if "workflow_version" not in meta and meta.get("current_version") is not None:
            meta["workflow_version"] = meta["current_version"]
        return payload["doc"], meta
    raise CompilationError(
        "workflow payload missing 'nodes'/'edges' "
        "(or a 'definition'/'workflow'/'doc' wrapper)"
    )


def _normalize_react_flow_nodes(definition: dict[str, Any]) -> dict[str, Any]:
    """Hoist React Flow node wrappers (``type: "dynamic"`` + ``data.*``).

    The frontend builder emits nodes with the actual type/name/config nested
    under ``data``. The runtime needs them flat. This is a pure transformation
    — nodes that are already flat are left alone.
    """
    nodes = definition.get("nodes")
    if not isinstance(nodes, list):
        return definition
    new_nodes: list[dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            new_nodes.append(n)
            continue
        data = n.get("data")
        is_wrapped = (
            isinstance(data, dict)
            and data.get("type")
            and data.get("name")
        )
        if not is_wrapped:
            new_nodes.append(n)
            continue
        flat = dict(n)
        flat["type"] = data["type"]
        flat["name"] = data["name"]
        flat["config"] = data.get("config") or n.get("config") or {}
        if data.get("description") and not flat.get("description"):
            flat["description"] = data["description"]
        new_nodes.append(flat)
    return {**definition, "nodes": new_nodes}
