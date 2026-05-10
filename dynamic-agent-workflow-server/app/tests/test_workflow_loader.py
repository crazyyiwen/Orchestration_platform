"""Loader tests — inline, metadata-API (httpx mock), Mongo, and normalization."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import (
    CompilationError,
    ConfigurationError,
    WorkflowNotFoundError,
)
from app.workflow.loader import (
    LOCAL_DEFINITIONS_COLLECTION,
    MetadataApiWorkflowSource,
    MongoWorkflowSource,
    WorkflowLoader,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _minimal() -> dict:
    return json.loads((FIXTURES / "sample_minimal_workflow.json").read_text(encoding="utf-8"))


# --- Inline / normalize ---------------------------------------------------


def test_load_inline_with_explicit_id_and_version() -> None:
    wf = WorkflowLoader.load_inline(_minimal(), workflow_id="wf-A", version=2)
    assert wf.workflow_id == "wf-A"
    assert wf.workflow_version == 2
    assert len(wf.nodes) == 3


def test_load_inline_unwraps_definition_wrapper() -> None:
    wrapped = {
        "workflow_id": "wf-B",
        "version": 3,
        "name": "Channel Assistant",
        "definition": _minimal(),
    }
    wf = WorkflowLoader.load_inline(wrapped)
    assert wf.workflow_id == "wf-B"
    assert wf.workflow_version == 3
    assert wf.name == "Channel Assistant"


def test_load_inline_unwraps_workflow_wrapper() -> None:
    wrapped = {"workflow_id": "wf-C", "version": 1, "workflow": _minimal()}
    wf = WorkflowLoader.load_inline(wrapped)
    assert wf.workflow_id == "wf-C"


def test_load_inline_rejects_payload_without_id() -> None:
    with pytest.raises(CompilationError, match="workflow_id"):
        WorkflowLoader.load_inline(_minimal())


def test_load_inline_rejects_unrecognized_shape() -> None:
    with pytest.raises(CompilationError, match="missing"):
        WorkflowLoader.load_inline({"foo": "bar"}, workflow_id="x")


def test_load_inline_explicit_version_wins_over_metadata() -> None:
    payload = {"workflow_id": "wf-D", "version": 9, "definition": _minimal()}
    wf = WorkflowLoader.load_inline(payload, version=5)
    assert wf.workflow_version == 5


def test_load_inline_invalid_version_raises() -> None:
    payload = {"workflow_id": "wf-E", "version": "not-a-number", "definition": _minimal()}
    with pytest.raises(CompilationError, match="invalid workflow_version"):
        WorkflowLoader.load_inline(payload)


# --- Metadata API source (httpx mock) -------------------------------------


def _mock_transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_metadata_api_source_fetches_and_returns_payload() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "workflow_id": "wf-API",
                "version": 7,
                "definition": _minimal(),
            },
        )

    async with _mock_transport(handler) as http:
        src = MetadataApiWorkflowSource("http://meta.local", http_client=http)
        payload = await src.fetch("wf-API", version=7)
    assert "workflow_id" in payload
    assert "definition" in payload
    assert "version=7" in captured["url"]


@pytest.mark.asyncio
async def test_metadata_api_source_404_raises_workflow_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no"})

    async with _mock_transport(handler) as http:
        src = MetadataApiWorkflowSource("http://meta.local", http_client=http)
        with pytest.raises(WorkflowNotFoundError):
            await src.fetch("missing")


@pytest.mark.asyncio
async def test_metadata_api_source_5xx_raises_configuration_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    async with _mock_transport(handler) as http:
        src = MetadataApiWorkflowSource("http://meta.local", http_client=http)
        with pytest.raises(ConfigurationError):
            await src.fetch("wf-API")


@pytest.mark.asyncio
async def test_loader_uses_metadata_api_when_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "workflow_id": "wf-via-api",
                "version": 4,
                "name": "Routed Through API",
                "definition": _minimal(),
            },
        )

    settings = Settings(
        METADATA_API_ENABLED=True,
        METADATA_API_BASE_URL="http://meta.local",
    )
    async with _mock_transport(handler) as http:
        loader = WorkflowLoader(settings, mongo=None, http_client=http)
        wf = await loader.load_by_id("wf-via-api", version=4)
    assert wf.workflow_id == "wf-via-api"
    assert wf.workflow_version == 4
    assert wf.name == "Routed Through API"


# --- Mongo source (integration; auto-skips when Mongo unreachable) --------


@pytest.mark.asyncio
async def test_mongo_source_reads_seeded_definition(mongo_db) -> None:
    seed = {
        "workflow_id": "wf-local-1",
        "workflow_version": 2,
        "name": "Seeded",
        **_minimal(),
    }
    await mongo_db[LOCAL_DEFINITIONS_COLLECTION].insert_one(seed)

    src = MongoWorkflowSource(mongo_db)
    payload = await src.fetch("wf-local-1")
    assert payload["workflow_id"] == "wf-local-1"
    assert payload["workflow_version"] == 2
    assert "_id" not in payload  # stripped


@pytest.mark.asyncio
async def test_mongo_source_returns_highest_version_when_unspecified(mongo_db) -> None:
    base = _minimal()
    for v in (1, 3, 2):
        await mongo_db[LOCAL_DEFINITIONS_COLLECTION].insert_one(
            {"workflow_id": "wf-multi", "workflow_version": v, **base}
        )
    src = MongoWorkflowSource(mongo_db)
    payload = await src.fetch("wf-multi")
    assert payload["workflow_version"] == 3


@pytest.mark.asyncio
async def test_mongo_source_404_raises_workflow_not_found(mongo_db) -> None:
    src = MongoWorkflowSource(mongo_db)
    with pytest.raises(WorkflowNotFoundError):
        await src.fetch("nope")
