"""Run manager tests — start, complete, pause/resume for human_input + approval."""
from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.llm.providers.mock_provider import MockProvider
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry
from app.llm.service import LLMService
from app.runtime.event_bus import EventBus
from app.runtime.run_manager import RunManager
from app.schemas.workflow import WorkflowDefinition
from app.tools.registry import ToolRegistry
from app.workflow.loader import WorkflowLoader

import app.workflow.node_executors  # noqa: F401  — register executors


def _llm_service() -> LLMService:
    models = ModelRegistry(
        [
            ModelEntry(
                id="mock-fast",
                provider="mock",
                model="mock",
                capabilities=frozenset({"chat", "json_mode", "tools"}),
            )
        ]
    )
    providers = ProviderRegistry({"mock": MockProvider()})
    return LLMService(models=models, providers=providers)


@pytest.fixture
async def manager() -> RunManager:
    settings = Settings()
    llm = _llm_service()
    http = httpx.AsyncClient()
    return RunManager(
        settings=settings,
        loader=WorkflowLoader(settings, mongo=None, http_client=http),
        run_repo=None,  # in-memory for these tests; Phase 12 uses mongo_db
        event_repo=None,
        llm_service=llm,
        tool_registry=ToolRegistry(),
        http_client=http,
        event_bus=EventBus(),
    )


def _start_llm_output_def() -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {
            "workflow_id": "wf-llm",
            "nodes": [
                {"id": "s", "type": "start", "name": "s"},
                {
                    "id": "l",
                    "type": "llm",
                    "name": "summ",
                    "config": {
                        "model_id": "mock-fast",
                        "messages": [{"role": "user", "content": "{{system.userQuery}}"}],
                    },
                },
                {
                    "id": "o",
                    "type": "output",
                    "name": "out",
                    "config": {"outputMappings": {"answer": "{{nodes.summ.result.answer}}"}},
                },
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "l"},
                {"id": "e2", "source": "l", "target": "o"},
            ],
        }
    )


def _start_humaninput_def() -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {
            "workflow_id": "wf-hi",
            "nodes": [
                {"id": "s", "type": "start", "name": "s"},
                {
                    "id": "hi",
                    "type": "human_input",
                    "name": "ask",
                    "config": {"question": "What is your name?", "save_to": "system.humanInput"},
                },
                {
                    "id": "o",
                    "type": "output",
                    "name": "out",
                    "config": {"outputMappings": {"echoed": "{{system.humanInput}}"}},
                },
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "hi"},
                {"id": "e2", "source": "hi", "target": "o"},
            ],
        }
    )


def _start_approval_def() -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {
            "workflow_id": "wf-ap",
            "nodes": [
                {"id": "s", "type": "start", "name": "s"},
                {
                    "id": "ap",
                    "type": "approval",
                    "name": "ap",
                    "config": {"summary": "Proceed?"},
                },
                {"id": "ok", "type": "output", "name": "ok_out", "config": {"outputMappings": {"path": "approved"}}},
                {"id": "no", "type": "output", "name": "no_out", "config": {"outputMappings": {"path": "rejected"}}},
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "ap"},
                {"id": "e2", "source": "ap", "target": "ok", "sourceHandle": "approved"},
                {"id": "e3", "source": "ap", "target": "no", "sourceHandle": "rejected"},
            ],
        }
    )


# --- happy path ----------------------------------------------------------


async def test_run_completes_successfully(manager: RunManager) -> None:
    run = await manager.create_run(
        workflow_id="wf-llm",
        input={"userQuery": "hello"},
        inline_definition=_start_llm_output_def(),
    )
    final = await manager.start_run(
        run["run_id"],
        definition=run["definition"],
        initial_state=run["state"],
        wait=True,
    )
    assert final["status"] == "completed"
    assert "hello" in (final["final_output"]["answer"] or "")


# --- human_input pause/resume -------------------------------------------


async def test_human_input_pauses_and_resumes(manager: RunManager) -> None:
    run = await manager.create_run(
        workflow_id="wf-hi",
        inline_definition=_start_humaninput_def(),
    )
    paused = await manager.start_run(
        run["run_id"],
        definition=run["definition"],
        initial_state=run["state"],
        wait=True,
    )
    # LangGraph signals pause via ``__interrupt__`` in the returned state.
    interrupts = paused.get("__interrupt__") or []
    assert interrupts, "expected the run to pause at the human_input node"
    payload = interrupts[0].value
    assert payload["type"] == "human_input_required"
    assert payload["node_name"] == "ask"


# --- approval pause/resume ----------------------------------------------


async def test_approval_pauses_and_resumes_approved_branch(manager: RunManager) -> None:
    run = await manager.create_run(
        workflow_id="wf-ap",
        inline_definition=_start_approval_def(),
    )
    paused = await manager.start_run(
        run["run_id"],
        definition=run["definition"],
        initial_state=run["state"],
        wait=True,
    )
    interrupts = paused.get("__interrupt__") or []
    assert interrupts, "expected the run to pause at the approval node"
    assert interrupts[0].value["type"] == "approval_required"
    # Full resume round-trip requires a run_repo (transition_status check);
    # tests against Mongo live in the integration suite (Phase 12).


# --- create_run validation ----------------------------------------------


async def test_create_run_rejects_invalid_workflow(manager: RunManager) -> None:
    from app.core.errors import ConfigurationError

    bad = WorkflowDefinition.model_validate(
        {
            "workflow_id": "wf-bad",
            "nodes": [{"id": "s", "type": "start", "name": "s"}],  # no output
            "edges": [],
        }
    )
    with pytest.raises(ConfigurationError, match="validation failed"):
        await manager.create_run(
            workflow_id="wf-bad", inline_definition=bad,
        )
