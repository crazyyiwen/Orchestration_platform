"""End-to-end LangGraph compiler tests (Phase 8)."""
from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.langgraph_runtime.checkpointing import make_checkpointer
from app.langgraph_runtime.graph_builder import compile_workflow
from app.llm.providers.mock_provider import MockProvider
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry
from app.llm.service import LLMService
from app.schemas.workflow import WorkflowDefinition
from app.tools.registry import ToolRegistry
from app.workflow.node_executors.base import ExecutionContext
from app.workflow.state import empty_runtime_state

# Ensure executors register on import.
import app.workflow.node_executors  # noqa: F401


def _services():
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
    return LLMService(models=models, providers=providers), ToolRegistry()


def _ctx_factory(settings: Settings | None = None):
    s = settings or Settings()
    llm, tools = _services()

    def factory():
        return ExecutionContext(
            run_id="r-1",
            workflow_id="wf-1",
            workflow_version=1,
            settings=s,
            llm_service=llm,
            tool_registry=tools,
            http_client=httpx.AsyncClient(),
        )

    return factory


def _make_wf(workflow_id: str, nodes: list[dict], edges: list[dict]) -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {"workflow_id": workflow_id, "nodes": nodes, "edges": edges}
    )


# ---- Start → LLM → Output (the canonical happy path) -------------------


async def test_start_llm_output_e2e_with_mock_provider() -> None:
    wf = _make_wf(
        "wf-llm",
        nodes=[
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "l",
                "type": "llm",
                "name": "summarizer",
                "config": {
                    "model_id": "mock-fast",
                    "messages": [
                        {"role": "user", "content": "{{system.userQuery}}"}
                    ],
                },
            },
            {
                "id": "o",
                "type": "output",
                "name": "out",
                "config": {
                    "outputMappings": {"answer": "{{nodes.summarizer.result.answer}}"}
                },
            },
        ],
        edges=[
            {"id": "e1", "source": "s", "target": "l", "sourceHandle": "out"},
            {"id": "e2", "source": "l", "target": "o", "sourceHandle": "out"},
        ],
    )

    graph = compile_workflow(
        wf,
        context_factory=_ctx_factory(),
        checkpointer=make_checkpointer(),
    )

    initial = empty_runtime_state(run_id="r-1", workflow_id="wf-llm", workflow_version=1)
    initial["variables"]["system"]["userQuery"] = "hello there"

    final = await graph.ainvoke(
        initial, config={"configurable": {"thread_id": "r-1"}}
    )
    assert final["status"] == "completed"
    assert "hello there" in (final["final_output"]["answer"] or "")
    assert final["step_count"] == 3


# ---- Start → Rule → branch routing -------------------------------------


async def test_rule_branch_routes_via_next_handle() -> None:
    wf = _make_wf(
        "wf-rule",
        nodes=[
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "r",
                "type": "rule",
                "name": "r",
                "config": {
                    "branches": [
                        {
                            "handle": "case_weather",
                            "conditions": [
                                {"field": "{{system.category}}", "operator": "equals", "value": "weather"}
                            ],
                        },
                        {
                            "handle": "case_stocks",
                            "conditions": [
                                {"field": "{{system.category}}", "operator": "equals", "value": "stocks"}
                            ],
                        },
                    ],
                    "default_handle": "else",
                },
            },
            {"id": "o1", "type": "output", "name": "weather_out", "config": {"outputMappings": {"path": "weather"}}},
            {"id": "o2", "type": "output", "name": "stocks_out", "config": {"outputMappings": {"path": "stocks"}}},
            {"id": "o3", "type": "output", "name": "fallback_out", "config": {"outputMappings": {"path": "fallback"}}},
        ],
        edges=[
            {"id": "e0", "source": "s", "target": "r", "sourceHandle": "out"},
            {"id": "e1", "source": "r", "target": "o1", "sourceHandle": "case_weather"},
            {"id": "e2", "source": "r", "target": "o2", "sourceHandle": "case_stocks"},
            {"id": "e3", "source": "r", "target": "o3", "sourceHandle": "else"},
        ],
    )

    graph = compile_workflow(wf, context_factory=_ctx_factory(), checkpointer=make_checkpointer())

    # Path 1: stocks → o2
    initial = empty_runtime_state(run_id="r-stocks", workflow_id="wf-rule", workflow_version=1)
    initial["variables"]["system"]["category"] = "stocks"
    final = await graph.ainvoke(initial, config={"configurable": {"thread_id": "r-stocks"}})
    assert final["final_output"]["path"] == "stocks"

    # Path 2: weather → o1
    initial2 = empty_runtime_state(run_id="r-weather", workflow_id="wf-rule", workflow_version=1)
    initial2["variables"]["system"]["category"] = "weather"
    final2 = await graph.ainvoke(initial2, config={"configurable": {"thread_id": "r-weather"}})
    assert final2["final_output"]["path"] == "weather"

    # Path 3: unknown → else → o3
    initial3 = empty_runtime_state(run_id="r-fb", workflow_id="wf-rule", workflow_version=1)
    initial3["variables"]["system"]["category"] = "music"
    final3 = await graph.ainvoke(initial3, config={"configurable": {"thread_id": "r-fb"}})
    assert final3["final_output"]["path"] == "fallback"


# ---- Compile cache -----------------------------------------------------


def test_compile_cache_returns_same_instance_for_unchanged_definition() -> None:
    from app.langgraph_runtime.compile_cache import CompileCache

    wf = _make_wf(
        "wf-cache",
        nodes=[
            {"id": "s", "type": "start", "name": "s"},
            {"id": "o", "type": "output", "name": "o", "config": {"outputMappings": {"x": 1}}},
        ],
        edges=[{"id": "e", "source": "s", "target": "o", "sourceHandle": "out"}],
    )
    cache = CompileCache()
    g1 = cache.get_or_compile(wf, context_factory=_ctx_factory())
    g2 = cache.get_or_compile(wf, context_factory=_ctx_factory())
    assert g1 is g2

    cache.invalidate(wf)
    g3 = cache.get_or_compile(wf, context_factory=_ctx_factory())
    assert g3 is not g1


# ---- MAX_WORKFLOW_STEPS guard -----------------------------------------


async def test_max_workflow_steps_guard_aborts_runaway() -> None:
    """Two-node ping-pong cycle: validate it raises before unbounded run."""
    from app.core.errors import ExecutionLimitExceeded

    wf = _make_wf(
        "wf-loop",
        nodes=[
            {"id": "s", "type": "start", "name": "s"},
            {"id": "a", "type": "rule", "name": "a", "config": {"branches": [
                {"handle": "loop", "conditions": []}
            ], "default_handle": "loop"}},
            {"id": "o", "type": "output", "name": "o"},  # never reached
        ],
        edges=[
            {"id": "e1", "source": "s", "target": "a"},
            # rule routes to itself forever via "loop"
            {"id": "e2", "source": "a", "target": "a", "sourceHandle": "loop"},
        ],
    )
    settings = Settings(MAX_WORKFLOW_STEPS=10)
    graph = compile_workflow(
        wf, context_factory=_ctx_factory(settings=settings), checkpointer=make_checkpointer()
    )
    initial = empty_runtime_state(run_id="r-loop", workflow_id="wf-loop", workflow_version=1)
    with pytest.raises(ExecutionLimitExceeded):
        await graph.ainvoke(initial, config={"configurable": {"thread_id": "r-loop"}, "recursion_limit": 50})
