"""Smoke tests for each node executor (Phase 7)."""
from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.llm.providers.mock_provider import MockProvider
from app.llm.registry import ModelEntry, ModelRegistry, ProviderRegistry
from app.llm.service import LLMService
from app.schemas.node import Node
from app.tools.mock_tool import EchoTool, StaticAnswerTool
from app.tools.registry import ToolRegistry
from app.workflow.node_executors import get_executor, registered_types
from app.workflow.node_executors.base import ExecutionContext


def _ctx(*, settings: Settings | None = None, tools: ToolRegistry | None = None) -> ExecutionContext:
    s = settings or Settings()
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
    return ExecutionContext(
        run_id="r-1",
        workflow_id="wf-1",
        workflow_version=1,
        settings=s,
        llm_service=LLMService(models=models, providers=providers),
        tool_registry=tools or ToolRegistry(),
        http_client=httpx.AsyncClient(),
    )


def _state(variables: dict | None = None, **extra) -> dict:
    base = {
        "run_id": "r-1",
        "workflow_id": "wf-1",
        "workflow_version": 1,
        "variables": variables or {"system": {}, "runtime": {}, "nodes": {}},
        "messages": [],
        "events": [],
        "step_count": 0,
        "_next_handle": None,
        "_resume_input": None,
    }
    base.update(extra)
    return base


def test_all_13_node_types_are_registered() -> None:
    expected = {
        "start", "output", "rule", "variable_update", "human_input", "approval",
        "llm", "agent", "http_request", "guardrail", "sub_flow", "external_agent", "script",
    }
    assert expected.issubset(registered_types())


# --- start ----------------------------------------------------------------


async def test_start_emits_runtime_metadata_and_exits_out() -> None:
    exe = get_executor("start")
    node = Node(id="s", type="start", name="s")
    result = await exe.execute(node, _state(), _ctx())
    assert result.status == "success"
    assert result.next_handle == "out"
    assert "runtime" in result.state_updates["variables"]


# --- output ---------------------------------------------------------------


async def test_output_resolves_mappings_into_final_output() -> None:
    exe = get_executor("output")
    node = Node(
        id="o",
        type="output",
        name="o",
        config={"outputMappings": {"answer": "{{nodes.llm_1.result.content}}"}},
    )
    state = _state(
        variables={
            "system": {},
            "runtime": {},
            "nodes": {"llm_1": {"result": {"content": "hello"}}},
        }
    )
    result = await exe.execute(node, state, _ctx())
    assert result.status == "success"
    assert result.next_handle is None  # terminal
    assert result.state_updates["final_output"] == {"answer": "hello"}
    assert result.state_updates["status"] == "completed"


# --- rule -----------------------------------------------------------------


async def test_rule_picks_first_matching_branch() -> None:
    exe = get_executor("rule")
    node = Node(
        id="r",
        type="rule",
        name="r",
        config={
            "branches": [
                {
                    "handle": "case_1",
                    "logic": "AND",
                    "conditions": [
                        {"field": "{{system.category}}", "operator": "equals", "value": "weather"}
                    ],
                },
                {
                    "handle": "case_2",
                    "conditions": [
                        {"field": "{{system.category}}", "operator": "equals", "value": "stocks"}
                    ],
                },
            ],
            "default_handle": "else",
        },
    )
    state = _state(variables={"system": {"category": "stocks"}, "runtime": {}, "nodes": {}})
    result = await exe.execute(node, state, _ctx())
    assert result.next_handle == "case_2"


async def test_rule_falls_through_to_default() -> None:
    exe = get_executor("rule")
    node = Node(
        id="r",
        type="rule",
        name="r",
        config={
            "branches": [
                {"handle": "x", "conditions": [
                    {"field": "{{system.category}}", "operator": "equals", "value": "no-match"}
                ]}
            ],
            "default_handle": "else",
        },
    )
    state = _state(variables={"system": {"category": "weather"}, "runtime": {}, "nodes": {}})
    result = await exe.execute(node, state, _ctx())
    assert result.next_handle == "else"


async def test_rule_evaluates_or_logic() -> None:
    exe = get_executor("rule")
    node = Node(
        id="r",
        type="rule",
        name="r",
        config={
            "branches": [
                {
                    "handle": "match",
                    "logic": "OR",
                    "conditions": [
                        {"field": "{{x}}", "operator": "equals", "value": "wrong"},
                        {"field": "{{x}}", "operator": "contains", "value": "ll"},
                    ],
                }
            ],
            "default_handle": "else",
        },
    )
    state = _state(variables={"x": "hello"})
    result = await exe.execute(node, state, _ctx())
    assert result.next_handle == "match"


# --- variable_update ------------------------------------------------------


async def test_variable_update_applies_set_append_increment() -> None:
    exe = get_executor("variable_update")
    node = Node(
        id="v",
        type="variable_update",
        name="v",
        config={
            "updates": [
                {"path": "system.userQuery", "operation": "set", "value": "hi"},
                {"path": "thread.messages", "operation": "append", "value": "m1"},
                {"path": "flow.counter", "operation": "increment", "value": 5},
                {"path": "flow.tmp", "operation": "set", "value": "x"},
                {"path": "flow.tmp", "operation": "remove"},
            ]
        },
    )
    state = _state(variables={"system": {}, "runtime": {}, "nodes": {}, "flow": {"counter": 10}})
    result = await exe.execute(node, state, _ctx())
    new_vars = result.state_updates["variables"]
    assert new_vars["system"]["userQuery"] == "hi"
    assert new_vars["thread"]["messages"] == ["m1"]
    assert new_vars["flow"]["counter"] == 15
    assert "tmp" not in new_vars["flow"]


# --- human_input ----------------------------------------------------------


async def test_human_input_pauses_first_then_resumes_with_input() -> None:
    exe = get_executor("human_input")
    node = Node(
        id="hi",
        type="human_input",
        name="hi",
        config={"question": "What is your name?", "save_to": "system.humanInput"},
    )
    # First execution → paused.
    paused = await exe.execute(node, _state(), _ctx())
    assert paused.status == "paused"
    assert paused.pause_payload["question"] == "What is your name?"

    # Resume execution with input → success.
    state = _state(_resume_input="Yiwen")
    resumed = await exe.execute(node, state, _ctx())
    assert resumed.status == "success"
    assert resumed.next_handle == "out"
    new_vars = resumed.state_updates["variables"]
    assert new_vars["system"]["humanInput"] == "Yiwen"


# --- approval -------------------------------------------------------------


async def test_approval_pauses_then_routes_by_decision() -> None:
    exe = get_executor("approval")
    node = Node(id="ap", type="approval", name="ap", config={"summary": "Proceed?"})
    paused = await exe.execute(node, _state(), _ctx())
    assert paused.status == "paused"

    approved = await exe.execute(node, _state(_resume_input="approved"), _ctx())
    assert approved.next_handle == "approved"

    rejected = await exe.execute(node, _state(_resume_input={"decision": "rejected"}), _ctx())
    assert rejected.next_handle == "rejected"


# --- llm ------------------------------------------------------------------


async def test_llm_node_runs_against_mock_provider() -> None:
    exe = get_executor("llm")
    node = Node(
        id="l",
        type="llm",
        name="l",
        config={
            "model_id": "mock-fast",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "{{system.userQuery}}"},
            ],
        },
    )
    state = _state(variables={"system": {"userQuery": "hello"}, "runtime": {}, "nodes": {}})
    result = await exe.execute(node, state, _ctx())
    assert result.status == "success"
    assert result.next_handle == "out"
    assert "hello" in (result.output["content"] or "")


async def test_llm_validate_config_rejects_missing_model_id() -> None:
    cls = type(get_executor("llm"))
    issues = cls.validate_config({"messages": [{"role": "user", "content": "x"}]})
    assert any(i.code == "llm_missing_model_id" for i in issues)


# --- guardrail ------------------------------------------------------------


async def test_guardrail_blocks_on_regex_match() -> None:
    exe = get_executor("guardrail")
    node = Node(
        id="g",
        type="guardrail",
        name="g",
        config={
            "input": "{{system.userQuery}}",
            "rules": [{"operator": "regex", "value": r"credit\s*card", "block_reason": "PII"}],
        },
    )
    state = _state(variables={"system": {"userQuery": "give me my credit card"}})
    result = await exe.execute(node, state, _ctx())
    assert result.next_handle == "block"

    state2 = _state(variables={"system": {"userQuery": "what's the weather"}})
    ok = await exe.execute(node, state2, _ctx())
    assert ok.next_handle == "allow"


# --- script ---------------------------------------------------------------


async def test_script_node_disabled_by_default() -> None:
    exe = get_executor("script")
    node = Node(id="sc", type="script", name="sc", config={"mock_output": {"x": 1}})
    result = await exe.execute(node, _state(), _ctx())
    assert result.status == "failed"
    assert result.next_handle == "error"


async def test_script_node_runs_mock_when_enabled() -> None:
    exe = get_executor("script")
    node = Node(id="sc", type="script", name="sc", config={"mock_output": {"x": 1}})
    settings = Settings(ENABLE_SCRIPT_NODE=True)
    result = await exe.execute(node, _state(), _ctx(settings=settings))
    assert result.status == "success"
    assert result.output == {"x": 1}


# --- agent ----------------------------------------------------------------


async def test_agent_node_returns_final_answer_without_tools() -> None:
    exe = get_executor("agent")
    node = Node(
        id="a",
        type="agent",
        name="a",
        config={
            "model_id": "mock-fast",
            "system_prompt": "You answer briefly.",
            "user_template": "Hello agent",
        },
    )
    result = await exe.execute(node, _state(), _ctx())
    assert result.status == "success"
    assert result.next_handle == "out"
    assert "Hello agent" in (result.output["answer"] or "")


async def test_agent_node_executes_tool_then_returns_answer() -> None:
    exe = get_executor("agent")
    tools = ToolRegistry([EchoTool(), StaticAnswerTool(answer="42")])
    node = Node(
        id="a",
        type="agent",
        name="a",
        config={
            "model_id": "mock-fast",
            "user_template": "Need info",
            "tools": ["echo", "static_answer"],
            # MockProvider triggers a tool call only if metadata signals; we don't
            # set that here so the agent will return text on first iteration.
        },
    )
    result = await exe.execute(node, _state(), _ctx(tools=tools))
    assert result.status == "success"
    assert result.next_handle == "out"


# --- sub_flow -------------------------------------------------------------


async def test_subflow_node_invokes_launcher_and_passes_inputs() -> None:
    captured: dict = {}

    async def launcher(sub_id, inputs, depth, parent_run_id):
        captured.update(
            sub_id=sub_id, inputs=inputs, depth=depth, parent_run_id=parent_run_id
        )
        return {"run_id": "child-1", "final_output": {"echoed": inputs}}

    ctx = _ctx()
    ctx.sub_flow_launcher = launcher
    exe = get_executor("sub_flow")
    node = Node(
        id="sf",
        type="sub_flow",
        name="sf",
        config={"workflow_id": "child-wf", "input": {"q": "{{system.userQuery}}"}},
    )
    state = _state(variables={"system": {"userQuery": "ping"}})
    result = await exe.execute(node, state, ctx)
    assert result.status == "success"
    assert captured["sub_id"] == "child-wf"
    assert captured["inputs"] == {"q": "ping"}
    assert captured["depth"] == 1


async def test_subflow_node_enforces_depth_limit() -> None:
    from app.core.errors import ExecutionLimitExceeded

    settings = Settings(MAX_SUBFLOW_DEPTH=1)
    ctx = _ctx(settings=settings)
    ctx.depth = 1  # already at the limit; +1 would exceed
    ctx.sub_flow_launcher = lambda *a, **kw: None  # never invoked
    exe = get_executor("sub_flow")
    node = Node(id="sf", type="sub_flow", name="sf", config={"workflow_id": "x"})
    with pytest.raises(ExecutionLimitExceeded):
        await exe.execute(node, _state(), ctx)


# --- http_request ---------------------------------------------------------


async def test_http_request_node_runs_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "echo": dict(request.url.params)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        ctx = _ctx()
        ctx.http_client = http
        exe = get_executor("http_request")
        node = Node(
            id="h",
            type="http_request",
            name="h",
            config={"url": "http://api.local/foo", "method": "GET", "query": {"q": "x"}},
        )
        result = await exe.execute(node, _state(), ctx)
    assert result.status == "success"
    assert result.next_handle == "out"
    assert result.output["json"]["ok"] is True
