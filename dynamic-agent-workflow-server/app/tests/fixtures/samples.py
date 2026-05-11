"""Programmatically constructed sample workflows for the integration suite."""
from __future__ import annotations

from typing import Any


def start_llm_output() -> dict[str, Any]:
    return {
        "workflow_id": "sample-llm",
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
            {"id": "e1", "source": "s", "target": "l", "sourceHandle": "out"},
            {"id": "e2", "source": "l", "target": "o", "sourceHandle": "out"},
        ],
    }


def start_rule_branch_output() -> dict[str, Any]:
    return {
        "workflow_id": "sample-rule",
        "nodes": [
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "r",
                "type": "rule",
                "name": "rule",
                "config": {
                    "branches": [
                        {
                            "handle": "weather",
                            "conditions": [
                                {"field": "{{system.category}}", "operator": "equals", "value": "weather"}
                            ],
                        },
                        {
                            "handle": "stocks",
                            "conditions": [
                                {"field": "{{system.category}}", "operator": "equals", "value": "stocks"}
                            ],
                        },
                    ],
                    "default_handle": "else",
                },
            },
            {"id": "w", "type": "output", "name": "w_out", "config": {"outputMappings": {"path": "weather"}}},
            {"id": "k", "type": "output", "name": "k_out", "config": {"outputMappings": {"path": "stocks"}}},
            {"id": "x", "type": "output", "name": "x_out", "config": {"outputMappings": {"path": "fallback"}}},
        ],
        "edges": [
            {"id": "e0", "source": "s", "target": "r", "sourceHandle": "out"},
            {"id": "e1", "source": "r", "target": "w", "sourceHandle": "weather"},
            {"id": "e2", "source": "r", "target": "k", "sourceHandle": "stocks"},
            {"id": "e3", "source": "r", "target": "x", "sourceHandle": "else"},
        ],
    }


def start_humaninput_llm_output() -> dict[str, Any]:
    return {
        "workflow_id": "sample-hi",
        "nodes": [
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "hi",
                "type": "human_input",
                "name": "ask",
                "config": {"question": "What's your name?", "save_to": "system.humanInput"},
            },
            {
                "id": "l",
                "type": "llm",
                "name": "greet",
                "config": {
                    "model_id": "mock-fast",
                    "messages": [
                        {"role": "user", "content": "Greet {{system.humanInput}}"}
                    ],
                },
            },
            {
                "id": "o",
                "type": "output",
                "name": "out",
                "config": {"outputMappings": {"reply": "{{nodes.greet.result.answer}}"}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "s", "target": "hi", "sourceHandle": "out"},
            {"id": "e2", "source": "hi", "target": "l", "sourceHandle": "out"},
            {"id": "e3", "source": "l", "target": "o", "sourceHandle": "out"},
        ],
    }


def start_approval_branches() -> dict[str, Any]:
    return {
        "workflow_id": "sample-ap",
        "nodes": [
            {"id": "s", "type": "start", "name": "s"},
            {"id": "ap", "type": "approval", "name": "ap", "config": {"summary": "Proceed?"}},
            {"id": "ok", "type": "output", "name": "ok", "config": {"outputMappings": {"path": "approved"}}},
            {"id": "no", "type": "output", "name": "no", "config": {"outputMappings": {"path": "rejected"}}},
        ],
        "edges": [
            {"id": "e1", "source": "s", "target": "ap", "sourceHandle": "out"},
            {"id": "e2", "source": "ap", "target": "ok", "sourceHandle": "approved"},
            {"id": "e3", "source": "ap", "target": "no", "sourceHandle": "rejected"},
        ],
    }


def start_http_output() -> dict[str, Any]:
    return {
        "workflow_id": "sample-http",
        "nodes": [
            {"id": "s", "type": "start", "name": "s"},
            {
                "id": "h",
                "type": "http_request",
                "name": "fetch",
                "config": {"url": "http://api.local/echo", "method": "GET"},
            },
            {
                "id": "o",
                "type": "output",
                "name": "out",
                "config": {"outputMappings": {"status": "{{nodes.fetch.result.status}}"}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "s", "target": "h", "sourceHandle": "out"},
            {"id": "e2", "source": "h", "target": "o", "sourceHandle": "out"},
        ],
    }
