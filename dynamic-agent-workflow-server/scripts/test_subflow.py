"""End-to-end demonstration of a sub-flow.

Two workflows:

  * CHILD ``intent_classifier``: start → llm → output
    Receives ``{userQuery}`` in its ``system.*``, returns ``{category, confidence}``.

  * PARENT ``categorize_query``: start → sub_flow → output
    The sub_flow node launches the child synchronously, then the parent's
    output node exposes ``intent`` (the child's category) + ``echo`` (the
    original query) as its final_output.

Flow:
  1. POST /api/workflows/register-inline   (register CHILD in cache)
  2. POST /api/workflows/run-inline        (run PARENT — sub_flow node fires the child)
  3. GET  /api/runs/{parent_run_id}        (inspect the parent's row)
"""
from __future__ import annotations

import json
import sys

import httpx

SERVER = "http://localhost:8765"


CHILD = {
    "workflow_id": "intent_classifier",
    "workflow_version": 1,
    "nodes": [
        {"id": "s", "type": "start", "name": "s"},
        {
            "id": "l",
            "type": "llm",
            "name": "classify",
            "config": {
                "model_id": "mock-fast",
                "messages": [
                    {
                        "role": "system",
                        "content": "Classify the user's intent into a category.",
                    },
                    {"role": "user", "content": "{{system.userQuery}}"},
                ],
                "response_format": "json",
            },
        },
        {
            "id": "o",
            "type": "output",
            "name": "out",
            "config": {
                "outputMappings": {
                    "category": "{{nodes.classify.result.parsed_json.echo}}",
                    "confidence": 0.92,
                    "echoed_query": "{{system.userQuery}}",
                }
            },
        },
    ],
    "edges": [
        {"id": "e1", "source": "s", "target": "l", "sourceHandle": "out"},
        {"id": "e2", "source": "l", "target": "o", "sourceHandle": "out"},
    ],
}


PARENT = {
    "workflow_id": "categorize_query",
    "workflow_version": 1,
    "nodes": [
        {"id": "s", "type": "start", "name": "s"},
        {
            "id": "sf",
            "type": "sub_flow",
            "name": "classify_step",
            "config": {
                "workflow_id": "intent_classifier",
                "input": {
                    # Map parent's system.userQuery into the child's seed input.
                    "userQuery": "{{system.userQuery}}",
                },
            },
        },
        {
            "id": "o",
            "type": "output",
            "name": "out",
            "config": {
                # The child's final_output is nested under the sub_flow node's result.
                "outputMappings": {
                    "intent": "{{nodes.classify_step.result.final_output.category}}",
                    "child_run_id": "{{nodes.classify_step.result.run_id}}",
                    "echo": "{{system.userQuery}}",
                }
            },
        },
    ],
    "edges": [
        {"id": "e1", "source": "s", "target": "sf", "sourceHandle": "out"},
        {"id": "e2", "source": "sf", "target": "o", "sourceHandle": "out"},
    ],
}


def main() -> int:
    with httpx.Client(base_url=SERVER, timeout=30) as client:
        # 1. Register the CHILD definition so the parent's sub_flow node can
        #    find it by workflow_id.
        print("=== 1. Register CHILD workflow ===")
        r = client.post("/api/workflows/register-inline", json={"payload": CHILD})
        r.raise_for_status()
        print(f"   registered: {r.json()['summary']}\n")

        # 2. Run the PARENT.
        print("=== 2. Run PARENT workflow (wait=true) ===")
        r = client.post(
            "/api/workflows/run-inline",
            json={
                "wait": True,
                "input": {"userQuery": "how is the weather in Tokyo today?"},
                "payload": PARENT,
            },
        )
        r.raise_for_status()
        result = r.json()
        print(f"   run_id:        {result['run_id']}")
        print(f"   status:        {result['status']}")
        print(f"   final_output:")
        print(_indent(json.dumps(result.get("final_output"), indent=2), 4))
        parent_run_id = result["run_id"]

        # 3. Inspect the PARENT row in Mongo.
        print("\n=== 3. GET /api/runs/{parent_run_id} ===")
        parent = client.get(f"/api/runs/{parent_run_id}").json()
        print(f"   status:         {parent['status']}")
        visited = sorted(
            ((parent.get("state") or {}).get("variables") or {}).get("nodes", {}).keys()
        )
        print(f"   nodes visited:  {visited}")

        # 4. Find the CHILD run via parent_run_id and inspect.
        child_run_id = (
            (parent.get("state") or {}).get("variables", {})
            .get("nodes", {})
            .get("classify_step", {})
            .get("result", {})
            .get("run_id")
        )
        print(f"\n=== 4. CHILD run row (run_id={child_run_id}) ===")
        if child_run_id:
            child = client.get(f"/api/runs/{child_run_id}").json()
            print(f"   status:         {child['status']}")
            print(f"   parent_run_id:  {child.get('parent_run_id')}")
            print(f"   final_output:")
            print(_indent(json.dumps(child.get("final_output"), indent=2), 4))

    return 0


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
