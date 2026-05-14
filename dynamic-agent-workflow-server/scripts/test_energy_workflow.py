"""End-to-end test of the energy-analysis-workflow.json against the running server.

The frontend's JSON uses a React Flow wrapper shape (top-level ``type: "dynamic"``,
real fields under ``data.*``) plus two node types we register as aliases
(``uiView``, ``http``). This script:

  1. Loads the workflow JSON.
  2. Normalizes the wrapper shape into our flat (``id/type/name/config``) schema.
  3. Swaps the real OpenAI model display names for ``mock-fast`` so we can run
     without API keys.
  4. Rewrites the LLM node's config into the canonical ``messages`` form.
  5. POSTs to ``/api/workflows/compile-inline`` (validation report).
  6. POSTs to ``/api/workflows/run-inline?wait=true`` and prints the result.

Run while uvicorn is up on port 8765::

    uvicorn app.main:app --port 8765 --log-level warning
    python scripts/test_energy_workflow.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx

SERVER = "http://localhost:8765"
WORKFLOW_PATH = Path(__file__).resolve().parents[2] / "energy-analysis-workflow.json"


def normalize(definition: dict) -> dict:
    """Translate the frontend's React Flow shape into the runtime's schema."""
    out = copy.deepcopy(definition)

    # Top-level: id → workflow_id, version → workflow_version.
    if "workflow_id" not in out and "id" in out:
        out["workflow_id"] = out.pop("id")
    if "workflow_version" not in out and "version" in out:
        out["workflow_version"] = out.pop("version")

    # Per-node: hoist data.{name,type,config} to the top level.
    for n in out.get("nodes", []):
        data = n.get("data") or {}
        n["name"] = data.get("name") or n.get("name") or n["id"]
        n["type"] = data.get("type") or n.get("type")
        n["config"] = data.get("config") or {}
        n["description"] = data.get("description")

        cfg = n["config"]
        # Swap the real model display names for our mock.
        if "model" in cfg and "model_id" not in cfg:
            cfg["model_id"] = "mock-fast"

        # LLM node: build canonical messages from `instructions` + the
        # frontend's `messages: [{fields: [...]}]` shape.
        if n["type"] == "llm":
            cfg["messages"] = _build_llm_messages(cfg)
            cfg["response_format"] = "json"
        # Agent node: map handoffs → handoff_handles + adjust prompt fields.
        if n["type"] == "agent":
            cfg.setdefault("system_prompt", cfg.get("instructions", ""))
            cfg.setdefault("user_template", cfg.get("userQuery", "{{system.userQuery}}"))
            cfg["handoff_handles"] = [h.get("id") for h in cfg.get("handoffs", []) if h.get("id")]

    return out


def _build_llm_messages(cfg: dict) -> list[dict]:
    out: list[dict] = []
    instructions = cfg.get("instructions")
    if instructions:
        out.append({"role": "system", "content": instructions})
    for m in cfg.get("messages") or []:
        content = m.get("content") or ""
        if not content and m.get("fields"):
            content = "\n".join(
                f"{f.get('label')}: {f.get('value')}" for f in m["fields"]
            )
        out.append({"role": m.get("role", "user"), "content": content})
    if not any(msg.get("role") == "user" for msg in out):
        out.append({"role": "user", "content": "{{system.userQuery}}"})
    return out


def main() -> int:
    raw = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    print(f"Loaded: {WORKFLOW_PATH.name}")
    print(f"  workflow id:   {raw.get('id')}")
    print(f"  version:       {raw.get('version')}")
    print(f"  nodes:         {len(raw.get('nodes', []))}")
    print(f"  edges:         {len(raw.get('edges', []))}")

    normalized = normalize(raw)
    print()
    print("After normalization:")
    for n in normalized["nodes"]:
        print(f"  {n['id']:25s} type={n['type']:10s} name={n['name']}")

    with httpx.Client(base_url=SERVER, timeout=30) as client:
        print()
        print("=== POST /api/workflows/compile-inline ===")
        resp = client.post("/api/workflows/compile-inline", json={"payload": normalized})
        resp.raise_for_status()
        report = resp.json()
        v = report["validation"]
        print(f"  is_valid:  {v['is_valid']}")
        print(f"  errors:    {len(v['errors'])}")
        print(f"  warnings:  {len(v['warnings'])}")
        for w in v["warnings"][:5]:
            print(f"    [warn] {w['code']}: {w['message']}")

        if not v["is_valid"]:
            print("  errors:")
            for e in v["errors"]:
                print(f"    [err]  {e['code']}: {e['message']}")
            return 1

        print()
        print("=== POST /api/workflows/run-inline (wait=true) ===")
        run_resp = client.post(
            "/api/workflows/run-inline",
            json={
                "wait": True,
                "input": {
                    "userQuery": "Show me daily sum and average energy data from May 1 to May 12, limit 10",
                    "conversationHistory": "",
                    "attachments": [],
                },
                "payload": normalized,
            },
            timeout=120,
        )
        run_resp.raise_for_status()
        result = run_resp.json()
        print(f"  run_id:        {result['run_id']}")
        print(f"  status:        {result['status']}")
        if result.get("pause"):
            print(f"  pause:         {json.dumps(result['pause'], indent=2)}")
        print(f"  final_output:")
        print(json.dumps(result.get("final_output"), indent=2)[:1500])

        print()
        print("=== GET /api/runs/{id} (persisted run row) ===")
        row = client.get(f"/api/runs/{result['run_id']}").json()
        print(f"  status:         {row.get('status')}")
        print(f"  started_at:     {row.get('started_at')}")
        print(f"  completed_at:   {row.get('completed_at')}")
        print(f"  step_count:     {(row.get('state') or {}).get('step_count')}")
        visited = sorted(((row.get("state") or {}).get("variables") or {}).get("nodes", {}).keys())
        print(f"  nodes visited:  {visited}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
