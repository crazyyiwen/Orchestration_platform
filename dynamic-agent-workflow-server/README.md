# Dynamic Agent Workflow Server

Runtime server that dynamically loads workflow JSON (produced by the React Workflow
Builder), validates it, compiles it into a [LangGraph](https://github.com/langchain-ai/langgraph)
graph, and executes it. No workflow-specific code lives in this server — every node type
is handled by a generic registry-driven executor, and topology comes purely from data.

> Status: **Phase 1 — Foundation only**. Health endpoint and config are wired up. Mongo,
> validator, compiler, and runtime arrive in subsequent phases. See the architecture plan
> for the full roadmap.

## Tech stack

- Python 3.11+ / FastAPI
- LangGraph (dynamic compilation, native interrupts for pause/resume)
- LangChain core abstractions where useful
- MongoDB via PyMongo Async (runs, events, checkpoints, compile cache)
- Langfuse (optional tracing)
- Pydantic v2 + pydantic-settings
- httpx, sse-starlette
- pytest

## Quickstart (Phase 1)

```bash
cd dynamic-agent-workflow-server
python -m venv .venv
.\.venv\Scripts\activate          # PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env             # macOS/Linux: cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"dynamic-agent-workflow-server","env":"local","version":"0.1.0"}
```

## Project layout (target)

The full file tree is described in the architecture plan. Phase 1 creates only:

```
dynamic-agent-workflow-server/
├── app/
│   ├── main.py              # FastAPI factory + lifespan
│   ├── core/
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   ├── logging.py       # Structured logging setup
│   │   ├── errors.py        # Domain error hierarchy
│   │   └── security.py      # sanitize_error / provider-key visibility
│   └── api/
│       ├── deps.py
│       └── routes/
│           └── health.py
├── pyproject.toml
├── .env.example
└── README.md
```

## Configuration

All runtime knobs live in environment variables; see [.env.example](.env.example) for the
complete list and defaults. Notable gates:

| Env var | Default | Purpose |
|---|---|---|
| `ENABLE_SCRIPT_NODE` | `false` | Script node is disabled by default; never run untrusted code in-process |
| `ALLOW_EXTERNAL_HTTP` | `true` | Gates HTTP node + HTTP tool egress |
| `MAX_WORKFLOW_STEPS` | `100` | Wrapper-enforced step ceiling |
| `MAX_AGENT_ITERATIONS` | `8` | Tool-call loop bound |
| `MAX_SUBFLOW_DEPTH` | `3` | Sub Flow recursion limit |
| `NODE_TIMEOUT_SECONDS` | `60` | Per-node `asyncio.wait_for` |
| `WORKFLOW_TIMEOUT_SECONDS` | `600` | Per-run timeout |

## Roadmap

Phase 1 (this PR) → Foundation. Subsequent phases follow the architecture plan:

1. ✅ **Foundation** — FastAPI app, settings, logging, `/health`, CORS
2. Mongo persistence (run / event / checkpoint / compile-cache repos)
3. Workflow schemas + 3-backend loader
4. Validation + graph utilities
5. Variable resolver + state merge semantics
6. LLM provider system + `config/models.yaml`
7. Node executor registry (13 node types)
8. Dynamic LangGraph compiler + Mongo checkpointer
9. Run manager (compile/invoke/pause/resume)
10. Langfuse observability
11. API routes (compile / runs / SSE / resume / human-input / approval)
12. Tests + sample workflows
13. Docs + polish (docker-compose, frontend integration)
