# Dynamic Agent Workflow Server

Runtime server that loads workflow JSON (produced by the React Workflow
Builder), validates it, compiles it into a [LangGraph](https://github.com/langchain-ai/langgraph)
graph **dynamically**, and executes it. No workflow-specific code lives in
this server — every node type is handled by a generic registry-driven
executor, and topology comes purely from data.

```
Frontend workflow JSON  →  Loader  →  Validator  →  Dynamic LangGraph Compiler
                                                            ↓
                       Generic Node Wrapper  ←  Node Executor Registry
                                                            ↓
                    LLM Service / Tool Registry / HTTP / Sub Flow
                                                            ↓
            MongoDB run state  +  Langfuse traces  +  SSE event stream
```

## Tech stack

- Python 3.11+, FastAPI, Uvicorn
- LangGraph 1.x (native interrupts for pause/resume)
- PyMongo async (runs, events, checkpoints, compile cache)
- Pydantic v2 / pydantic-settings
- httpx-based LLM providers (no SDK dependencies)
- Langfuse (optional tracing) / sse-starlette (event streaming)

## Quickstart

```bash
# 1. Start MongoDB
docker compose up -d mongo

# 2. Install + run
cd dynamic-agent-workflow-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env                    # macOS/Linux: cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

```bash
# 3. Smoke test
curl http://localhost:8000/health
# {"status":"ok","service":"dynamic-agent-workflow-server",
#  "env":"local","version":"0.1.0","mongo":"ok"}

# Run a workflow inline (uses the always-on mock LLM provider)
curl -X POST http://localhost:8000/api/workflows/run-inline \
  -H 'Content-Type: application/json' \
  -d '{
    "wait": true,
    "input": {"userQuery": "hello"},
    "payload": {
      "workflow_id": "demo",
      "nodes": [
        {"id":"s","type":"start","name":"s"},
        {"id":"l","type":"llm","name":"summ","config":{
          "model_id":"mock-fast",
          "messages":[{"role":"user","content":"{{system.userQuery}}"}]
        }},
        {"id":"o","type":"output","name":"out","config":{
          "outputMappings":{"answer":"{{nodes.summ.result.answer}}"}
        }}
      ],
      "edges": [
        {"id":"e1","source":"s","target":"l"},
        {"id":"e2","source":"l","target":"o"}
      ]
    }
  }'
```

## Supported node types

13 canonical types per spec §6 — all registered via `@register` and selected at
compile time by `node.type`:

`start` · `llm` · `agent` · `rule` · `script` · `human_input` · `output` ·
`sub_flow` · `variable_update` · `http_request` · `guardrail` · `approval` ·
`external_agent`

Unknown types are tolerated as warnings by the validator (real frontend JSON
uses extras like `tool`/`variable`). Strict mode (`/api/workflows/{id}/validate-runtime`)
gates them against the registered set.

## Variable syntax (spec §5)

```
{{system.userQuery}}                            — input seeded by /runs
{{system.attachments}}                          — etc.
{{runtime.workflowMetaData.workflowId}}         — set by start node
{{nodes.<node-name>.result.<field>}}            — any prior node's result
{{flow.*}} / {{thread.*}} / {{nodeOutput.*}}    — extra namespaces work too
```

Whole-value single refs preserve type (`{{nodes.x.result}}` returns the dict).
Mid-string refs interpolate as strings (`"hi {{system.userQuery}}"`).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | App + Mongo status |
| GET | `/api/models` | Loaded model registry + per-entry availability |
| GET | `/api/providers` | Registered LLM providers + capability flags |
| POST | `/api/models/test` | Round-trip a single chat call |
| POST | `/api/workflows/{id}/compile` | Load + validate by id |
| POST | `/api/workflows/compile-inline` | Validate an inline payload |
| POST | `/api/workflows/{id}/validate-runtime` | Strict (registry-gated) validation |
| POST | `/api/workflows/{id}/runs` | Start a run by id |
| POST | `/api/workflows/run-inline` | Start a run with an inline payload (`wait=true` for sync) |
| GET | `/api/runs/{id}` | Fetch a run row |
| GET | `/api/runs/{id}/state` | Fetch the current state + pause payload |
| GET | `/api/runs/{id}/events` | SSE stream — replay via `?since=N` then live |
| POST | `/api/runs/{id}/cancel` | Cancel an in-flight run |
| POST | `/api/runs/{id}/resume` | Generic resume |
| POST | `/api/runs/{id}/human-input` | Resume a `human_input` pause |
| POST | `/api/runs/{id}/approval` | Resume an `approval` pause (`approved`/`rejected`) |
| GET | `/api/tools` | List registered tools (for Agent nodes) |
| POST | `/api/tools/test` | Invoke a tool directly |
| GET | `/api/runs/{id}/trace-link` | Langfuse URL when enabled |

## Configuration

All knobs live in env vars; see [.env.example](.env.example) for defaults.

| Var | Default | Purpose |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | Mongo connection |
| `METADATA_API_ENABLED` | `false` | Load workflows via metadata API instead of local Mongo |
| `LANGFUSE_ENABLED` | `false` | Toggle Langfuse tracing |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Enable real LLM providers |
| `ENABLE_SCRIPT_NODE` | `false` | Script node is mock-only; do not enable |
| `ALLOW_EXTERNAL_HTTP` | `true` | Gate HTTP node + HTTP tool |
| `MAX_WORKFLOW_STEPS` | `100` | Wrapper-enforced step ceiling |
| `MAX_AGENT_ITERATIONS` | `8` | Agent tool-call loop bound |
| `MAX_SUBFLOW_DEPTH` | `3` | Sub Flow recursion limit |
| `NODE_TIMEOUT_SECONDS` | `60` | Per-node `asyncio.wait_for` |
| `WORKFLOW_TIMEOUT_SECONDS` | `600` | Per-run timeout |

## Model registry — `config/models.yaml`

```yaml
models:
  - id: mock-fast
    provider: mock
    model: mock
    capabilities: [chat, json_mode, tools, streaming]

  - id: gpt-4o
    provider: openai
    model: gpt-4o-2024-08-06
    capabilities: [chat, json_mode, tools, streaming]

  - id: claude-sonnet-4-6
    provider: anthropic
    model: claude-sonnet-4-6
    capabilities: [chat, tools, streaming]

  - id: llama3-local
    provider: ollama
    model: llama3:8b
    capabilities: [chat]
```

Add entries freely — no code change required. The LLM service validates
requests against declared capabilities (so unsupported `tools` or `json_mode`
fails fast at compile time).

## Frontend integration

The frontend should call `POST /api/workflows/run-inline` with the JSON it
already produces (top-level `nodes`, `edges`, `interface`, etc.). Three
wrapper shapes are tolerated by the loader:

  * bare: `{nodes, edges, ...}`
  * `{definition: {...}, workflow_id, version, name}`
  * `{workflow: {...}, ...}`

For pause/resume:

  1. POST with `wait: true` → returns `{status: "paused", pause: {...}}` if the
     run hits a `human_input` or `approval` node, else `{status: "completed", final_output: ...}`.
  2. Submit input via `POST /api/runs/{id}/human-input` or `/approval` — by
     default the call waits for the resumed run to complete.
  3. Stream events via `GET /api/runs/{id}/events` (SSE) — reconnect with
     `?since=<last_sequence>` to replay.

## Tests

```bash
pytest                           # full suite
pytest -k "test_compiler"        # compiler tests only
pytest -k "test_run_manager"     # pause/resume tests
```

Tests that need MongoDB auto-skip when the daemon is unreachable. With
`docker compose up -d mongo` running, every test in the suite executes.

## Project layout

```
dynamic-agent-workflow-server/
├── app/
│   ├── main.py                       # FastAPI factory + lifespan
│   ├── core/                         # config, logging, errors, security
│   ├── api/routes/                   # health, models, workflows, runs, tools, observability
│   ├── db/                           # async Mongo client + index provisioning
│   ├── schemas/                      # WorkflowDefinition, Node, Edge, ValidationReport, ...
│   ├── workflow/
│   │   ├── loader.py                 # 3-backend strategy (metadata API / Mongo / inline)
│   │   ├── validation.py             # validate(definition) → ValidationReport
│   │   ├── graph_utils.py            # adjacency, topo sort, reachability, ...
│   │   ├── variables.py              # {{path.to.value}} resolver
│   │   ├── state.py                  # merge_updates + path helpers
│   │   ├── hashing.py                # canonical definition hash
│   │   └── node_executors/           # 13 executors, each registered via @register
│   ├── langgraph_runtime/
│   │   ├── graph_builder.py          # compile_workflow + generic node wrapper
│   │   ├── dynamic_router.py         # next_handle → target lookup
│   │   ├── state_schema.py           # TypedDict + reducers
│   │   ├── checkpointing.py          # In-memory checkpointer (v1)
│   │   └── compile_cache.py          # hash-keyed in-process cache
│   ├── llm/
│   │   ├── service.py                # LLMService (dispatch + capability gating + JSON fallback)
│   │   ├── registry.py               # ProviderRegistry + ModelRegistry
│   │   └── providers/                # 7 providers (mock, openai, anthropic, ollama, vllm, openai_compatible, huggingface)
│   ├── tools/                        # ToolRegistry + base + mock + http
│   ├── runtime/
│   │   ├── run_manager.py            # the central orchestrator
│   │   └── event_bus.py              # in-process pub/sub for SSE
│   ├── repositories/                 # async Mongo repos (run, event, checkpoint, compile cache)
│   ├── observability/                # Langfuse wrapper (no-op safe)
│   └── tests/                        # 130+ tests
├── config/models.yaml                # the live model registry
├── docker-compose.yml                # Mongo (+ optional Langfuse) for local dev
├── pyproject.toml
└── .env.example
```

## Design invariants

The plan + spec are non-negotiable on these — and tests pin them:

  1. **No workflow topology is hard-coded.** Every edge, node id, branch
     handle, and model id comes from data.
  2. **No node-type if/else in routing or runtime.** The wrapper looks up an
     executor by `node.type` once; there is no runtime switch.
  3. **Routing is by `next_handle` matched against edge `sourceHandle`.**
     Defaults (`out`/`error`/`else`) are fallthroughs, not hard-codes.
  4. **Output node is terminal.** Outgoing edges from output → validation error.
  5. **Pause/resume uses native LangGraph `interrupt()`** with a checkpointer.
     Resume goes through `Command(resume=value)` against the same `thread_id`.
  6. **Secrets never leak.** `sanitize_error()` redacts API-key-shaped values
     in error responses; `/api/models` reports availability booleans only.
  7. **Limits are enforced.** `MAX_WORKFLOW_STEPS`, `NODE_TIMEOUT_SECONDS`,
     `WORKFLOW_TIMEOUT_SECONDS`, `MAX_AGENT_ITERATIONS`, `MAX_SUBFLOW_DEPTH`
     all check at the wrapper / executor / run manager level.
  8. **Script node is disabled by default** and only mock-executes when
     enabled. No arbitrary Python ever runs in-process.

## Known v1 deferrals (callable but not yet shipped)

- **Parallel branches** — the validator surfaces handoff fan-out as a warning
  only; reducer-aware parallel state is out of scope.
- **Cross-process resume** — the in-memory checkpointer means a server restart
  loses in-flight pauses. Domain state still persists to Mongo; the run row
  records `status: paused` and the pause payload, so it's recoverable
  out-of-band. A Mongo-backed `BaseCheckpointSaver` is the upgrade path.
- **WebSocket streaming** — SSE covers spec §14; WS can be added later.
- **HuggingFace provider** — placeholder per spec §8.
- **Real Script node sandbox** — TODO in `script_node.py`.

## Acceptance

All 22 acceptance criteria from spec §19 are met. See `app/tests/` for the
verification suite.
