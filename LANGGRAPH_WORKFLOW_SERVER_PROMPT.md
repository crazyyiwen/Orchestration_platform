You are Claude Code.

I already have:

1. A React Agent Workflow Builder frontend.
   - Users can drag/drop nodes.
   - Users can connect nodes with edges.
   - Node UI is registry/schema-driven.
   - Workflow JSON is pure serializable JSON.
   - Frontend supports Start, LLM, Agent, Rule, Script, Human Input, Output, Sub Flow, Variable Update, HTTP Request, Guardrail, Approval, External Agent, and handoff-style connections.

2. A workflow metadata API.
   - Built with Python FastAPI + MongoDB.
   - It can save/update/delete/list workflow metadata and workflow definition JSON.
   - It stores full workflow JSON and searchable summary metadata.

Now I need a new backend project:

Dynamic Agent Workflow Server

Tech stack:
- Python 3.11+
- FastAPI
- LangGraph
- LangChain core abstractions where useful
- Langfuse for tracing/observability
- MongoDB for workflow definitions, run state, checkpoints, events, and run history
- PyMongo Async API preferred
- Pydantic v2
- pydantic-settings
- httpx
- Server-Sent Events or WebSocket streaming
- pytest

Main goal:
Build a server that can dynamically build and run an agent workflow from workflow JSON created by my frontend.

Very important:
The workflow server must not hard-code workflow-specific logic.

The server should dynamically:
- load workflow definition JSON from MongoDB or from the metadata API
- validate graph structure
- compile the workflow into a LangGraph graph
- run the workflow
- support streaming status/events back to frontend
- support LLM nodes
- support Agent nodes
- support Rule branch nodes
- support Human Input pause/resume
- support Approval pause/resume
- support Sub Flow execution
- support Variable Update
- support HTTP Request
- support Guardrail
- support External Agent
- support Output nodes
- support Langfuse tracing
- persist run state and events into MongoDB

Design principle:
Everything must be dynamic, registry-driven, and configuration-driven.

Allowed:
- Generic node executor classes by node type.
- Generic node compiler functions by node type.
- Generic node registry.
- Generic provider registry.
- Generic tool registry.

Not allowed:
- Hard-coded workflow IDs.
- Hard-coded node names.
- Hard-coded graph topology.
- Hard-coded edges.
- Hard-coded model names.
- Hard-coded provider choices.
- Hard-coded route logic for specific workflows.
- Node execution logic directly inside FastAPI route handlers.
- Executing arbitrary Script node code inside the FastAPI process.

The server should behave like this:

Frontend workflow JSON
    ↓
Workflow Metadata API / MongoDB
    ↓
Workflow Loader
    ↓
Workflow Validator
    ↓
Dynamic LangGraph Compiler
    ↓
Node Executor Registry
    ↓
LLM Provider Registry / Tool Registry / HTTP Client / Runtime State
    ↓
LangGraph Runtime
    ↓
MongoDB run state + Langfuse traces + SSE/WebSocket events

Core architecture:

1. Workflow Loader
- Load workflow definition by workflow_id.
- Support loading from local MongoDB.
- Support loading from existing metadata API via HTTP if configured.
- Support direct run with inline workflow JSON for testing.
- Return normalized WorkflowDefinition.

2. Workflow Validator
Validate workflow before compilation:
- workflow has nodes
- workflow has edges
- node ids unique
- node names unique
- edge ids unique
- every edge source exists
- every edge target exists
- Start node exists or one clear start node can be inferred
- Output node exists
- duplicate edges rejected
- disconnected nodes reported
- cycles rejected by default unless controlled-loop mode is enabled
- rule branch handles have matching outgoing edges if required
- human input node has an output path
- approval node has approved/rejected paths if configured
- required node config fields are present
- unknown node types are rejected unless plugin mode is enabled

3. Dynamic LangGraph Compiler
Build a LangGraph graph from workflow JSON dynamically.

The compiler should:
- create a LangGraph node for every workflow node
- map each workflow node to a generic executor wrapper
- add graph edges based on workflow edges
- add conditional edges for nodes whose next edge depends on runtime result
- use sourceHandle / next_handle to route branches
- support Start node as entry point
- support Output node as terminal node
- support pause/resume nodes
- compile graph only from workflow JSON, never from hard-coded topology

Important routing model:
Each node execution returns:
- status
- output
- next_handle
- error
- pause_payload if paused

The compiled graph routes by next_handle.

Example:
Rule node returns next_handle = "case_1".
The graph routes to the edge whose sourceHandle is "case_1".

4. Runtime State Model
Each run should have JSON-serializable state:

{
  "run_id": "...",
  "workflow_id": "...",
  "workflow_version": 1,
  "status": "pending | running | paused | completed | failed | cancelled",
  "current_node_id": "...",
  "current_node_name": "...",
  "variables": {
    "system": {},
    "runtime": {},
    "nodes": {}
  },
  "messages": [],
  "events": [],
  "final_output": null,
  "error": null,
  "pause": null
}

Each node result must be stored at:

variables.nodes.<nodeName>.result

Example:
variables.nodes.llm_1.result.answer

5. Variable Resolver
Support frontend variable syntax:

{{system.userQuery}}
{{system.attachments}}
{{system.files}}
{{system.humanInput}}
{{runtime.workflowMetaData.workflowId}}
{{runtime.workflowMetaData.agentName}}
{{nodes.llm_1.result.answer}}
{{nodes.agent_1.result.toolResult}}

Implement:
- resolve variable path
- resolve template string
- resolve mappings recursively
- preserve original object type if the whole value is one variable reference
- set value by dot path
- merge state updates safely

6. Node Executor Registry
Create a dynamic node executor registry.

Node types:
- start
- llm
- agent
- rule
- script
- human_input
- output
- sub_flow
- variable_update
- http_request
- guardrail
- approval
- external_agent

Every executor must implement the same interface:
- receive node definition
- receive runtime state
- receive execution context
- return NodeExecutionResult

NodeExecutionResult should include:
- status: success | failed | paused | skipped
- output
- next_handle
- error
- state_updates
- pause_payload
- events

Do not put node-specific if/else logic inside the workflow runner.
Use the executor registry.

7. Node behavior requirements

Start Node:
- Initializes runtime context.
- Returns next_handle = out.

LLM Node:
- Resolve messages and input mappings.
- Select model by model_id from node config.
- Call LLM service.
- Support response format text/json.
- Support structured output when configured.
- Store provider, model, content, parsed_json, usage.
- Return out on success.
- Return error handle if configured and failed.

Agent Node:
- Dynamically build agent behavior based on node config.
- Support tools from ToolRegistry.
- Support max_iterations.
- Support tool-call loop.
- Support handoff handles if configured.
- Store final answer, intermediate steps, tool calls.
- Trace all steps in Langfuse.

Rule Node:
- Evaluate conditions from node config.
- Support equals, not_equals, contains, not_contains, greater_than, less_than, exists, empty, regex.
- Support AND/OR condition groups.
- Return next_handle for matched branch.
- Return else if no match.

Human Input Node:
- Pause execution.
- Store pause payload with question, input type, save variable path.
- Emit event human_input_required.
- Resume endpoint should accept input.
- Save input to configured variable path.
- Continue through out handle.

Approval Node:
- Pause execution.
- Emit approval_required.
- Resume endpoint accepts approved/rejected.
- Continue using approved or rejected handle.
- Store decision in runtime state.

Output Node:
- Resolve output mappings.
- Set final_output.
- Mark run completed.

Variable Update Node:
- Support set, append, merge, increment, remove.
- Update runtime state.
- Return out.

HTTP Request Node:
- Resolve URL, headers, query params, body.
- Use async httpx.
- Respect allow_external_http config.
- Store status, headers, text, json.
- Return out or error.

Guardrail Node:
- Validate mapped input against rules.
- Return allow or block handle.
- Store reason.

Sub Flow Node:
- Load and execute another workflow by workflow_id.
- Pass mapped inputs.
- Store subflow result.
- Enforce max_subflow_depth.

External Agent Node:
- Call configured external agent endpoint.
- Resolve input mappings.
- Store response.
- Return out or error.

Script Node:
- Disabled by default.
- Do not execute arbitrary Python in the FastAPI process.
- For local demo, allow mock-only script executor.
- Add TODO for future sandbox worker.

8. LLM Provider Registry
The server must support multiple LLM providers dynamically.

Provider types:
- openai
- anthropic
- ollama
- vllm
- huggingface
- openai_compatible
- mock

Do not hard-code models.
Models must come from config and environment variables.

Create:
- BaseLLMProvider
- OpenAIProvider
- AnthropicProvider
- OllamaProvider
- VLLMProvider through OpenAI-compatible endpoint
- HuggingFaceProvider placeholder
- OpenAICompatibleProvider
- MockProvider
- LLMService
- ModelRegistry
- ProviderRegistry

LLMService responsibilities:
- select provider by model_id
- validate capability
- normalize request
- normalize response
- handle JSON mode fallback
- return usage metadata
- integrate with Langfuse tracing

9. Tool Registry
Agent nodes should use a dynamic tool registry.

Tool system should support:
- built-in tools
- HTTP tools
- external tools
- workflow tools
- subflow tools
- mock tools for tests

Do not allow arbitrary Python execution as a tool.

ToolRegistry responsibilities:
- list tools
- get tool schema
- execute tool
- validate tool input
- return tool result
- trace tool execution in Langfuse

10. LangGraph Integration
Use LangGraph as the workflow runtime.

Dynamic compilation strategy:
- convert frontend workflow nodes to LangGraph nodes
- each LangGraph node calls a generic executor wrapper
- each executor returns next_handle
- compiler adds conditional routing based on next_handle
- graph state is a TypedDict-like state model or Pydantic-compatible dict state
- support checkpointing if practical
- persist state to MongoDB after each node execution
- emit events after each node execution

The graph compiler should be generic:
- no hard-coded workflow topology
- no hard-coded node names
- no hard-coded edges
- no hard-coded branch labels except standard defaults like out/error/else

11. Langfuse Integration
Add Langfuse tracing for every run.

Trace structure:
- trace per workflow run
- span per node execution
- generation span for LLM calls
- span for tool calls
- span for HTTP calls
- metadata includes workflow_id, run_id, node_id, node_name, node_type, workflow_version
- capture errors
- capture latency
- capture token usage when available

Langfuse should be optional:
- enabled by config
- disabled safely if keys are missing

Do not expose Langfuse keys to frontend.

12. MongoDB Persistence
Collections:

workflow_runs:
- run_id
- workflow_id
- workflow_version
- status
- input
- state
- final_output
- error
- started_at
- completed_at
- updated_at
- created_by

workflow_run_events:
- event_id
- run_id
- workflow_id
- sequence
- type
- node_id
- node_name
- node_type
- payload
- created_at

workflow_checkpoints:
- checkpoint_id
- run_id
- workflow_id
- node_id
- node_name
- state
- created_at

workflow_compiled_cache:
- workflow_id
- workflow_version
- definition_hash
- compiled_metadata
- created_at
- updated_at

Optional:
workflow_definitions can be read from existing metadata API. Do not duplicate unless local mode is enabled.

Indexes:
- workflow_runs.run_id unique
- workflow_runs.workflow_id
- workflow_runs.status
- workflow_runs.created_at desc
- workflow_run_events.run_id + sequence unique
- workflow_run_events.workflow_id
- workflow_checkpoints.run_id + created_at
- workflow_compiled_cache.workflow_id + workflow_version unique

13. API Endpoints

Health:
- GET /health

Models:
- GET /api/models
- GET /api/providers
- POST /api/models/test

Workflow compile:
- POST /api/workflows/{workflow_id}/compile
- POST /api/workflows/compile-inline
- POST /api/workflows/{workflow_id}/validate-runtime

Runs:
- POST /api/workflows/{workflow_id}/runs
- POST /api/workflows/run-inline
- GET /api/runs/{run_id}
- GET /api/runs/{run_id}/events
- GET /api/runs/{run_id}/state
- POST /api/runs/{run_id}/cancel
- POST /api/runs/{run_id}/resume
- POST /api/runs/{run_id}/human-input
- POST /api/runs/{run_id}/approval

Tools:
- GET /api/tools
- POST /api/tools/test

Observability:
- GET /api/runs/{run_id}/trace-link if Langfuse is enabled

14. Event Streaming
Support Server-Sent Events first.

Events:
- run_started
- workflow_loaded
- workflow_validated
- workflow_compiled
- node_started
- node_completed
- node_failed
- llm_started
- llm_completed
- llm_token if streaming supported
- tool_started
- tool_completed
- rule_evaluated
- http_request_started
- http_request_completed
- human_input_required
- approval_required
- run_paused
- run_resumed
- run_completed
- run_failed
- run_cancelled

Events should be:
- persisted in MongoDB
- streamed to frontend
- optionally sent to Langfuse as spans/events

15. Configuration
Use pydantic-settings.

Environment variables:
- APP_NAME
- APP_ENV
- API_PREFIX
- FRONTEND_ORIGINS
- MONGODB_URI
- MONGODB_DATABASE
- METADATA_API_BASE_URL
- METADATA_API_ENABLED
- LANGFUSE_ENABLED
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_HOST
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- HUGGINGFACE_API_KEY
- DEFAULT_MODEL_ID
- ENABLE_SCRIPT_NODE=false
- ALLOW_EXTERNAL_HTTP=true
- MAX_WORKFLOW_STEPS=100
- MAX_AGENT_ITERATIONS=8
- MAX_SUBFLOW_DEPTH=3
- NODE_TIMEOUT_SECONDS=60
- WORKFLOW_TIMEOUT_SECONDS=600

16. Security
- Do not expose API keys.
- Do not log secrets.
- Do not execute arbitrary code.
- Script node disabled by default.
- Enforce max workflow steps.
- Enforce node timeout.
- Enforce workflow timeout.
- Enforce max agent iterations.
- Enforce max subflow depth.
- Restrict external HTTP if configured.
- Validate workflow before compile/run.
- Keep route handlers thin.
- Sanitize errors before returning to frontend.

17. Project Structure

dynamic-agent-workflow-server/
  app/
    main.py

    core/
      config.py
      logging.py
      errors.py
      security.py

    api/
      deps.py
      routes/
        health.py
        models.py
        workflows.py
        runs.py
        tools.py
        observability.py

    db/
      mongodb.py
      indexes.py
      collections.py

    schemas/
      workflow.py
      node.py
      edge.py
      run.py
      events.py
      validation.py
      llm.py
      tools.py

    workflow/
      loader.py
      validation.py
      variables.py
      graph_utils.py
      compiler.py
      runtime.py
      state.py
      routing.py
      hashing.py

      node_executors/
        base.py
        start_node.py
        llm_node.py
        agent_node.py
        rule_node.py
        script_node.py
        human_input_node.py
        output_node.py
        subflow_node.py
        variable_update_node.py
        http_request_node.py
        guardrail_node.py
        approval_node.py
        external_agent_node.py

    langgraph_runtime/
      graph_builder.py
      state_schema.py
      checkpointing.py
      dynamic_router.py
      compile_cache.py

    llm/
      service.py
      registry.py
      types.py
      providers/
        base.py
        openai_provider.py
        anthropic_provider.py
        ollama_provider.py
        openai_compatible_provider.py
        vllm_provider.py
        huggingface_provider.py
        mock_provider.py

    tools/
      registry.py
      base.py
      builtin_tools.py
      http_tool.py
      workflow_tool.py

    observability/
      langfuse_client.py
      tracing.py
      event_mapper.py

    runtime/
      run_manager.py
      event_bus.py
      state_store.py
      checkpoint_store.py
      resume_manager.py

    repositories/
      run_repository.py
      event_repository.py
      checkpoint_repository.py
      compile_cache_repository.py

    services/
      workflow_run_service.py
      workflow_compile_service.py
      workflow_validation_service.py
      model_service.py
      tool_service.py

    tests/
      test_workflow_loader.py
      test_workflow_validation.py
      test_variable_resolution.py
      test_dynamic_compiler.py
      test_routing.py
      test_rule_node.py
      test_output_node.py
      test_human_input_pause_resume.py
      test_mock_llm_node.py
      test_run_manager.py

  .env.example
  config.example.yaml
  pyproject.toml
  README.md
  docker-compose.yml

18. Implementation Phases

Phase 1: Project foundation
- FastAPI app
- config
- logging
- health endpoint
- CORS
- README
- pyproject

Phase 2: MongoDB runtime persistence
- MongoDB async client
- collections
- indexes
- run repository
- event repository
- checkpoint repository

Phase 3: Workflow schemas and loader
- Pydantic models for workflow JSON
- loader from metadata API
- loader from MongoDB/local
- inline workflow loading for tests

Phase 4: Validation and graph utilities
- validate graph
- build adjacency
- build incoming map
- detect start
- detect output
- detect cycles
- detect unreachable nodes
- validate sourceHandle/targetHandle
- validate branch handles

Phase 5: Variable resolver and state store
- resolve template variables
- resolve mappings recursively
- set by path
- merge state updates
- persist runtime state

Phase 6: LLM provider system
- provider registry
- model registry
- mock provider
- OpenAI provider
- Anthropic provider
- OpenAI-compatible provider
- Ollama/vLLM support
- /api/models

Phase 7: Node executor registry
- base executor
- start
- output
- rule
- variable update
- human input
- approval
- llm
- http request
- guardrail
- subflow
- external agent
- script mock/disabled
- agent node with tool loop

Phase 8: Dynamic LangGraph compiler
- compile workflow JSON to LangGraph
- add dynamic nodes
- add dynamic routing
- map sourceHandle to next_handle
- support conditional edges
- support terminal output
- support pause nodes
- no hard-coded topology

Phase 9: Run manager
- create run
- execute compiled graph
- persist state
- persist events
- support cancel
- support pause/resume
- support human input resume
- support approval resume
- enforce max steps/timeouts

Phase 10: Langfuse observability
- trace workflow run
- span per node
- generation per LLM
- span per tool/http call
- trace link endpoint
- optional enable/disable

Phase 11: API routes
- compile
- validate-runtime
- start run
- inline run
- get run
- get state
- stream events
- cancel
- resume
- human input
- approval
- models
- tools
- trace link

Phase 12: Tests and sample workflows
- simple Start → LLM → Output
- Start → Rule → branch → Output
- Start → Human Input → LLM → Output
- Start → Approval → approved/rejected Output
- Start → HTTP Request → Output
- Start → Agent with mock tool → Output
- Sub Flow test
- invalid graph tests
- pause/resume tests

Phase 13: Final documentation and polish
- README
- .env.example
- config.example.yaml
- curl examples
- frontend integration examples
- Langfuse setup
- MongoDB setup
- Docker compose

19. Acceptance Criteria
- Server starts with uvicorn.
- GET /health works.
- MongoDB connection works.
- Server can load workflow JSON by workflow_id.
- Server can validate workflow JSON.
- Server can compile workflow JSON into LangGraph dynamically.
- No workflow topology is hard-coded.
- No node names are hard-coded.
- Run can start from Start node.
- Rule branch routing works through next_handle/sourceHandle.
- Human Input pauses and resumes.
- Approval pauses and resumes.
- LLM node works with Mock provider.
- LLM node works with OpenAI if configured.
- LLM node works with Anthropic if configured.
- Open-source model works through Ollama or OpenAI-compatible endpoint.
- Output node produces final_output.
- Run state is persisted in MongoDB.
- Run events are persisted in MongoDB.
- SSE streams events to frontend.
- Langfuse traces are created when enabled.
- Secrets are never exposed.
- Script node does not run unsafe code.
- Tests pass.

20. Most Important Design Reminder
This server dynamically builds executable agent workflows from frontend workflow JSON.

The workflow definition is data.
The graph topology is data.
Node configuration is data.
Model provider selection is config.
Execution is registry-driven.
LangGraph compilation is dynamic.
No workflow-specific code should be required to run a new workflow.

Start by creating a detailed architecture plan and file structure.
Do not write code until the plan is approved.