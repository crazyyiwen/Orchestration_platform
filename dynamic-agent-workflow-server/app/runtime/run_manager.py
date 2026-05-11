"""Run manager — orchestrates compile + invoke + pause + resume + cancel.

This is the single component that ties together: workflow loader, validator,
LangGraph compiler, repositories, event bus, and LLM service. Route handlers
in Phase 11 are thin wrappers around its public methods.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from app.core.config import Settings
from app.core.errors import (
    ConfigurationError,
    RunNotFoundError,
    RunStateConflictError,
    WorkflowServerError,
)
from app.langgraph_runtime.checkpointing import make_checkpointer
from app.langgraph_runtime.compile_cache import CompileCache
from app.llm.service import LLMService
from app.repositories.event_repository import EventRepository
from app.repositories.run_repository import RunRepository
from app.runtime.event_bus import EventBus
from app.schemas.workflow import WorkflowDefinition
from app.tools.registry import ToolRegistry
from app.workflow.loader import WorkflowLoader
from app.workflow.node_executors.base import ExecutionContext
from app.workflow.state import empty_runtime_state
from app.workflow.validation import validate

log = logging.getLogger(__name__)


class RunManager:
    """Owns run lifecycle. One instance per process; injected into routes."""

    def __init__(
        self,
        *,
        settings: Settings,
        loader: WorkflowLoader,
        run_repo: RunRepository | None,
        event_repo: EventRepository | None,
        llm_service: LLMService,
        tool_registry: ToolRegistry,
        http_client: httpx.AsyncClient,
        event_bus: EventBus | None = None,
    ) -> None:
        self._settings = settings
        self._loader = loader
        self._runs = run_repo
        self._events = event_repo
        self._llm = llm_service
        self._tools = tool_registry
        self._http = http_client
        self._bus = event_bus or EventBus()
        self._compile_cache = CompileCache()
        self._checkpointer: BaseCheckpointSaver = make_checkpointer()
        # Per-run task handles so we can cancel mid-flight.
        self._tasks: dict[str, asyncio.Task] = {}
        # In-memory cache of inline workflow definitions, keyed by
        # ``(workflow_id, version)``. Lets resume work for inline workflows
        # that aren't in the metadata API / local Mongo.
        self._inline_defs: dict[tuple[str, int], WorkflowDefinition] = {}

    # ----- public API -----------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    async def create_run(
        self,
        *,
        workflow_id: str,
        input: dict[str, Any] | None = None,
        version: int | None = None,
        inline_definition: WorkflowDefinition | None = None,
        parent_run_id: str | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Validate the workflow and create a run row (status=pending)."""
        definition = inline_definition or await self._loader.load_by_id(
            workflow_id, version=version
        )
        report = validate(definition, allow_cycles=True)
        if not report.is_valid:
            raise ConfigurationError(
                "workflow validation failed",
                details={"errors": [i.model_dump() for i in report.errors]},
            )
        # Cache inline definitions so resume_run() can find them again.
        if inline_definition is not None:
            self._inline_defs[
                (definition.workflow_id, definition.workflow_version)
            ] = definition

        run_id = f"run-{uuid.uuid4().hex[:16]}"
        initial_state = empty_runtime_state(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.workflow_version,
        )
        # Seed user inputs into ``system.*``.
        if input:
            initial_state["variables"]["system"].update(input)

        if self._runs is not None:
            await self._runs.create(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.workflow_version,
                input=input or {},
                parent_run_id=parent_run_id,
                initial_state=initial_state,
            )

        return {"run_id": run_id, "definition": definition, "state": initial_state, "depth": depth}

    async def start_run(
        self,
        run_id: str,
        *,
        definition: WorkflowDefinition,
        initial_state: dict[str, Any],
        depth: int = 0,
        parent_run_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """Compile + invoke. Returns run row (final on wait=True; current on False)."""
        graph = self._compile_cache.get_or_compile(
            definition,
            context_factory=lambda: ExecutionContext(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.workflow_version,
                settings=self._settings,
                llm_service=self._llm,
                tool_registry=self._tools,
                http_client=self._http,
                depth=depth,
                parent_run_id=parent_run_id,
                sub_flow_launcher=self._sub_flow_launcher(depth),
                event_bus=self._bus,
            ),
            checkpointer=self._checkpointer,
            on_node_event=lambda ev, rid=run_id, wfid=definition.workflow_id: self._record_event(
                rid, wfid, ev
            ),
        )

        if self._runs is not None:
            await self._runs.update_state(
                run_id, status="running", started_at_now=True
            )
        await self._bus.publish(run_id, {"type": "run_started", "payload": {"run_id": run_id}})

        coro = self._invoke_graph(
            graph,
            run_id=run_id,
            workflow_id=definition.workflow_id,
            state=initial_state,
        )
        if wait:
            final = await coro
            return final
        task = asyncio.create_task(coro)
        self._tasks[run_id] = task
        return {"run_id": run_id, "status": "running"}

    async def cancel_run(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        if self._runs is not None:
            ok = await self._runs.transition_status(
                run_id, from_status="running", to_status="cancelled"
            )
            if not ok:
                # Maybe was paused — also let pause→cancelled through.
                await self._runs.transition_status(
                    run_id, from_status="paused", to_status="cancelled"
                )
        await self._bus.publish(run_id, {"type": "run_cancelled", "payload": {}})
        await self._bus.close(run_id)
        return True

    async def resume_run(
        self,
        run_id: str,
        resume_input: Any,
        *,
        wait: bool = False,
    ) -> dict[str, Any]:
        """Submit a resume value (Human Input or Approval) and continue execution."""
        if self._runs is not None:
            transitioned = await self._runs.transition_status(
                run_id, from_status="paused", to_status="running"
            )
            if not transitioned:
                raise RunStateConflictError(
                    f"run {run_id!r} is not paused (resume race or already cancelled)"
                )
            row = await self._runs.get(run_id)
            if row is None:
                raise RunNotFoundError(f"run not found: {run_id}", details={"run_id": run_id})
            workflow_id = row["workflow_id"]
            workflow_version = row["workflow_version"]
        else:
            # No persistence — best-effort.
            workflow_id = ""
            workflow_version = 1

        # Prefer the in-memory inline cache so /run-inline workflows can resume.
        cached = self._inline_defs.get((workflow_id, workflow_version))
        if cached is not None:
            definition = cached
        else:
            definition = await self._loader.load_by_id(
                workflow_id, version=workflow_version
            )

        graph = self._compile_cache.get_or_compile(
            definition,
            context_factory=lambda: ExecutionContext(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.workflow_version,
                settings=self._settings,
                llm_service=self._llm,
                tool_registry=self._tools,
                http_client=self._http,
                depth=0,
                sub_flow_launcher=self._sub_flow_launcher(0),
                event_bus=self._bus,
            ),
            checkpointer=self._checkpointer,
            on_node_event=lambda ev, rid=run_id, wfid=definition.workflow_id: self._record_event(
                rid, wfid, ev
            ),
        )

        await self._bus.publish(run_id, {"type": "run_resumed", "payload": {}})
        # ``_resume_input`` is what executors read on the second pass.
        # We use Command(resume=...) to continue from the interrupt point.
        config = {"configurable": {"thread_id": run_id}}
        # Inject resume input into state via Command.update.
        resume_cmd = Command(
            resume=resume_input,
            update={"_resume_input": resume_input, "status": "running", "pause": None},
        )
        coro = self._invoke_graph_resume(
            graph, run_id=run_id, workflow_id=workflow_id, command=resume_cmd, config=config
        )
        if wait:
            final = await coro
            return {
                "run_id": run_id,
                "status": _resolve_resume_status(final),
                "final_output": final.get("final_output") if isinstance(final, dict) else None,
            }
        task = asyncio.create_task(coro)
        self._tasks[run_id] = task
        return {"run_id": run_id, "status": "running"}

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        if self._runs is None:
            return None
        return await self._runs.get(run_id)

    # ----- internals ------------------------------------------------------

    async def _invoke_graph(
        self,
        graph,
        *,
        run_id: str,
        workflow_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
        try:
            final = await asyncio.wait_for(
                graph.ainvoke(state, config=config),
                timeout=self._settings.WORKFLOW_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            await self._mark_failed(run_id, "cancelled")
            raise
        except asyncio.TimeoutError:
            await self._mark_failed(run_id, "workflow timeout exceeded")
            return {}
        except WorkflowServerError as e:
            await self._mark_failed(run_id, str(e))
            return {}
        except Exception as e:  # noqa: BLE001
            log.exception("graph invocation crashed run_id=%s", run_id)
            await self._mark_failed(run_id, f"{type(e).__name__}: {e}")
            return {}

        await self._on_invoke_finished(run_id, final)
        return final

    async def _invoke_graph_resume(
        self,
        graph,
        *,
        run_id: str,
        workflow_id: str,
        command: Command,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            final = await asyncio.wait_for(
                graph.ainvoke(command, config={**config, "recursion_limit": 200}),
                timeout=self._settings.WORKFLOW_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            await self._mark_failed(run_id, "cancelled")
            raise
        except WorkflowServerError as e:
            await self._mark_failed(run_id, str(e))
            return {}
        except Exception as e:  # noqa: BLE001
            log.exception("resume graph invocation crashed run_id=%s", run_id)
            await self._mark_failed(run_id, f"{type(e).__name__}: {e}")
            return {}

        await self._on_invoke_finished(run_id, final)
        return final

    async def _on_invoke_finished(self, run_id: str, final: dict[str, Any]) -> None:
        status = final.get("status", "completed")
        # Native LangGraph interrupt: the returned state carries
        # ``__interrupt__: [Interrupt(value=pause_payload)]``. We translate that
        # into our domain ``paused`` status + pause payload.
        interrupts = final.get("__interrupt__")
        if interrupts:
            payload = _extract_pause_payload(interrupts)
            paused_state = _sanitize_for_persistence(
                {**final, "status": "paused", "pause": payload}
            )
            if self._runs is not None:
                await self._runs.update_state(
                    run_id,
                    state=paused_state,
                    status="paused",
                )
            await self._bus.publish(
                run_id, {"type": "run_paused", "payload": {"pause": payload}}
            )
            return
        if final.get("status") == "paused" or final.get("pause"):
            if self._runs is not None:
                await self._runs.update_state(
                    run_id,
                    state=_sanitize_for_persistence(final),
                    status="paused",
                )
            await self._bus.publish(
                run_id, {"type": "run_paused", "payload": {"pause": final.get("pause")}}
            )
            return

        if status not in {"completed", "failed", "cancelled"}:
            status = "completed"
        if self._runs is not None:
            await self._runs.update_state(
                run_id,
                state=_sanitize_for_persistence(final),
                status=status,
                final_output=_sanitize_for_persistence(final.get("final_output")),
                completed_at_now=True,
            )
        await self._bus.publish(
            run_id,
            {
                "type": "run_completed" if status == "completed" else f"run_{status}",
                "payload": {"final_output": final.get("final_output")},
            },
        )
        await self._bus.close(run_id)

    async def _mark_failed(self, run_id: str, message: str) -> None:
        if self._runs is not None:
            await self._runs.update_state(
                run_id,
                status="failed",
                error={"message": message},
                completed_at_now=True,
            )
        await self._bus.publish(run_id, {"type": "run_failed", "payload": {"error": message}})
        await self._bus.close(run_id)

    async def _record_event(
        self, run_id: str, workflow_id: str, event: dict[str, Any]
    ) -> None:
        """Persist + publish a single event (atomic-enough for SSE replay)."""
        if self._runs is not None and self._events is not None:
            try:
                seq = await self._runs.allocate_event_sequence(run_id)
                await self._events.append(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    sequence=seq,
                    type=event.get("type", "unknown"),
                    payload=event.get("payload"),
                    node_id=event.get("node_id"),
                    node_name=event.get("node_name"),
                    node_type=event.get("node_type"),
                )
                event = {**event, "sequence": seq}
            except Exception:  # noqa: BLE001
                log.exception("event persistence failed run_id=%s", run_id)
        await self._bus.publish(run_id, event)

    def _sub_flow_launcher(self, depth: int):
        """Returns a coroutine that runs another workflow synchronously."""
        manager = self

        async def launcher(
            sub_workflow_id: str,
            inputs: dict[str, Any],
            new_depth: int,
            parent_run_id: str,
        ) -> dict[str, Any]:
            run = await manager.create_run(
                workflow_id=sub_workflow_id,
                input=inputs,
                parent_run_id=parent_run_id,
                depth=new_depth,
            )
            final = await manager.start_run(
                run["run_id"],
                definition=run["definition"],
                initial_state=run["state"],
                depth=new_depth,
                parent_run_id=parent_run_id,
                wait=True,
            )
            return {
                "run_id": run["run_id"],
                "final_output": final.get("final_output") if isinstance(final, dict) else None,
                "status": final.get("status") if isinstance(final, dict) else "completed",
            }

        return launcher


def _sanitize_for_persistence(v: Any) -> Any:
    """Strip non-BSON-serializable values (e.g. LangGraph ``Interrupt``).

    Drops the transient ``__interrupt__`` key entirely and recursively
    converts any object with a ``model_dump`` method (Pydantic) or a ``value``
    attribute (Interrupt) to a plain dict.
    """
    if v is None:
        return None
    if isinstance(v, dict):
        return {
            k: _sanitize_for_persistence(val)
            for k, val in v.items()
            if k != "__interrupt__"
        }
    if isinstance(v, list):
        return [_sanitize_for_persistence(x) for x in v]
    if isinstance(v, tuple):
        return [_sanitize_for_persistence(x) for x in v]
    if hasattr(v, "model_dump"):
        return v.model_dump()
    if hasattr(v, "value") and v.__class__.__name__ == "Interrupt":
        return {"value": _sanitize_for_persistence(v.value)}
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _resolve_resume_status(final: Any) -> str:
    if not isinstance(final, dict):
        return "completed"
    if final.get("__interrupt__"):
        return "paused"
    return final.get("status", "completed")


def _extract_pause_payload(interrupts: Any) -> dict[str, Any]:
    """Pull the payload out of a list of LangGraph Interrupt objects.

    Each Interrupt carries the value passed to ``interrupt(...)`` — for our
    executors that's the ``pause_payload`` dict.
    """
    if not interrupts:
        return {}
    first = interrupts[0]
    val = getattr(first, "value", None)
    if isinstance(val, dict):
        return val
    return {"raw": val}

