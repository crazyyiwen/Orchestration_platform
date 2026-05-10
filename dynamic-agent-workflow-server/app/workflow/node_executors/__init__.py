"""Auto-imports every executor module so ``@register`` populates the registry.

Adding a new executor: create the module under this package and import it
here. Order doesn't matter — registration is idempotent within one process.
"""
from __future__ import annotations

# Import order is alphabetical; effects are import-time @register side-effects.
from app.workflow.node_executors import (  # noqa: F401
    agent_node,
    approval_node,
    external_agent_node,
    guardrail_node,
    http_request_node,
    human_input_node,
    llm_node,
    output_node,
    rule_node,
    script_node,
    start_node,
    subflow_node,
    variable_update_node,
)
from app.workflow.node_executors.base import (  # noqa: F401
    BaseNodeExecutor,
    ExecutionContext,
    NodeExecutionResult,
    get_executor,
    register,
    registered_types,
)
