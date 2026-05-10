from __future__ import annotations

from typing import Any


class WorkflowServerError(Exception):
    """Base for all server-domain errors. Carries a stable code + http status."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class WorkflowNotFoundError(WorkflowServerError):
    code = "workflow_not_found"
    http_status = 404


class WorkflowValidationError(WorkflowServerError):
    code = "workflow_validation_failed"
    http_status = 422


class RunNotFoundError(WorkflowServerError):
    code = "run_not_found"
    http_status = 404


class RunStateConflictError(WorkflowServerError):
    """Raised on resume race / cancel race / pause-state mismatch."""

    code = "run_state_conflict"
    http_status = 409


class CompilationError(WorkflowServerError):
    code = "workflow_compilation_failed"
    http_status = 422


class ExecutionLimitExceeded(WorkflowServerError):
    code = "execution_limit_exceeded"
    http_status = 422


class ConfigurationError(WorkflowServerError):
    code = "configuration_error"
    http_status = 500
