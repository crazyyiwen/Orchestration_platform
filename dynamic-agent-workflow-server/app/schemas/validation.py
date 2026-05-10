"""Validation DTOs (used by Phase 4's validator and the validate-runtime route)."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    code: str
    severity: ValidationSeverity
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    is_valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    infos: list[ValidationIssue] = Field(default_factory=list)

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> "ValidationReport":
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR.value]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING.value]
        infos = [i for i in issues if i.severity == ValidationSeverity.INFO.value]
        return cls(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )
