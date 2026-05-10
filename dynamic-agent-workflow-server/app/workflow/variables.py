"""Variable resolver for ``{{path.to.value}}`` template syntax (spec §5).

Single string template that's *exactly* one ``{{ref}}`` returns the resolved
object with its original type preserved (dict / list / int / None / etc.).
Anything else interpolates to a string. Mappings and sequences are walked
recursively, bounded by ``max_recursion_depth`` (plan tricky-detail #8).

The resolver is namespace-agnostic. Callers (Phase 9 run_manager) decide
which subtree of state to root it at — typically ``state.variables`` so paths
read like ``system.userQuery``, ``nodes.llm_1.result.answer``,
``runtime.workflowMetaData.agentName``, plus any other namespaces the
frontend uses (``flow.*``, ``thread.*``, etc.).
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.core.errors import WorkflowServerError
from app.workflow.state import get_path

# Match ``{{ path.to.value }}``. The captured group is the path without
# surrounding whitespace; the regex tolerates inner whitespace too.
_TEMPLATE_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

# Sentinel used to distinguish "missing key" from "key whose value is None".
_MISSING = object()


class VariableResolver:
    def __init__(self, root: dict[str, Any], *, max_recursion_depth: int = 50) -> None:
        if not isinstance(root, dict):
            raise TypeError("VariableResolver root must be a dict")
        self._root = root
        self._max_depth = max_recursion_depth

    # -- public API --------------------------------------------------------

    def resolve_path(self, path: str) -> Any:
        """Walk a dotted path through the root. Returns ``None`` on miss."""
        return get_path(self._root, path, default=None)

    def resolve_string(self, template: str) -> Any:
        """Resolve a single template string.

        * If the string is *exactly* one ``{{ref}}`` (no surrounding text or
          additional refs), the resolved value is returned with its original
          type preserved.
        * Otherwise, every ``{{ref}}`` is replaced inline (collections become
          JSON, ``None`` becomes empty string, scalars become ``str(...)``).
        """
        if not isinstance(template, str):
            return template
        single = self._maybe_single_ref(template)
        if single is not _MISSING:
            return single
        return _TEMPLATE_RE.sub(lambda m: _stringify(self.resolve_path(m.group(1))), template)

    def resolve_value(self, value: Any) -> Any:
        """Recursively resolve a dict / list / string. Other types pass through."""
        return self._walk(value, depth=0)

    # -- internals ---------------------------------------------------------

    def _maybe_single_ref(self, template: str) -> Any:
        """If the trimmed template is exactly one ``{{ref}}``, return the
        resolved value. Otherwise return the ``_MISSING`` sentinel.
        """
        stripped = template.strip()
        match = _TEMPLATE_RE.fullmatch(stripped)
        if match is None:
            return _MISSING
        # Confirm there are no *other* ``{{...}}`` inside (the fullmatch already
        # ensures that, but we also reject e.g. ``{{a}}{{b}}`` which fullmatches
        # neither but is worth being explicit about).
        if _TEMPLATE_RE.search(template[match.end() :]):
            return _MISSING
        return self.resolve_path(match.group(1))

    def _walk(self, value: Any, *, depth: int) -> Any:
        if depth > self._max_depth:
            raise WorkflowServerError(
                f"variable resolution exceeded max_recursion_depth={self._max_depth}"
            )
        if isinstance(value, str):
            return self.resolve_string(value)
        if isinstance(value, dict):
            return {k: self._walk(v, depth=depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [self._walk(v, depth=depth + 1) for v in value]
        if isinstance(value, tuple):
            return tuple(self._walk(v, depth=depth + 1) for v in value)
        return value


def _stringify(value: Any) -> str:
    """Convert a value to its template-interpolation form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=False)
    except (TypeError, ValueError):
        return str(value)
