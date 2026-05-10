from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quieter third-party loggers in dev.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.INFO))

    _CONFIGURED = True


def get_logger(name: str, **bind: Any) -> logging.LoggerAdapter:
    """Return a logger adapter with bound context fields rendered as key=value."""
    return _ContextLogger(logging.getLogger(name), bind)


class _ContextLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.extra:
            ctx = " ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"{msg} | {ctx}"
        return msg, kwargs
