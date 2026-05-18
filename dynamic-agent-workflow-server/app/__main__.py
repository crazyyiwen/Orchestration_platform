"""Run the server using host/port from settings (``.env``).

Usage::

    python -m app                    # uses API_HOST/API_PORT from .env
    python -m app --reload           # adds uvicorn flags

This is just a thin uvicorn launcher that reads the same ``Settings`` the app
itself uses, so the port lives in one place: ``.env``. You can still bypass it
with the original ``uvicorn app.main:app --port N`` form if you want.
"""
from __future__ import annotations

import sys

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    # Anything after the first arg gets forwarded to uvicorn so flags like
    # --reload, --workers N, --log-level debug still work.
    extra_args = sys.argv[1:]
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload="--reload" in extra_args,
    )


if __name__ == "__main__":
    main()
