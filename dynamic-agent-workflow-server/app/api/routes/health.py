from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import settings_dep
from app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request, settings: Settings = Depends(settings_dep)) -> dict[str, object]:
    mongo = getattr(request.app.state, "mongo", None)
    mongo_status: str
    if mongo is None:
        mongo_status = "unconfigured"
    elif mongo.available:
        mongo_status = "ok"
    else:
        mongo_status = "unavailable"

    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "env": settings.APP_ENV,
        "version": "0.1.0",
        "mongo": mongo_status,
    }
