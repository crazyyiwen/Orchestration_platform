"""Tools API surface (spec §13)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.tools.registry import ToolRegistry

router = APIRouter(tags=["tools"])


def _tools(request: Request) -> ToolRegistry:
    tr = getattr(request.app.state, "tool_registry", None)
    if tr is None:
        raise HTTPException(status_code=503, detail="tool registry not initialized")
    return tr


@router.get("/api/tools")
async def list_tools(request: Request) -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in _tools(request).list()
        ]
    }


class TestToolBody(BaseModel):
    name: str
    args: dict[str, Any] = {}


@router.post("/api/tools/test")
async def test_tool(body: TestToolBody, request: Request) -> dict[str, Any]:
    tr = _tools(request)
    if not tr.has(body.name):
        raise HTTPException(status_code=404, detail=f"unknown tool {body.name!r}")
    result = await tr.get(body.name).execute(body.args)
    return {"tool": body.name, "result": result}
