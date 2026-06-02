"""FastMCP proxy exposing exactly two ARC game tools to Hermes.

Forwards calls to the ARC game server (cli_agents/server/app.py) over HTTP.
Hermes uses the streamable-http endpoint (/mcp); other clients can use /sse.

Run via:
    python -m cli_agents.server.arc_mcp_server
or:
    MCP_PORT=8002 ARC_GAME_SERVER_URL=http://localhost:8000 python -m cli_agents.server.arc_mcp_server
"""
from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage

logger = logging.getLogger(__name__)

_GAME_SERVER_URL = os.environ.get("ARC_GAME_SERVER_URL", "http://localhost:8000")
_MCP_PORT = int(os.environ.get("MCP_PORT", "8002"))
_MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
_TIMEOUT = int(os.environ.get("ARC_MCP_TIMEOUT", "30"))

mcp = FastMCP(name="arc-agi-3", host=_MCP_HOST, port=_MCP_PORT)


def _post(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{_GAME_SERVER_URL}{path}"
    try:
        r = requests.post(url, json=body or {}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        logger.error("MCP proxy _post %s failed: %s", path, exc)
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_game_state() -> list[Any]:
    """Get the current ARC game state.

    Returns the grid as a labelled text description plus a rendered PNG image
    so you can visually inspect the current puzzle state. Call this whenever
    you need to observe the game before or after taking actions.

    Returns:
      - state_text: multi-layer integer grid rendered as labelled text rows
      - available_actions: list of valid action names (and whether each needs x,y)
      - state: current game state (NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER)
      - levels_completed, budget_remaining, guid
      - A PNG image of the grid (the next content block)
    """
    result = _post("/mcp/get_game_state")
    b64 = result.pop("screenshot_base64", None)
    if b64:
        result["_screenshot_note"] = (
            "The next content block is the PNG image of the current grid. "
            "Use it to visually inspect cell colours and layout."
        )
        img_bytes = base64.b64decode(b64)
        logger.info(
            "get_game_state: state=%s levels=%s budget=%s image=%dKB",
            result.get("state"),
            result.get("levels_completed"),
            result.get("budget_remaining"),
            len(img_bytes) // 1024,
        )
        return [result, MCPImage(data=img_bytes, format="png")]
    return [result]


@mcp.tool()
def take_actions(actions: list[dict[str, Any]], reasoning: str = "") -> dict[str, Any]:
    """Apply an ordered list of ARC actions.

    Actions are applied in sequence and stop early on the first invalid action,
    budget exhaustion, level change, or terminal state. After calling this tool,
    call get_game_state to observe the result.

    Each action item must have:
      - name (str): one of the available action names (ACTION1..ACTION7, RESET)
      - reasoning (str): brief explanation of why this action (required)
      - x, y (int, 0..63): required only for ACTION6 (click), x=column y=row

    Args:
      actions: list of action dicts, e.g.
               [{"name": "ACTION1", "reasoning": "testing effect"},
                {"name": "ACTION6", "x": 5, "y": 3, "reasoning": "click cell"}]
      reasoning: optional overall reasoning for this batch

    Returns:
      applied: list of successfully executed actions with resulting state
      rejected: first invalid/rejected action and reason (if any)
      state, levels_completed, available_actions, budget_remaining, done
    """
    return _post("/mcp/take_actions", {"actions": actions, "reasoning": reasoning})


def _run_combined_transport() -> None:
    """Serve both SSE (/sse) and Streamable HTTP (/mcp). Hermes uses /mcp."""
    import anyio
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount

    streamable_app = mcp.streamable_http_app()
    sse_app = mcp.sse_app()

    combined = Starlette(
        debug=False,
        routes=list(sse_app.routes) + list(streamable_app.routes),
        lifespan=streamable_app.router.lifespan_context,
    )
    config = uvicorn.Config(combined, host=_MCP_HOST, port=_MCP_PORT, log_level="warning")

    async def _serve() -> None:
        server = uvicorn.Server(config)
        await server.serve()

    anyio.run(_serve)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    transport = os.environ.get("MCP_TRANSPORT", "combined").lower()
    logger.info("Starting ARC MCP server on %s:%s (transport=%s)", _MCP_HOST, _MCP_PORT, transport)
    if transport == "sse":
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        _run_combined_transport()
