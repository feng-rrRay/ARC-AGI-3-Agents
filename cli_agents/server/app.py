"""ARC-AGI-3 game server for Hermes CLI evaluation.

Owns one Arcade instance, one open scorecard, and one EnvironmentWrapper for a
single game. Exposes the ARC turn-based environment as HTTP endpoints consumed by
arc_mcp_server.py. The container firewall blocks direct access to this server;
agents interact only through the MCP proxy.

Usage:
    python -m cli_agents.server.app --game <game_id> --port 8000
"""
from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import logging
import os
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path setup so helpers can be imported from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from agents.templates.continual_harness.context import pretty_print_3d
from agents.templates.continual_harness.helpers import (
    _action_from_name,
    available_game_actions,
    describe_action,
    grid_to_image,
    validate_action_sequence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (single-tenant; one server process per game per run)
# ---------------------------------------------------------------------------
_arc: Arcade | None = None
_card_id: str | None = None
_env = None  # EnvironmentWrapper
_latest: dict[str, Any] | None = None  # cached converted frame
_budget: int = 0
_win_levels: Any = None
_game_id: str = ""
_tags: list[str] = []
_run_dir: Path | None = None
_recordings_dir: Path | None = None
_scorecard_closed = False
_scorecard_closing = False
_scorecard_payload: dict[str, Any] | None = None
_state_lock = threading.Lock()

ARC_MAX_ACTIONS = int(os.environ.get("ARC_MAX_ACTIONS", "5000"))
UPSCALE_FACTOR = int(os.environ.get("ARC_IMAGE_UPSCALE", "8"))

app = FastAPI(title="arc-game-server")


# ---------------------------------------------------------------------------
# Frame conversion helpers
# ---------------------------------------------------------------------------

def _convert_raw(raw: Any) -> dict[str, Any]:
    """Convert FrameDataRaw to a JSON-serialisable dict, mirroring agent.py."""
    frame: list[list[list[int]]] = [arr.tolist() for arr in raw.frame]
    avail = list(raw.available_actions or [])
    return {
        "game_id": raw.game_id or _game_id,
        "frame": frame,
        "state": raw.state.name if raw.state else GameState.NOT_PLAYED.name,
        "levels_completed": int(raw.levels_completed or 0),
        "win_levels": raw.win_levels,
        "available_actions": avail,
        "guid": raw.guid or "",
        "full_reset": bool(raw.full_reset),
    }


def _render_screenshot(frame: list[list[list[int]]]) -> str:
    """Render the first (top) grid layer to an upscaled PNG; return base64."""
    if not frame or not frame[0]:
        return ""
    grid = frame[-1]
    img = grid_to_image(grid)
    if UPSCALE_FACTOR > 1:
        new_w = img.width * UPSCALE_FACTOR
        new_h = img.height * UPSCALE_FACTOR
        from PIL import Image as _Image
        img = img.resize((new_w, new_h), _Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frame_payload(cached: dict[str, Any]) -> dict[str, Any]:
    """Build the get_game_state response payload from a cached converted frame."""
    frame: list[list[list[int]]] = cached["frame"]
    avail_ids: list[int] = cached["available_actions"]
    avail_actions = available_game_actions(avail_ids)
    action_menu = [
        {
            "name": a.name,
            "description": describe_action(a),
            "needs_xy": a is GameAction.ACTION6,
        }
        for a in avail_actions
    ]
    return {
        "game_id": cached["game_id"],
        "state": cached["state"],
        "levels_completed": cached["levels_completed"],
        "win_levels": cached["win_levels"],
        "available_actions": [a.name for a in avail_actions],
        "action_menu": action_menu,
        "state_text": pretty_print_3d(frame),
        "screenshot_base64": _render_screenshot(frame),
        "num_layers": len(frame),
        "guid": cached["guid"],
    }


def _compact_names(names: list[str], head: int = 15, tail: int = 15) -> list[str]:
    """Echo applied-action names compactly: collapse long batches to head + an
    elision marker + tail so a take_actions response never balloons."""
    if len(names) <= head + tail:
        return names
    omitted = len(names) - head - tail
    return names[:head] + [f"...({omitted} more)..."] + names[-tail:]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "game_id": _game_id}


@app.post("/mcp/get_game_state")
async def mcp_get_game_state() -> JSONResponse:
    with _state_lock:
        cached = _latest
    if cached is None:
        return JSONResponse({"success": False, "error": "environment not initialised"})
    payload = _frame_payload(cached)
    return JSONResponse(payload)


@app.post("/mcp/take_actions")
async def mcp_take_actions(request: Request) -> JSONResponse:
    global _latest, _budget, _scorecard_closed

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "invalid JSON body"})

    raw_actions = body.get("actions", [])
    outer_reasoning = body.get("reasoning", "")

    with _state_lock:
        cached = _latest
        bud = _budget
        env = _env
        closed = _scorecard_closed

    if cached is None or env is None:
        return JSONResponse({"success": False, "error": "environment not initialised"})
    if closed:
        return JSONResponse({"success": False, "error": "run already finished"})
    if bud <= 0:
        return JSONResponse({
            "success": False,
            "error": "action budget exhausted",
            "budget_remaining": 0,
            "state": cached["state"],
            "available_actions": [
                a.name for a in available_game_actions(cached["available_actions"])
            ],
            "done": True,
        })

    # Normalize "action" key → "name" so validate_action_sequence works
    normalised: list[Any] = []
    for item in raw_actions:
        if isinstance(item, dict) and "action" in item and "name" not in item:
            item = {**item, "name": item.pop("action")}
        normalised.append(item)

    avail_actions = available_game_actions(cached["available_actions"])
    # Include RESET in the valid set for take_actions
    all_actions = list(avail_actions)
    if GameAction.RESET not in all_actions:
        all_actions.append(GameAction.RESET)

    steps, rejected = validate_action_sequence(normalised, all_actions)

    applied: list[dict[str, Any]] = []
    final_cached = cached

    for step in steps:
        with _state_lock:
            bud = _budget
            if bud <= 0:
                rejected.insert(0, type("R", (), {
                    "to_dict": lambda self: {
                        "raw": step.raw,
                        "reason": "budget exhausted mid-sequence",
                        "position": step.position,
                    }
                })())
                break
            action = step.action
            data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
            reasoning_text = getattr(action, "reasoning", None) or outer_reasoning
            step_reasoning = {"reasoning": reasoning_text} if reasoning_text else {}
            try:
                raw = env.step(action, data=data, reasoning=step_reasoning)
            except Exception as exc:
                logger.error("env.step failed: %s", exc)
                return JSONResponse({
                    "success": False,
                    "error": f"env.step error: {exc}",
                    "applied_count": len(applied),
                    "applied_actions": _compact_names([a["name"] for a in applied]),
                    "rejected": [r.to_dict() for r in rejected],
                })
            _budget -= 1
            bud = _budget
            if raw is not None:
                new_cached = _convert_raw(raw)
                _latest = new_cached
                final_cached = new_cached

        prev_levels = applied[-1]["levels_completed"] if applied else cached["levels_completed"]
        applied.append({
            "name": action.name,
            "position": step.position,
            "reasoning": reasoning_text,
            "state": final_cached["state"],
            "levels_completed": final_cached["levels_completed"],
            "budget_remaining": bud,
        })

        state_name = final_cached["state"]
        done = (
            state_name in (GameState.WIN.name, GameState.GAME_OVER.name)
            or bud <= 0
            or final_cached["levels_completed"] != prev_levels
        )
        if done:
            break

    avail_names = [a.name for a in available_game_actions(final_cached["available_actions"])]
    state_name = final_cached["state"]
    done = (
        state_name in (GameState.WIN.name, GameState.GAME_OVER.name)
        or _budget <= 0
    )
    # Compact response: echo only action NAMES (capped), not the full per-action
    # dicts. A verbose per-action echo grows unbounded with batch size (seen at
    # ~100 KB), gets truncated by the agent runtime, and destroys feedback. The
    # detailed per-step trajectory lives in the arc_agi recording + the Hermes
    # trajectory log; the agent re-observes the grid via get_game_state.
    applied_names = [a["name"] for a in applied]
    echo = _compact_names(applied_names)
    return JSONResponse({
        "success": True,
        "applied_count": len(applied_names),
        "applied_actions": echo,
        "levels_gained": final_cached["levels_completed"] - cached["levels_completed"],
        "rejected": [r.to_dict() for r in rejected],
        "state": state_name,
        "levels_completed": final_cached["levels_completed"],
        "available_actions": avail_names,
        "budget_remaining": _budget,
        "done": done,
        "note": "Call get_game_state to observe the resulting grid.",
    })


@app.get("/termination_condition")
async def termination_condition() -> JSONResponse:
    with _state_lock:
        cached = _latest
        bud = _budget
    if cached is None:
        return JSONResponse({"condition_met": False, "error": "not initialised"})
    state = cached["state"]
    levels = cached["levels_completed"]
    win_lvls = cached["win_levels"]
    # Terminal ONLY on WIN or budget exhaustion. GAME_OVER is recoverable in
    # ARC — the agent should RESET and keep trying within its action budget
    # (mirrors agents/agent.py, where is_done == WIN only). Treating GAME_OVER
    # as terminal would kill runs the agent could still win.
    done = (
        state == GameState.WIN.name
        or bud <= 0
        or (win_lvls is not None and levels >= win_lvls)
    )
    return JSONResponse({
        "condition_met": done,
        "state": state,
        "levels_completed": levels,
        "win_levels": win_lvls,
        "budget_remaining": bud,
    })


@app.post("/sync_llm_metrics")
async def sync_llm_metrics(request: Request) -> JSONResponse:
    """Stub — logs metrics; full implementation is a phase-2 parity item."""
    try:
        data = await request.json()
        logger.debug("sync_llm_metrics: %s", json.dumps(data)[:200])
    except Exception:
        pass
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Scorecard lifecycle
# ---------------------------------------------------------------------------

def _flatten_scorecard_recordings(
    recordings_dir: Path | None,
    card_id: str,
) -> list[Path]:
    """Move ARC recordings out of the scorecard-id subdirectory."""
    if recordings_dir is None:
        return []
    scorecard_dir = recordings_dir / card_id
    if not scorecard_dir.is_dir():
        return []

    moved: list[Path] = []
    for source in sorted(scorecard_dir.iterdir()):
        if not source.is_file():
            continue
        target = _unique_recording_target(recordings_dir / source.name, card_id)
        shutil.move(str(source), str(target))
        moved.append(target)

    try:
        scorecard_dir.rmdir()
    except OSError:
        logger.warning("Recording scorecard directory is not empty: %s", scorecard_dir)

    if moved:
        logger.info(
            "Moved %d recording file(s) directly under %s",
            len(moved),
            recordings_dir,
        )
    return moved


def _unique_recording_target(target: Path, card_id: str) -> Path:
    if not target.exists():
        return target

    suffix = "".join(target.suffixes)
    stem = target.name[: -len(suffix)] if suffix else target.name
    candidate = target.with_name(f"{stem}.{card_id}{suffix}")
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = target.with_name(f"{stem}.{card_id}.{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _close_scorecard() -> dict[str, Any] | None:
    global _scorecard_closed, _scorecard_closing, _scorecard_payload
    with _state_lock:
        if _scorecard_closed:
            return _scorecard_payload
        if _scorecard_closing:
            return _scorecard_payload
        if _arc is None or _card_id is None:
            return _scorecard_payload
        _scorecard_closing = True
        arc = _arc
        card_id = _card_id
        run_dir = _run_dir
        recordings_dir = _recordings_dir

    try:
        scorecard = arc.close_scorecard(card_id)
        if scorecard is None:
            logger.warning("close_scorecard returned None (card_id=%s)", card_id)
            return None
        payload: dict[str, Any] = scorecard.model_dump()
        if arc.operation_mode == OperationMode.ONLINE:
            base = os.environ.get("ARC_BASE_URL", "https://three.arcprize.org")
            payload["scorecard_url"] = f"{base}/scorecards/{card_id}"
        else:
            payload["scorecard_url"] = None
        _flatten_scorecard_recordings(recordings_dir, card_id)
        if run_dir:
            out = run_dir / "scorecard.json"
            out.write_text(json.dumps(payload, default=str, indent=2))
            logger.info("Scorecard written to %s", out)
        logger.info(
            "Scorecard closed: score=%.4f card_id=%s",
            payload.get("score", 0),
            card_id,
        )
        with _state_lock:
            _scorecard_payload = payload
            _scorecard_closed = True
        return payload
    except Exception as exc:
        logger.error("Error closing scorecard: %s", exc)
        return None
    finally:
        with _state_lock:
            _scorecard_closing = False


@app.post("/close_scorecard")
async def close_scorecard() -> JSONResponse:
    """Close the open scorecard explicitly and return the final payload.

    The orchestrator calls this before terminating the server so Ctrl-C and
    normal shutdown use the same scorecard path.
    """
    payload = _close_scorecard()
    if payload is None:
        return JSONResponse(
            {"success": False, "error": "scorecard was not closed"},
            status_code=500,
        )
    return JSONResponse({"success": True, "scorecard": payload})


def _signal_handler(signum: int, _frame: Any) -> None:
    logger.info("Signal %s received — closing scorecard and exiting", signum)
    _close_scorecard()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    global _arc, _card_id, _env, _latest, _budget, _win_levels
    global _game_id, _tags, _run_dir, _recordings_dir

    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="ARC game server for Hermes eval")
    parser.add_argument("--game", required=True, help="ARC game_id to play")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory to write scorecard.json",
    )
    parser.add_argument("--tags", default="", help="Comma-separated scorecard tags")
    parser.add_argument("--recordings-dir", default=None,
                        help="Directory for arc_agi gameplay recordings (JSONL per run)")
    args = parser.parse_args()

    _game_id = args.game
    _tags = []
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()] + [
        "hermes-eval",
        args.game,
    ]:
        if tag not in _tags:
            _tags.append(tag)
    if args.run_dir:
        _run_dir = Path(args.run_dir)
        _run_dir.mkdir(parents=True, exist_ok=True)

    recordings_dir = args.recordings_dir or os.environ.get("ARC_RECORDINGS_DIR")

    # Honour OPERATION_MODE / legacy ONLINE_ONLY env vars (arc_agi reads them).
    # recordings_dir + save_recording=True make arc_agi write a JSONL recording
    # of every frame/action to {recordings_dir}/{scorecard_id}/. The scorecard
    # directory is flattened after close_scorecard so completed Hermes runs expose
    # recordings directly under recordings/.
    arc_kwargs: dict[str, Any] = {}
    if recordings_dir:
        _recordings_dir = Path(recordings_dir)
        _recordings_dir.mkdir(parents=True, exist_ok=True)
        arc_kwargs["recordings_dir"] = str(_recordings_dir)
    _arc = Arcade(**arc_kwargs)
    _card_id = _arc.open_scorecard(tags=_tags)
    logger.info("Opened scorecard %s for game %s", _card_id, _game_id)

    env = _arc.make(_game_id, scorecard_id=_card_id,
                    save_recording=bool(recordings_dir))
    if env is None:
        logger.error("arc.make returned None for game %s", _game_id)
        sys.exit(1)
    _env = env
    if recordings_dir:
        logger.info(
            "Recording gameplay to %s/%s/ until scorecard close",
            recordings_dir,
            _card_id,
        )

    raw = getattr(env, "observation_space", None)
    if raw is None:
        raw = env.reset()
    if raw is None:
        logger.error("initial environment observation returned None")
        sys.exit(1)
    _latest = _convert_raw(raw)
    _win_levels = _latest["win_levels"]
    _budget = ARC_MAX_ACTIONS
    logger.info("Game %s ready; budget=%d win_levels=%s", _game_id, _budget, _win_levels)

    atexit.register(_close_scorecard)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
