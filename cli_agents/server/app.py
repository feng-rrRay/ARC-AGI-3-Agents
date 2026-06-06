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

from agents.templates.continual_harness.context import (
    _grid_diff_stats,
    build_observation_section,
    current_state_rendered_grid,
    pretty_print_grid,
)
from agents.templates.continual_harness.helpers import (
    _action_from_name,
    available_game_actions,
    describe_action,
    grid_to_image,
    validate_action_sequence,
)
from agents.templates.continual_harness.models import PendingActionObservation

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
_observations_dir: Path | None = None  # logs/<run>/<game>/logs/observations/ (rendered PNGs)
_state_file: Path | None = None        # <game>/scratch/arc_state/latest_frame.json (-> /workspace in-container)
_scorecard_closed = False
_scorecard_closing = False
_scorecard_payload: dict[str, Any] | None = None
_state_lock = threading.Lock()

# Observation tracking: the frames + per-action observations accumulated since
# the last get_game_state call (the ContinualHarness `_pending_observations`
# analog). Reset on every get_game_state flush so memory stays bounded to one
# observe window rather than the whole game.
_obs_window: list[Any] = []                       # _ObsFrame list; index 0 is the pre-batch baseline
_pending_obs: list[PendingActionObservation] = []
_batch_counter: int = 0
_action_counter: int = 0
_observe_counter: int = 0                          # get_game_state calls, for step-sequenced image names

ARC_MAX_ACTIONS = int(os.environ.get("ARC_MAX_ACTIONS", "50000"))
UPSCALE_FACTOR = int(os.environ.get("ARC_IMAGE_UPSCALE", "8"))
# Full per-action RESULT/keyframe detail is rendered for the last N actions in
# the observe window; older ones collapse to a one-line summary (the grid count
# is separately capped by context.MAX_OBSERVATION_TEXT_GRIDS=4).
MAX_OBSERVATION_DETAIL_BLOCKS = 15

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


def _render_grid_png(grid: list[list[int]]) -> str:
    """Render one 2D grid to an upscaled PNG; return base64."""
    if not grid or not grid[0]:
        return ""
    img = grid_to_image(grid)
    if UPSCALE_FACTOR > 1:
        from PIL import Image as _Image
        img = img.resize(
            (img.width * UPSCALE_FACTOR, img.height * UPSCALE_FACTOR),
            _Image.NEAREST,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _ObsState:
    """Minimal stand-in for FrameData.state — exposes `.name`."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class _ObsFrame:
    """Attribute-access adapter over a converted-frame dict.

    The ContinualHarness renderers (build_observation_section /
    current_state_rendered_grid) read `.frame`, `.state.name`, and
    `.levels_completed`; the server stores converted dicts, so this shim bridges
    dict->attribute without a real arcengine.FrameData (and without editing the
    shared context.py).
    """

    __slots__ = ("frame", "levels_completed", "state")

    def __init__(self, cached: dict[str, Any]) -> None:
        self.frame = cached["frame"]
        self.levels_completed = cached["levels_completed"]
        self.state = _ObsState(cached["state"])


def _collapse_summary(
    obs_list: list[PendingActionObservation],
    window: list[_ObsFrame],
) -> str:
    """One-line summary for observations beyond the per-window detail cap."""
    names = _compact_names([o.action_name for o in obs_list])
    first_pre = window[obs_list[0].pre_frame_index]
    last_post = window[obs_list[-1].post_frame_index]
    pre_grid = first_pre.frame[-1] if first_pre.frame else None
    post_grid = last_post.frame[-1] if last_post.frame else None
    changed = _grid_diff_stats(pre_grid, post_grid).count
    return (
        f"EARLIER ACTIONS (collapsed): {len(obs_list)} actions {names}; "
        f"state {first_pre.state.name}->{last_post.state.name}; "
        f"score {first_pre.levels_completed}->{last_post.levels_completed}; "
        f"net_cells_changed={changed}"
    )


def _frame_payload(
    cached: dict[str, Any],
    pending: list[PendingActionObservation],
    window: list[_ObsFrame],
    latest_adapter: _ObsFrame,
) -> dict[str, Any]:
    """Build the get_game_state response: observations-since-last-query plus the
    current grid rendered once, with one PNG per rendered grid (same order)."""
    avail_actions = available_game_actions(cached["available_actions"])
    action_menu = [
        {
            "name": a.name,
            "description": describe_action(a),
            "needs_xy": a is GameAction.ACTION6,
        }
        for a in avail_actions
    ]

    # OBSERVATIONS SINCE LAST QUERY — full RESULT/keyframe blocks for the last
    # MAX_OBSERVATION_DETAIL_BLOCKS actions; older ones collapse to one line.
    obs_text = ""
    obs_grids: list[Any] = []
    if pending:
        detail = pending[-MAX_OBSERVATION_DETAIL_BLOCKS:]
        collapsed = (
            pending[:-MAX_OBSERVATION_DETAIL_BLOCKS]
            if len(pending) > MAX_OBSERVATION_DETAIL_BLOCKS
            else []
        )
        obs_body, obs_grids = build_observation_section(detail, window)
        obs_text = (
            _collapse_summary(collapsed, window) + "\n\n" + obs_body
            if collapsed
            else obs_body
        )

    # CURRENT STATE grid (latest_frame.frame[-1]) rendered exactly once.
    current_grid = current_state_rendered_grid(latest_adapter)
    rendered = list(obs_grids)
    current_text = ""
    if current_grid is not None:
        rendered.append(current_grid)
        current_text = pretty_print_grid(current_grid.grid, current_grid.label)

    return {
        "game_id": cached["game_id"],
        "state": cached["state"],
        "levels_completed": cached["levels_completed"],
        "win_levels": cached["win_levels"],
        "available_actions": [a.name for a in avail_actions],
        "action_menu": action_menu,
        "observations_since_last_query": obs_text,
        "current_grid": current_text,
        "screenshots_base64": [_render_grid_png(rg.grid) for rg in rendered],
        # Parallel grid labels (e.g. action_step_3_frame_2, current_state_frame)
        # used by mcp_get_game_state to name the persisted PNGs; popped before the
        # payload is returned to the agent.
        "screenshot_labels": [rg.label for rg in rendered],
        "guid": cached["guid"],
    }


def _compact_names(names: list[str], head: int = 15, tail: int = 15) -> list[str]:
    """Echo applied-action names compactly: collapse long batches to head + an
    elision marker + tail so a take_actions response never balloons."""
    if len(names) <= head + tail:
        return names
    omitted = len(names) - head - tail
    return names[:head] + [f"...({omitted} more)..."] + names[-tail:]


def _write_state_file(cached: dict[str, Any] | None, budget: int) -> None:
    """Dump the machine-readable current frame to the mounted workspace.

    FULL-toolset code execution (Hermes `execute_code`) cannot call the ARC MCP
    tools and has no game state injected, so it would otherwise paste grids in as
    string literals. Writing the raw frame here lets sandboxed code load live
    state from /workspace/arc_state/latest_frame.json instead. Best-effort; never
    raises into the request path. This file never enters the model's context.
    """
    if _state_file is None or cached is None:
        return
    frame = cached.get("frame") or []
    try:
        avail = [a.name for a in available_game_actions(cached.get("available_actions") or [])]
    except Exception:
        avail = []
    payload = {
        "state": cached.get("state"),
        "levels_completed": cached.get("levels_completed"),
        "win_levels": cached.get("win_levels"),
        "available_actions": avail,
        "current_grid": frame[-1] if frame else None,
        "frame": frame,
        "budget_remaining": budget,
        "guid": cached.get("guid"),
    }
    try:
        _state_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write state file %s: %s", _state_file, exc)


def _record_step(
    pre_cached: dict[str, Any],
    new_cached: dict[str, Any],
    *,
    action_name: str,
    action_data: dict[str, Any] | None,
    batch_id: str,
    position: int,
    total: int,
    source: str,
) -> None:
    """Record one executed action's pre->post transition for the next
    get_game_state, and advance _latest.

    Shared by the take_actions step loop and the GAME_OVER auto-reset. The caller
    holds _state_lock and has already run env.step + _convert_raw + decremented
    _budget. The observe window is seeded with the pre-action frame as baseline
    (index 0) the first time an action runs after a flush.
    """
    global _latest, _action_counter
    if not _obs_window:
        _obs_window.append(_ObsFrame(pre_cached))
    pre_index = len(_obs_window) - 1
    _obs_window.append(_ObsFrame(new_cached))
    _action_counter += 1
    _pending_obs.append(PendingActionObservation(
        action_counter=_action_counter,
        action_name=action_name,
        action_data=action_data or {},
        source=source,
        batch_id=batch_id,
        batch_position=position,
        batch_total=total,
        pre_frame_index=pre_index,
        post_frame_index=len(_obs_window) - 1,
        valid_frame=True,
    ))
    _latest = new_cached


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "game_id": _game_id}


@app.post("/mcp/get_game_state")
async def mcp_get_game_state() -> JSONResponse:
    global _pending_obs, _obs_window, _observe_counter
    with _state_lock:
        cached = _latest
        if cached is None:
            return JSONResponse(
                {"success": False, "error": "environment not initialised"}
            )
        # Snapshot the observe window, then flush so the next take_actions starts
        # a fresh window seeded with the current frame as its baseline. Rendering
        # (diffing + PNG encoding) happens outside the lock on the captured refs.
        pending = _pending_obs
        window = _obs_window
        latest_adapter = _ObsFrame(cached)
        _pending_obs = []
        _obs_window = [latest_adapter]
        _observe_counter += 1
        observe_idx = _observe_counter
    payload = _frame_payload(cached, pending, window, latest_adapter)

    # Persist each rendered grid PNG with a step-sequenced, navigable name
    # (obs<NNNN>_<seq>_<label>.png) so the trace links to readable files instead
    # of Hermes's hash-named multimodal cache. The base64 still feeds the agent
    # (the proxy turns it into image blocks); these files are for offline traces.
    labels = payload.pop("screenshot_labels", [])
    screenshot_files: list[str] = []
    if _observations_dir is not None:
        for seq, b64 in enumerate(payload.get("screenshots_base64", [])):
            if not b64:
                continue
            label = labels[seq] if seq < len(labels) else f"grid_{seq}"
            fname = f"obs{observe_idx:04d}_{seq}_{label}.png"
            try:
                (_observations_dir / fname).write_bytes(base64.b64decode(b64))
                screenshot_files.append(f"observations/{fname}")
            except OSError as exc:
                logger.warning("could not write observation image %s: %s", fname, exc)
    payload["observe_index"] = observe_idx
    payload["screenshot_files"] = screenshot_files
    _write_state_file(cached, _budget)  # keep the code-sandbox state file in sync
    return JSONResponse(payload)


@app.post("/mcp/take_actions")
async def mcp_take_actions(request: Request) -> JSONResponse:
    global _latest, _budget, _scorecard_closed
    global _pending_obs, _obs_window, _batch_counter, _action_counter

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

    with _state_lock:
        _batch_counter += 1
        batch_id = f"b_{_batch_counter:04d}"
    batch_total = len(steps)

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
            pre_cached = final_cached
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
                _record_step(
                    pre_cached, new_cached,
                    action_name=action.name,
                    action_data={k: v for k, v in data.items() if k != "game_id"},
                    batch_id=batch_id,
                    position=step.position,
                    total=batch_total,
                    source="take_actions",
                )
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

    # --- Force-recover from GAME_OVER (server-side auto-reset) ---
    # GAME_OVER is recoverable, but agents waste actions probing a dead board and a
    # post-GAME_OVER frame can report win_levels=0 (which would falsely trip
    # /termination_condition). Reset immediately so the next observation is a fresh,
    # playable board. Mirrors ContinualHarness.main()'s auto_reset on GAME_OVER.
    auto_reset = False
    if final_cached["state"] == GameState.GAME_OVER.name:
        with _state_lock:
            if _budget > 0 and not _scorecard_closed:
                try:
                    raw = env.step(GameAction.RESET, data={},
                                   reasoning={"reasoning": "auto-reset after GAME_OVER"})
                except Exception as exc:
                    logger.error("auto-reset env.step failed: %s", exc)
                    raw = None
                if raw is not None:
                    _budget -= 1
                    new_cached = _convert_raw(raw)
                    _record_step(
                        final_cached, new_cached,
                        action_name="RESET",
                        action_data=None,
                        batch_id=batch_id,
                        position=batch_total + 1,
                        total=batch_total + 1,
                        source="auto_reset",
                    )
                    final_cached = new_cached
                    auto_reset = True
                    logger.info("Game %s: auto-reset after GAME_OVER (budget=%d)",
                                _game_id, _budget)

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
    _write_state_file(final_cached, _budget)  # refresh live state for the code sandbox
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
        "auto_reset": auto_reset,
        "note": (
            "You hit GAME_OVER; the board was automatically reset — "
            "call get_game_state and keep playing."
            if auto_reset else
            "Call get_game_state to observe the resulting grid."
        ),
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
        # win_lvls > 0 guards against the degenerate post-GAME_OVER frame that
        # reports win_levels=0 (0 >= 0 would otherwise falsely terminate the run).
        or (win_lvls is not None and win_lvls > 0 and levels >= win_lvls)
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
    global _game_id, _tags, _run_dir, _recordings_dir, _observations_dir, _state_file

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
        # Sits beside the host-written observations.jsonl under <game>/logs/.
        _observations_dir = _run_dir / "logs" / "observations"
        _observations_dir.mkdir(parents=True, exist_ok=True)
        # scratch/ is bind-mounted to /workspace in the agent container, so the
        # FULL-toolset code sandbox can read this live state file.
        _state_file = _run_dir / "scratch" / "arc_state" / "latest_frame.json"
        _state_file.parent.mkdir(parents=True, exist_ok=True)

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
    _write_state_file(_latest, _budget)  # seed the file before the first action
    logger.info("Game %s ready; budget=%d win_levels=%s", _game_id, _budget, _win_levels)

    atexit.register(_close_scorecard)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
