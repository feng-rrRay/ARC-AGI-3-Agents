#!/usr/bin/env python3
"""Replay a recorded ARC-AGI-3 game onto a fresh online scorecard.

When a run's scorecard never closed (e.g. the card expired -> 404 on close), the
per-frame recording under ``recordings/<card_id>/<game-guid>.jsonl`` still holds
the full ordered action sequence. This script opens a NEW online scorecard,
replays those actions through the same ``arc_agi`` Arcade/env path the live
server uses (see cli_agents/server/app.py), then closes it -- producing an
authentic online scorecard (card_id + ``/scorecards/<id>`` URL) and writing
``scorecard.json`` into the run directory.

Purely mechanical: no LLM, no FastAPI, no MCP. The score is computed server-side
on close, so it matches whatever the live API would report.

Usage:
    uv run python cli_agents/server/replay_scorecard.py <run_dir> [<run_dir> ...] \
        [--base-url https://three.arcprize.org] [--env-file .env]

A <run_dir> is a per-game directory containing ``recordings/``. Dirs that
already have ``scorecard.json`` are skipped. The host MUST be the one the
recording's game_id lives on (these recordings belong to three.arcprize.org).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _load_env(env_file: str | None) -> None:
    if not env_file:
        return
    p = Path(env_file)
    if not p.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(str(p))


def _find_recording(run_dir: Path) -> Path | None:
    recs = sorted(run_dir.glob("recordings/*/*.jsonl")) or sorted(
        run_dir.glob("recordings/*.jsonl")
    )
    return recs[0] if recs else None


def _load_recording(rec_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with rec_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _frame_fields(raw: Any) -> tuple[int, str]:
    """Mirror app.py:_convert_raw for the two fields we compare against."""
    from arcengine import GameState

    state = raw.state.name if raw.state else GameState.NOT_PLAYED.name
    return int(raw.levels_completed or 0), state


def _step_with_retry(env: Any, action: Any, data: dict[str, Any], retries: int = 6) -> Any:
    """env.step returns None on a failed POST (incl. 429); back off and retry.

    Backoff is 1,2,4,8,16,32s (~63s total) to ride out rate-limit windows.
    Run games one at a time -- parallel streams trip the API's 429 limiter.
    """
    for attempt in range(retries):
        raw = env.step(action, data=data, reasoning={})
        if raw is not None:
            return raw
        time.sleep(2 ** attempt)
    return None


def replay_one(run_dir: Path, base_url: str) -> bool:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    if (run_dir / "scorecard.json").exists():
        print(f"[skip] {run_dir.name}: scorecard.json already present")
        return True

    rec_path = _find_recording(run_dir)
    if rec_path is None:
        print(f"[skip] {run_dir.name}: no recording found")
        return False

    rows = _load_recording(rec_path)
    game_id = rows[0]["data"]["game_id"]
    orig_card = rec_path.parent.name
    rec_peak = max(int(r["data"].get("levels_completed") or 0) for r in rows)
    print(f"\n=== {run_dir.name} ===")
    print(f"  game_id={game_id}  frames={len(rows)}  recorded_peak_levels={rec_peak}")
    print(f"  source recording={rec_path}  (orig card {orig_card})")

    arc = Arcade()  # ONLINE via env: ARC_API_KEY / ARC_BASE_URL / OPERATION_MODE
    card_id = arc.open_scorecard(tags=["hermes-eval", "replay", game_id])
    print(f"  opened NEW scorecard {card_id}")

    env = arc.make(game_id, scorecard_id=card_id, save_recording=False)
    if env is None:
        print(f"  [error] arc.make returned None for {game_id} on {base_url}")
        return False

    # Initial reset (recording frame[0] is the post-reset frame).
    raw = getattr(env, "observation_space", None)
    if raw is None:
        raw = env.reset()
    if raw is None:
        print("  [error] initial reset returned no frame")
        return False

    mismatches = 0
    applied = 0
    last_levels = 0
    for i in range(1, len(rows)):
        ai = rows[i]["data"].get("action_input") or {}
        aid = ai.get("id")
        if aid is None:
            continue
        if aid == "RESET":
            action, data = GameAction.RESET, {}
        else:
            try:
                action = GameAction[aid]
            except KeyError:
                print(f"  [warn] unknown action id {aid!r} at frame {i}; skipping")
                continue
            d = ai.get("data") or {}
            data = {k: d[k] for k in ("x", "y") if k in d}

        raw = _step_with_retry(env, action, data)
        if raw is None:
            print(f"  [error] step failed at frame {i} (action {aid}); aborting this game")
            return False
        applied += 1

        lvls, state = _frame_fields(raw)
        last_levels = lvls
        exp_lvls = rows[i]["data"].get("levels_completed")
        exp_state = rows[i]["data"].get("state")
        if lvls != exp_lvls or state != exp_state:
            mismatches += 1
            if mismatches <= 5:
                print(
                    f"  [diverge] frame {i} action={aid}: "
                    f"got lvl={lvls}/{state} exp lvl={exp_lvls}/{exp_state}"
                )
        if applied % 1000 == 0:
            print(f"  ... {applied}/{len(rows) - 1} actions, levels={lvls}, mismatches={mismatches}")

    print(f"  replay done: applied={applied} replay_peak_levels~={last_levels} "
          f"mismatches={mismatches}/{applied}")

    sc = arc.close_scorecard(card_id)
    if sc is None:
        print(f"  [error] close_scorecard returned None for {card_id}")
        return False

    payload: dict[str, Any] = sc.model_dump()
    if arc.operation_mode == OperationMode.ONLINE:
        base = os.environ.get("ARC_BASE_URL", base_url)
        payload["scorecard_url"] = f"{base}/scorecards/{card_id}"
    payload["replayed_from_card_id"] = orig_card
    payload["replay_source_recording"] = str(rec_path)

    out = run_dir / "scorecard.json"
    out.write_text(json.dumps(payload, default=str, indent=2))
    print(f"  CLOSED: score={payload.get('score', 0):.4f}  card_id={card_id}")
    print(f"  url={payload.get('scorecard_url')}")
    print(f"  wrote {out}")
    if mismatches:
        print(f"  NOTE: {mismatches} frame(s) diverged from the recording "
              f"(replay may not perfectly reproduce; check score vs recorded peak {rec_peak}).")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="per-game run dir(s) containing recordings/")
    ap.add_argument("--base-url", default="https://three.arcprize.org",
                    help="ARC API host the recording's game_id lives on")
    ap.add_argument("--env-file", default=".env", help="dotenv file for ARC_API_KEY")
    args = ap.parse_args()

    _load_env(args.env_file)
    # Force the host + online mode regardless of the .env value (these recordings
    # belong to three.arcprize.org, not the .env's host).
    os.environ["ARC_BASE_URL"] = args.base_url
    os.environ["OPERATION_MODE"] = "online"
    if not os.environ.get("ARC_API_KEY"):
        raise SystemExit("ARC_API_KEY not set (provide via --env-file or environment)")
    print(f"Using ARC_BASE_URL={args.base_url}  OPERATION_MODE=online")

    ok, fail = [], []
    for d in args.run_dirs:
        rd = Path(d)
        try:
            (ok if replay_one(rd, args.base_url) else fail).append(rd.name)
        except Exception as exc:  # noqa: BLE001 - report and continue to next game
            print(f"  [error] {rd.name}: {exc!r}")
            fail.append(rd.name)

    print(f"\nDONE. ok={ok} failed={fail}")


if __name__ == "__main__":
    main()
