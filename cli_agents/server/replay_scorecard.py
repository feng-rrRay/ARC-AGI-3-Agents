#!/usr/bin/env python3
"""Replay recorded ARC-AGI-3 games onto fresh online scorecards.

When a run's scorecard never closed (e.g. the card expired -> 404 on close), the
per-frame recording under ``recordings/`` still holds the full ordered action
sequence. This script opens a NEW online scorecard, replays those actions
through the same ``arc_agi`` Arcade/env path the live server uses (see
cli_agents/server/app.py), then closes it -- producing an authentic online
scorecard (card_id + ``/scorecards/<id>`` URL).

Purely mechanical: no LLM, no FastAPI, no MCP. The score is computed server-side
on close, so it matches whatever the live API would report when the replay is
deterministic.

Usage:
    uv run python cli_agents/server/replay_scorecard.py <run_dir> [<run_dir> ...] \
        [--base-url https://three.arcprize.org] [--env-file .env]

    uv run python cli_agents/server/replay_scorecard.py logs/v1-logs/continualharness-* \
        --output logs/v1-logs/scorecard.online.json

A <run_dir> is a per-game directory containing ``recordings/``. Multiple run
dirs are replayed into one aggregate scorecard by default. Use
``--per-run-scorecards`` only when you explicitly want one scorecard per run dir;
in that mode, dirs that already have ``scorecard.json`` are skipped unless
``--force`` is set. The host MUST be the one the recording's game_id lives on.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCORECARD_TAGS = ["agent", "ContinualHarness"]


@dataclass(frozen=True, slots=True)
class ReplaySource:
    run_dir: Path
    rec_path: Path
    rows: list[dict[str, Any]]
    game_id: str
    original_card_id: str | None
    recorded_peak_levels: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    run_dir: str
    game_id: str | None
    success: bool
    applied: int = 0
    mismatches: int = 0
    recorded_peak_levels: int = 0
    replay_levels: int = 0
    replay_source_recording: str | None = None
    replayed_from_card_id: str | None = None
    error: str | None = None


def _load_env(env_file: str | None) -> None:
    if not env_file:
        return
    p = Path(env_file)
    if not p.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(str(p))


def _resolve_operation_mode(default: str = "online") -> str:
    from arc_agi import OperationMode

    raw = os.environ.get("OPERATION_MODE", "").strip()
    mode = raw.lower() if raw else default
    api_modes = {OperationMode.ONLINE.value, OperationMode.COMPETITION.value}
    if mode not in api_modes:
        valid = ", ".join(sorted(api_modes))
        raise SystemExit(f"OPERATION_MODE must be one of: {valid}")
    return mode


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


def _source_card_id(run_dir: Path, rec_path: Path) -> str | None:
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text()).get("card_id")
            if value:
                return str(value)
        except (OSError, ValueError):
            pass
    if rec_path.parent.name != "recordings":
        return rec_path.parent.name
    return None


def _recording_game_id(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        data = row.get("data") or {}
        game_id = data.get("game_id")
        if game_id:
            return str(game_id)
    raise ValueError("recording does not contain a frame with data.game_id")


def _recorded_peak_levels(rows: list[dict[str, Any]]) -> int:
    peak = 0
    for row in rows:
        data = row.get("data") or {}
        try:
            peak = max(peak, int(data.get("levels_completed") or 0))
        except (TypeError, ValueError):
            continue
    return peak


def _load_replay_source(run_dir: Path) -> ReplaySource | None:
    rec_path = _find_recording(run_dir)
    if rec_path is None:
        return None

    rows = _load_recording(rec_path)
    if not rows:
        raise ValueError(f"{rec_path} is empty")

    return ReplaySource(
        run_dir=run_dir,
        rec_path=rec_path,
        rows=rows,
        game_id=_recording_game_id(rows),
        original_card_id=_source_card_id(run_dir, rec_path),
        recorded_peak_levels=_recorded_peak_levels(rows),
    )


def _frame_fields(raw: Any) -> tuple[int, str]:
    """Mirror app.py:_convert_raw for the two fields we compare against."""
    from arcengine import GameState

    state = raw.state.name if raw.state else GameState.NOT_PLAYED.name
    return int(raw.levels_completed or 0), state


def _decode_action_input(action_input: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Return ``(GameAction, step_data)`` for legacy/new recording schemas."""
    from arcengine import GameAction

    aid = action_input.get("id")
    if aid is None:
        raise ValueError("action_input.id is missing")

    if isinstance(aid, bool):
        raise ValueError(f"invalid boolean action id {aid!r}")
    if isinstance(aid, int):
        action = GameAction.from_id(aid)
    elif isinstance(aid, str):
        stripped = aid.strip()
        if not stripped:
            raise ValueError("action_input.id is empty")
        if stripped.isdigit():
            action = GameAction.from_id(int(stripped))
        else:
            try:
                action = GameAction[stripped]
            except KeyError:
                try:
                    action = GameAction.from_name(stripped)
                except ValueError as exc:
                    raise ValueError(f"unknown action id {aid!r}") from exc
    else:
        raise ValueError(f"unsupported action id {aid!r}")

    recorded_data = action_input.get("data") or {}
    data = {
        key: recorded_data[key]
        for key in ("x", "y")
        if isinstance(recorded_data, dict) and key in recorded_data
    }
    return action, data


def _step_with_retry(
    env: Any, action: Any, data: dict[str, Any], retries: int = 6
) -> Any:
    """env.step returns None on a failed POST (incl. 429); back off and retry.

    Backoff is 1,2,4,8,16,32s (~63s total) to ride out rate-limit windows.
    Run games one at a time -- parallel streams trip the API's 429 limiter.
    """
    for attempt in range(retries):
        raw = env.step(action, data=data, reasoning={})
        if raw is not None:
            return raw
        time.sleep(2**attempt)
    return None


def _replay_source(source: ReplaySource, arc: Any, card_id: str) -> ReplayResult:
    print(f"\n=== {source.run_dir.name} ===")
    print(
        f"  game_id={source.game_id}  frames={len(source.rows)}  "
        f"recorded_peak_levels={source.recorded_peak_levels}"
    )
    print(
        f"  source recording={source.rec_path}  "
        f"(orig card {source.original_card_id or 'unknown'})"
    )

    env = arc.make(source.game_id, scorecard_id=card_id, save_recording=False)
    if env is None:
        error = f"arc.make returned None for {source.game_id}"
        print(f"  [error] {error}")
        return ReplayResult(
            run_dir=str(source.run_dir),
            game_id=source.game_id,
            success=False,
            recorded_peak_levels=source.recorded_peak_levels,
            replay_source_recording=str(source.rec_path),
            replayed_from_card_id=source.original_card_id,
            error=error,
        )

    # Initial reset (recording frame[0] is the post-reset frame).
    raw = getattr(env, "observation_space", None)
    if raw is None:
        raw = env.reset()
    if raw is None:
        error = "initial reset returned no frame"
        print(f"  [error] {error}")
        return ReplayResult(
            run_dir=str(source.run_dir),
            game_id=source.game_id,
            success=False,
            recorded_peak_levels=source.recorded_peak_levels,
            replay_source_recording=str(source.rec_path),
            replayed_from_card_id=source.original_card_id,
            error=error,
        )

    mismatches = 0
    applied = 0
    last_levels, _ = _frame_fields(raw)
    for i in range(1, len(source.rows)):
        data_row = source.rows[i].get("data") or {}
        ai = data_row.get("action_input") or {}
        aid = ai.get("id")
        if aid is None:
            continue
        try:
            action, data = _decode_action_input(ai)
        except ValueError as exc:
            print(f"  [warn] {exc} at frame {i}; skipping")
            continue

        raw = _step_with_retry(env, action, data)
        if raw is None:
            error = f"step failed at frame {i} (action {aid})"
            print(f"  [error] {error}; aborting this game")
            return ReplayResult(
                run_dir=str(source.run_dir),
                game_id=source.game_id,
                success=False,
                applied=applied,
                mismatches=mismatches,
                recorded_peak_levels=source.recorded_peak_levels,
                replay_levels=last_levels,
                replay_source_recording=str(source.rec_path),
                replayed_from_card_id=source.original_card_id,
                error=error,
            )
        applied += 1

        lvls, state = _frame_fields(raw)
        last_levels = lvls
        exp_lvls = data_row.get("levels_completed")
        exp_state = data_row.get("state")
        if (
            exp_lvls is not None
            and exp_state is not None
            and (lvls != exp_lvls or state != exp_state)
        ):
            mismatches += 1
            if mismatches <= 5:
                print(
                    f"  [diverge] frame {i} action={aid}: "
                    f"got lvl={lvls}/{state} exp lvl={exp_lvls}/{exp_state}"
                )
        if applied % 1000 == 0:
            print(
                f"  ... {applied}/{len(source.rows) - 1} actions, "
                f"levels={lvls}, mismatches={mismatches}"
            )

    print(
        f"  replay done: applied={applied} replay_peak_levels~={last_levels} "
        f"mismatches={mismatches}/{applied}"
    )
    if mismatches:
        print(
            f"  NOTE: {mismatches} frame(s) diverged from the recording "
            f"(check score vs recorded peak {source.recorded_peak_levels})."
        )
    return ReplayResult(
        run_dir=str(source.run_dir),
        game_id=source.game_id,
        success=True,
        applied=applied,
        mismatches=mismatches,
        recorded_peak_levels=source.recorded_peak_levels,
        replay_levels=last_levels,
        replay_source_recording=str(source.rec_path),
        replayed_from_card_id=source.original_card_id,
    )


def _scorecard_payload(
    arc: Any,
    scorecard: Any,
    card_id: str,
    base_url: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    from arc_agi import OperationMode

    payload: dict[str, Any] = scorecard.model_dump()
    if arc.operation_mode in (OperationMode.ONLINE, OperationMode.COMPETITION):
        base = os.environ.get("ARC_BASE_URL", base_url)
        payload["scorecard_url"] = f"{base}/scorecards/{card_id}"
    payload.update(extra)
    return payload


def replay_one(run_dir: Path, base_url: str, *, force: bool = False) -> bool:
    from arc_agi import Arcade

    out = run_dir / "scorecard.json"
    if out.exists() and not force:
        print(f"[skip] {run_dir.name}: scorecard.json already present")
        return True

    try:
        source = _load_replay_source(run_dir)
    except ValueError as exc:
        print(f"[skip] {run_dir.name}: {exc}")
        return False
    if source is None:
        print(f"[skip] {run_dir.name}: no recording found")
        return False

    arc = Arcade()  # API mode via env: ARC_API_KEY / ARC_BASE_URL / OPERATION_MODE
    card_id = arc.open_scorecard(tags=SCORECARD_TAGS)
    print(f"  opened NEW scorecard {card_id}")

    try:
        result = _replay_source(source, arc, card_id)
    except Exception as exc:  # noqa: BLE001 - close and persist partial payload.
        print(f"  [error] unexpected replay failure: {exc!r}")
        result = ReplayResult(
            run_dir=str(run_dir),
            game_id=source.game_id,
            success=False,
            recorded_peak_levels=source.recorded_peak_levels,
            replay_source_recording=str(source.rec_path),
            replayed_from_card_id=source.original_card_id,
            error=repr(exc),
        )
    sc = arc.close_scorecard(card_id)
    if sc is None:
        print(f"  [error] close_scorecard returned None for {card_id}")
        return False

    payload = _scorecard_payload(
        arc,
        sc,
        card_id,
        base_url,
        {
            "replayed_from_card_id": source.original_card_id,
            "replay_source_recording": str(source.rec_path),
            "replay_result": asdict(result),
        },
    )

    out.write_text(json.dumps(payload, default=str, indent=2))
    print(f"  CLOSED: score={payload.get('score', 0):.4f}  card_id={card_id}")
    print(f"  url={payload.get('scorecard_url')}")
    print(f"  wrote {out}")
    return result.success


def _default_single_scorecard_output(run_dirs: list[Path]) -> Path:
    resolved = [str(run_dir.resolve()) for run_dir in run_dirs]
    common = Path(os.path.commonpath(resolved))
    return common / "scorecard.online.json"


def replay_single_scorecard(
    run_dirs: list[Path],
    base_url: str,
    output_path: Path,
    *,
    force: bool = False,
) -> bool:
    from arc_agi import Arcade

    if output_path.exists() and not force:
        print(f"[skip] {output_path}: already present (use --force to overwrite)")
        return True

    sources: list[ReplaySource] = []
    results: list[ReplayResult] = []
    for run_dir in run_dirs:
        try:
            source = _load_replay_source(run_dir)
        except ValueError as exc:
            print(f"[skip] {run_dir.name}: {exc}")
            results.append(
                ReplayResult(
                    run_dir=str(run_dir),
                    game_id=None,
                    success=False,
                    error=str(exc),
                )
            )
            continue
        if source is None:
            print(f"[skip] {run_dir.name}: no recording found")
            results.append(
                ReplayResult(
                    run_dir=str(run_dir),
                    game_id=None,
                    success=False,
                    error="no recording found",
                )
            )
            continue
        sources.append(source)

    if not sources:
        print("[error] no replayable recordings found")
        return False

    arc = Arcade()  # API mode via env: ARC_API_KEY / ARC_BASE_URL / OPERATION_MODE
    card_id = arc.open_scorecard(tags=SCORECARD_TAGS)
    print(f"opened NEW aggregate scorecard {card_id} for {len(sources)} recording(s)")

    for source in sources:
        try:
            results.append(_replay_source(source, arc, card_id))
        except Exception as exc:  # noqa: BLE001 - keep remaining games replayable.
            print(f"  [error] {source.run_dir.name}: unexpected replay failure {exc!r}")
            results.append(
                ReplayResult(
                    run_dir=str(source.run_dir),
                    game_id=source.game_id,
                    success=False,
                    recorded_peak_levels=source.recorded_peak_levels,
                    replay_source_recording=str(source.rec_path),
                    replayed_from_card_id=source.original_card_id,
                    error=repr(exc),
                )
            )

    sc = arc.close_scorecard(card_id)
    if sc is None:
        print(f"  [error] close_scorecard returned None for {card_id}")
        return False

    payload = _scorecard_payload(
        arc,
        sc,
        card_id,
        base_url,
        {
            "replay_source_count": len(sources),
            "replay_results": [asdict(result) for result in results],
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, default=str, indent=2))
    print(
        f"\nCLOSED aggregate scorecard: score={payload.get('score', 0):.4f}  "
        f"card_id={card_id}"
    )
    print(f"url={payload.get('scorecard_url')}")
    print(f"wrote {output_path}")
    return all(result.success for result in results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dirs", nargs="+", help="per-game run dir(s) containing recordings/"
    )
    ap.add_argument(
        "--base-url",
        default="https://three.arcprize.org",
        help="ARC API host the recording's game_id lives on",
    )
    ap.add_argument("--env-file", default=".env", help="dotenv file for ARC_API_KEY")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-scorecard",
        action="store_true",
        help="Replay all provided run dirs into one new online scorecard",
    )
    mode.add_argument(
        "--per-run-scorecards",
        action="store_true",
        help="Replay each provided run dir into a separate scorecard",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output path for aggregate scorecard mode (default: common-dir/scorecard.online.json)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing replay scorecard output instead of skipping",
    )
    args = ap.parse_args()

    _load_env(args.env_file)
    # Force the host, but preserve OPERATION_MODE from .env so competition
    # submissions can set OPERATION_MODE=COMPETITION. Default remains online.
    os.environ["ARC_BASE_URL"] = args.base_url
    operation_mode = _resolve_operation_mode()
    os.environ["OPERATION_MODE"] = operation_mode
    if not os.environ.get("ARC_API_KEY"):
        raise SystemExit("ARC_API_KEY not set (provide via --env-file or environment)")
    print(f"Using ARC_BASE_URL={args.base_url}  OPERATION_MODE={operation_mode}")

    run_dirs = [Path(d) for d in args.run_dirs]
    aggregate_scorecard = args.single_scorecard or (
        len(run_dirs) > 1 and not args.per_run_scorecards
    )
    if aggregate_scorecard:
        if len(run_dirs) > 1 and not args.single_scorecard:
            print(
                "Multiple run dirs supplied; replaying them into one aggregate "
                "scorecard. Use --per-run-scorecards for separate scorecards."
            )
        output_path = (
            Path(args.output)
            if args.output
            else _default_single_scorecard_output(run_dirs)
        )
        ok = replay_single_scorecard(
            run_dirs,
            args.base_url,
            output_path,
            force=args.force,
        )
        print(f"\nDONE. ok={ok} output={output_path}")
        return

    if args.output:
        raise SystemExit(
            "--output is only supported for aggregate scorecards; remove "
            "--per-run-scorecards or add --single-scorecard"
        )

    ok, fail = [], []
    for rd in run_dirs:
        try:
            (ok if replay_one(rd, args.base_url, force=args.force) else fail).append(
                rd.name
            )
        except Exception as exc:  # noqa: BLE001 - report and continue to next game
            print(f"  [error] {rd.name}: {exc!r}")
            fail.append(rd.name)

    print(f"\nDONE. ok={ok} failed={fail}")


if __name__ == "__main__":
    main()
