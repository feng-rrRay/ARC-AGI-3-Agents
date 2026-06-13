from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from arcengine import FrameData, GameAction

from .action_descriptions import ACTION_DESCRIPTIONS
from .helpers import available_game_actions, color_to_hex, grid_to_hex_lines
from .models import PendingActionObservation, RenderedGrid, ToolCallRecord


CURRENT_STATE_GRID_LABEL = "current_state_frame"
MAX_OBSERVATION_TEXT_GRIDS = 4


def pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    """Render a 3D grid stack as hex maps, one dense hex string per row.

    Output matches ``state.latest_frame.frame`` exactly: each cell is a single
    hex digit ``0-f`` (= color int 0-15), and each row is one hex string — so the
    model sees the same representation in the prompt and in skill code. Recover
    the int with ``int(ch, 16)``.
    """
    lines: list[str] = []
    for i, block in enumerate(array_3d):
        if not block:
            lines.append(f"Grid {i}: (empty)")
            lines.append("")
            continue
        height = len(block)
        width = max((len(row) for row in block), default=0)
        lines.append(f"Grid {i} ({height}x{width}) [hex 0-f = color 0-15]:")
        lines.extend(grid_to_hex_lines(block))
        lines.append("")
    return "\n".join(lines)


def _normalise_grid(grid: Sequence[Sequence[Any]]) -> list[list[int]]:
    normalised: list[list[int]] = []
    for row in grid:
        normalised.append([int(v) for v in row])
    return normalised


def pretty_print_grid(grid: Sequence[Sequence[Any]], label: str) -> str:
    """Render one 2D grid as a hex map (one dense hex string per row).

    Each cell is a single hex digit ``0-f`` = color int 0-15; this matches the
    sandbox `state` view exactly. Recover the int with ``int(ch, 16)``.
    """
    rows = _normalise_grid(grid)
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    lines = [f"Grid {label} ({height}x{width}) [hex 0-f = color 0-15]:"]
    lines.extend(grid_to_hex_lines(rows))
    return "\n".join(lines)


def current_state_rendered_grid(latest_frame: FrameData) -> RenderedGrid | None:
    """Return the current-state grid that is rendered in the working prompt."""
    if not latest_frame.frame:
        return None
    return RenderedGrid(
        label=CURRENT_STATE_GRID_LABEL,
        grid=_normalise_grid(latest_frame.frame[-1]),
    )


@dataclass(slots=True)
class _DiffStats:
    count: int
    bbox: tuple[int, int, int, int] | None = None
    colors: set[int] | None = None


@dataclass(slots=True)
class _ObservationGridCandidate:
    observation_index: int
    frame_index: int
    grid: list[list[int]]
    role: str
    priority: tuple[int, int, int]


@dataclass(slots=True)
class _ObservationRenderPlan:
    observation_index: int
    observation: PendingActionObservation
    lines: list[str]
    candidates: list[_ObservationGridCandidate]


def _frame_state_name(frame: FrameData | None) -> str:
    if frame is None:
        return "?"
    state = getattr(frame, "state", None)
    name = getattr(state, "name", None)
    return str(name if name is not None else state)


def _frame_score(frame: FrameData | None) -> str:
    if frame is None:
        return "?"
    value = getattr(frame, "levels_completed", getattr(frame, "score", "?"))
    return str(value)


def _frame_at(frames: Sequence[FrameData], index: int) -> FrameData | None:
    if 0 <= index < len(frames):
        return frames[index]
    return None


def _last_grid(frame: FrameData | None) -> list[list[int]] | None:
    if frame is None or not getattr(frame, "frame", None):
        return None
    return _normalise_grid(frame.frame[-1])


def _grid_value(grid: list[list[int]], y: int, x: int) -> int | None:
    if y < 0 or y >= len(grid):
        return None
    row = grid[y]
    if x < 0 or x >= len(row):
        return None
    return row[x]


def _grid_diff_stats(
    before: list[list[int]] | None,
    after: list[list[int]] | None,
) -> _DiffStats:
    if before is None or after is None:
        return _DiffStats(count=0, bbox=None, colors=set())

    height = max(len(before), len(after))
    width = max(
        max((len(row) for row in before), default=0),
        max((len(row) for row in after), default=0),
    )
    count = 0
    min_r, max_r = height, -1
    min_c, max_c = width, -1
    colors: set[int] = set()

    for y in range(height):
        for x in range(width):
            old = _grid_value(before, y, x)
            new = _grid_value(after, y, x)
            if old == new:
                continue
            count += 1
            min_r = min(min_r, y)
            max_r = max(max_r, y)
            min_c = min(min_c, x)
            max_c = max(max_c, x)
            if old is not None:
                colors.add(int(old))
            if new is not None:
                colors.add(int(new))

    bbox = (min_r, max_r, min_c, max_c) if count else None
    return _DiffStats(count=count, bbox=bbox, colors=colors)


def _merge_bbox(
    bboxes: Iterable[tuple[int, int, int, int] | None],
) -> tuple[int, int, int, int] | None:
    present = [bbox for bbox in bboxes if bbox is not None]
    if not present:
        return None
    return (
        min(b[0] for b in present),
        max(b[1] for b in present),
        min(b[2] for b in present),
        max(b[3] for b in present),
    )


def _range_label(prefix: str, start: int, end: int) -> str:
    return f"{prefix}{start}" if start == end else f"{prefix}{start}-{end}"


def _format_bbox(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "none"
    r0, r1, c0, c1 = bbox
    return f"{_range_label('r', r0, r1)} {_range_label('c', c0, c1)}"


def _format_index_ranges(indices: Sequence[int]) -> str:
    if not indices:
        return "none"
    sorted_indices = sorted(set(indices))
    ranges: list[str] = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = idx
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _format_action(obs: PendingActionObservation) -> str:
    if not obs.action_data:
        return obs.action_name
    if set(obs.action_data) == {"x", "y"}:
        return f"{obs.action_name}(x={obs.action_data['x']},y={obs.action_data['y']})"
    args = ",".join(f"{k}={v}" for k, v in sorted(obs.action_data.items()))
    return f"{obs.action_name}({args})"


def _select_transient_keyframes(
    diff_to_pre: Sequence[_DiffStats],
    diff_to_prev: Sequence[_DiffStats],
    changed_indices: Sequence[int],
    *,
    limit: int = 3,
) -> list[int]:
    if not changed_indices or limit <= 0:
        return []

    first_changed = changed_indices[0]
    last_changed = changed_indices[-1]
    peak_changed = max(
        changed_indices,
        key=lambda i: (diff_to_pre[i].count, -i),
    )
    motion_candidates = [i for i in range(1, len(diff_to_prev))]
    max_motion = (
        max(motion_candidates, key=lambda i: (diff_to_prev[i].count, -i))
        if motion_candidates
        else peak_changed
    )

    chosen: list[int] = []
    for idx in (first_changed, peak_changed, last_changed, max_motion):
        if idx in changed_indices and idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= limit:
            return sorted(chosen)

    if len(chosen) < limit and len(changed_indices) > len(chosen):
        for slot in range(limit - len(chosen)):
            if limit - len(chosen) <= 0:
                break
            idx = changed_indices[
                round(slot * (len(changed_indices) - 1) / max(1, limit - 1))
            ]
            if idx not in chosen:
                chosen.append(idx)

    return sorted(chosen[:limit])


def _animation_summary(
    animation: Sequence[Sequence[Sequence[Any]]],
    pre_grid: list[list[int]] | None,
) -> tuple[str, bool, bool, list[int], list[_DiffStats], list[_DiffStats]]:
    grids = [_normalise_grid(grid) for grid in animation]
    if not grids or pre_grid is None:
        return (
            "ANIMATION SUMMARY: unavailable",
            False,
            False,
            [],
            [],
            [],
        )

    diff_to_pre = [_grid_diff_stats(pre_grid, grid) for grid in grids]
    diff_to_prev = [_DiffStats(count=0, bbox=None, colors=set())]
    for i in range(1, len(grids)):
        diff_to_prev.append(_grid_diff_stats(grids[i - 1], grids[i]))

    changed_indices = [i for i, stats in enumerate(diff_to_pre) if stats.count > 0]
    final_changed = bool(diff_to_pre[-1].count > 0)
    transient_changed = any(
        i != len(grids) - 1 and stats.count > 0
        for i, stats in enumerate(diff_to_pre)
    )
    if changed_indices:
        peak_changed = max(
            changed_indices,
            key=lambda i: (diff_to_pre[i].count, -i),
        )
        motion_indices = [i for i in range(1, len(diff_to_prev))]
        max_motion = (
            max(motion_indices, key=lambda i: (diff_to_prev[i].count, -i))
            if motion_indices
            else 0
        )
        bbox = _merge_bbox(diff_to_pre[i].bbox for i in changed_indices)
        colors: set[int] = set()
        for i in changed_indices:
            colors.update(diff_to_pre[i].colors or set())
        colors_text = "[" + ",".join(color_to_hex(c) for c in sorted(colors)) + "]"
        changed_text = (
            f"{_format_index_ranges(changed_indices)} "
            f"({len(changed_indices)}/{len(grids)})"
        )
        summary = (
            "ANIMATION SUMMARY: "
            f"frame_count={len(grids)}; "
            f"changed_frames={changed_text}; "
            f"peak_change={diff_to_pre[peak_changed].count} cells "
            f"at grid {peak_changed}; "
            f"max_motion={diff_to_prev[max_motion].count} cells "
            f"at grid {max_motion}; "
            f"bbox={_format_bbox(bbox)}; "
            f"colors_seen(hex)={colors_text}"
        )
    else:
        summary = (
            "ANIMATION SUMMARY: "
            f"frame_count={len(grids)}; changed_frames=none; "
            "peak_change=0 cells; max_motion=0 cells; "
            "bbox=none; colors_seen(hex)=[]"
        )

    return (
        summary,
        final_changed,
        transient_changed,
        changed_indices,
        diff_to_pre,
        diff_to_prev,
    )


def _build_observation_plan(
    obs: PendingActionObservation,
    frames: Sequence[FrameData],
    observation_index: int,
) -> _ObservationRenderPlan:
    pre = _frame_at(frames, obs.pre_frame_index)
    post = _frame_at(frames, obs.post_frame_index) if obs.valid_frame else None
    pre_grid = _last_grid(pre)
    animation = list(getattr(post, "frame", None) or [])
    summary, final_changed, transient_changed, changed_indices, diff_to_pre, diff_to_prev = (
        _animation_summary(animation, pre_grid)
    )

    lines = [
        f"### ACTION step {obs.action_counter}: {_format_action(obs)}",
        (
            "RESULT: "
            f"state {_frame_state_name(pre)} -> {_frame_state_name(post)}; "
            f"score {_frame_score(pre)} -> {_frame_score(post)}; "
            f"source={obs.source}; "
            f"final_grid_changed={'yes' if final_changed else 'no'}; "
            "transient_animation="
            f"{'yes' if transient_changed and not final_changed else 'no'}"
        ),
    ]
    if obs.batch_id and obs.batch_position is not None and obs.batch_total is not None:
        lines[-1] += (
            f"; batch={obs.batch_id}[{obs.batch_position}/{obs.batch_total}]"
        )

    candidates: list[_ObservationGridCandidate] = []
    if not obs.valid_frame:
        lines.append("ANIMATION SUMMARY: invalid action returned no frame")
        return _ObservationRenderPlan(observation_index, obs, lines, candidates)

    if len(animation) > 1:
        lines.append(summary)

    grids = [_normalise_grid(grid) for grid in animation]
    if final_changed and grids:
        frame_index = len(grids) - 1
        candidates.append(
            _ObservationGridCandidate(
                observation_index=observation_index,
                frame_index=frame_index,
                grid=grids[frame_index],
                role="final_changed_result",
                priority=(4, -obs.action_counter, frame_index),
            )
        )
    elif transient_changed and grids:
        selected = _select_transient_keyframes(
            diff_to_pre,
            diff_to_prev,
            changed_indices,
            limit=3,
        )
        peak = (
            max(changed_indices, key=lambda i: (diff_to_pre[i].count, -i))
            if changed_indices
            else -1
        )
        first = changed_indices[0] if changed_indices else -1
        last = changed_indices[-1] if changed_indices else -1
        for frame_index in selected:
            if frame_index == peak:
                rank, role = 0, "peak_changed"
            elif frame_index == first:
                rank, role = 1, "first_changed"
            elif frame_index == last:
                rank, role = 2, "last_changed"
            else:
                rank, role = 3, "motion_sample"
            candidates.append(
                _ObservationGridCandidate(
                    observation_index=observation_index,
                    frame_index=frame_index,
                    grid=grids[frame_index],
                    role=role,
                    priority=(rank, -obs.action_counter, frame_index),
                )
            )

    return _ObservationRenderPlan(observation_index, obs, lines, candidates)


def build_observation_section(
    observations: Sequence[PendingActionObservation],
    frames: Sequence[FrameData],
    *,
    max_text_grids: int = MAX_OBSERVATION_TEXT_GRIDS,
) -> tuple[str, list[RenderedGrid]]:
    """Render action observations and return exactly the grids shown as text."""
    if not observations:
        return "", []

    plans = [
        _build_observation_plan(obs, frames, index)
        for index, obs in enumerate(observations)
    ]
    all_candidates = [
        candidate for plan in plans for candidate in plan.candidates
    ]
    selected_keys = {
        (candidate.observation_index, candidate.frame_index)
        for candidate in sorted(all_candidates, key=lambda c: c.priority)[
            : max(0, max_text_grids)
        ]
    }

    lines: list[str] = [
        "The grids below are selected transition/result grids from actions "
        "executed after the previous VLM query.",
        "Grid indices refer to that action's returned frame list.",
    ]
    rendered_grids: list[RenderedGrid] = []
    for plan in plans:
        lines.append("")
        lines.extend(plan.lines)
        selected = [
            candidate
            for candidate in plan.candidates
            if (candidate.observation_index, candidate.frame_index) in selected_keys
        ]
        selected.sort(key=lambda c: c.frame_index)
        if selected:
            indices = [candidate.frame_index for candidate in selected]
            lines.append(f"RENDERED GRIDS: frame indices {indices}")
            for candidate in selected:
                label = (
                    f"action_step_{plan.observation.action_counter}"
                    f"_frame_{candidate.frame_index}"
                )
                rendered = RenderedGrid(label=label, grid=candidate.grid)
                rendered_grids.append(rendered)
                lines.append(pretty_print_grid(rendered.grid, rendered.label))
        elif plan.candidates:
            lines.append("RENDERED GRIDS: omitted by observation grid budget")
        else:
            lines.append("RENDERED GRIDS: none")

    return "\n".join(lines), rendered_grids




# Keys whose multi-line string values are lifted out of the JSON skeleton and
# rendered as fenced blocks so the model reads them print-style, not escaped.
_CODE_KEYS = {"code"}
_TEXT_KEYS = {"stdout", "stderr", "instructions", "body", "description", "error"}


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def _lift_long_text(
    value: Any, blocks: list[tuple[str, str, str]], key: str | None = None
) -> Any:
    """Recursively pull multi-line code/text string values out of ``value`` into
    ``blocks`` (label, lang, text), replacing each in the returned JSON skeleton
    with a short pointer. Keeps code (e.g. a skill's source) print-style in a
    ```python fence instead of a ``\\n``-escaped JSON string."""
    if isinstance(value, dict):
        return {k: _lift_long_text(v, blocks, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_lift_long_text(v, blocks, key) for v in value]
    if (
        isinstance(value, str)
        and key in (_CODE_KEYS | _TEXT_KEYS)
        and ("\n" in value or len(value) > 80)
    ):
        lang = "python" if key in _CODE_KEYS else ""
        label = f"{key} [{len(blocks) + 1}]"
        blocks.append((label, lang, value))
        return f"<{label} — printed below>"
    return value


def format_tool_record_md(r: ToolCallRecord) -> str:
    """Render one tool-call record as a labelled markdown block: the JSON fields
    plus any code/text lifted into fenced blocks. For run_skill the skill's source
    is included in the result and surfaces as a ```python section."""
    blocks: list[tuple[str, str, str]] = []
    args_skel = _lift_long_text(r.args, blocks)
    result_skel = _lift_long_text(r.result, blocks) if r.result is not None else None

    lines = [
        "### TOOL RESULT",
        f"name: {r.name}",
        f"args: {json.dumps(args_skel, default=str)}",
        "",
        "result:",
        _fence(json.dumps(result_skel, default=str, indent=2), "json")
        if result_skel is not None
        else "(none)",
        "",
        "error:",
        str(r.error) if r.error else "(none)",
        "",
        "actions_taken_inline:",
        str(r.actions_taken_inline),
    ]
    for label, lang, text in blocks:
        lines += ["", f"{label}:", _fence(text, lang)]
    return "\n".join(lines)


# How many of the newest carried-over tool results render as full blocks;
# everything older collapses to a one-line recap entry.
TOOL_RESULTS_FULL_LAST_N = 2


def _tool_call_brief(r: ToolCallRecord) -> str:
    """One-line ``name(args)`` summary with the reasoning argument dropped."""
    args = {k: v for k, v in (r.args or {}).items() if k != "reasoning"}
    text = json.dumps(args, default=str)
    if len(text) > 80:
        text = text[:77] + "..."
    return f"{r.name}({text})"


def _tool_outcome_brief(r: ToolCallRecord) -> str:
    """Compact outcome: ok/failed/error tag + the start of the main output."""
    if r.error:
        tag, text = "error", str(r.error)
    elif isinstance(r.result, dict):
        tag = "ok" if r.result.get("success", True) else "failed"
        text = (
            r.result.get("stdout")
            or r.result.get("error")
            or r.result.get("message")
            or json.dumps(r.result, default=str)
        )
    else:
        tag, text = "ok", json.dumps(r.result, default=str)
    text = " ".join(str(text).split())
    if len(text) > 160:
        text = f"{text[:160]}... [+{len(text) - 160} chars]"
    return f"{tag}: {text}"


def _tool_dedup_key(r: ToolCallRecord) -> str:
    args = {k: v for k, v in (r.args or {}).items() if k != "reasoning"}
    return json.dumps([r.name, args, r.result, r.error], default=str, sort_keys=True)


def format_tool_results_markdown(
    records: Sequence[ToolCallRecord],
    *,
    full_last_n: int = TOOL_RESULTS_FULL_LAST_N,
) -> str:
    """Render carried-over tool results from the previous turn's deliberation.

    The newest ``full_last_n`` records render as full blocks; older ones
    collapse to a one-line recap each. A call whose output is identical to an
    earlier call's collapses to a pointer, so the model sees that re-running
    it returned nothing new.
    """
    if not records:
        return "(none)"
    if full_last_n <= 0 or len(records) <= full_last_n:
        return "\n\n".join(format_tool_record_md(r) for r in records)

    cut = len(records) - full_last_n
    recap: list[str] = [
        f"{len(records)} tool calls ran during the previous turn's deliberation. "
        f"Recap below (oldest first); the last {full_last_n} results are printed "
        "in full. Do NOT re-run a recapped call to re-read its output — the full "
        "output was already shown when it ran."
    ]
    first_seen: dict[str, int] = {}
    for i, r in enumerate(records):
        first = first_seen.setdefault(_tool_dedup_key(r), i)
        if i >= cut:
            continue
        brief = _tool_call_brief(r)
        if first != i:
            recap.append(
                f"- {i + 1}. {brief} -> identical output to call {first + 1} "
                "(re-running returned nothing new)"
            )
        else:
            recap.append(f"- {i + 1}. {brief} -> {_tool_outcome_brief(r)}")
    full_blocks = [format_tool_record_md(r) for r in records[cut:]]
    return "\n".join(recap) + "\n\n" + "\n\n".join(full_blocks)


def _format_available_actions(actions: Sequence[GameAction]) -> str:
    """Per-game list of usable actions WITH their semantic meanings.

    Pulls descriptions from `action_descriptions.ACTION_DESCRIPTIONS`. ACTION6
    is special-cased to remind the model that x/y are required (the static
    description there only says "Click, Point").
    """
    if not actions:
        return "(none — wait for the next frame)"
    lines: list[str] = []
    for a in actions:
        if a is GameAction.ACTION6:
            desc = "Complex click — provide x (column 0-63) and y (row 0-63)."
        else:
            desc = ACTION_DESCRIPTIONS.get(a.name, "")
        lines.append(f"  {a.name}: {desc}" if desc else f"  {a.name}")
    return "\n" + "\n".join(lines)


def build_working_prompt(
    latest_frame: FrameData,
    *,
    action_counter: int,
    recent_tool_results: Sequence[ToolCallRecord],
    history_block: str,
    memory_overview: str,
    skill_overview: str,
    subagent_overview: str,
    observation_block: str = "",
    base_prompt: str = "",
    previous_no_action_reason: str | None = None,
    max_deliberation_turns: int | None = None,
) -> str:
    """Assemble the per-VLM-call working prompt."""
    available = available_game_actions(latest_frame.available_actions)
    current_grid = current_state_rendered_grid(latest_frame)
    frame_text = (
        pretty_print_grid(current_grid.grid, current_grid.label)
        if current_grid is not None
        else "(empty frame)"
    )
    sections: list[str] = []

    # Orchestrator policy
    if base_prompt.strip():
        sections.append(base_prompt.strip())
    
    # persistent components
    if subagent_overview.strip():
        sections.append(subagent_overview.rstrip())
    if memory_overview.strip():
        sections.append(memory_overview.rstrip())
    if skill_overview.strip():
        sections.append(skill_overview.rstrip())

    # Append-only history sits in the cache-stable region: all lines but the
    # newest are byte-identical across calls, so placing it above the volatile
    # per-step blocks lets the prompt prefix (system + base + overviews +
    # history) be served from the model's KV cache.
    sections.append(
        "## RECENT HISTORY\n" + (history_block or "No previous actions recorded.")
    )

    # Volatile per-step tail — re-rendered every call, never cached.
    sections.append(f"# Step: {action_counter}")
    if observation_block.strip():
        sections.append("## OBSERVATIONS SINCE LAST QUERY\n" + observation_block.rstrip())
    sections.append(
        "## TOOL RESULTS FROM PREVIOUS TURN\n"
        + format_tool_results_markdown(recent_tool_results)
    )
    if previous_no_action_reason:
        sections.append(
            "## PREVIOUS CONVERSATION STOPPED WITHOUT ACTION\n"
            f"Reason: {previous_no_action_reason.strip()}"
        )
    state_block = (
        "## CURRENT STATE\n"
        f"state: {latest_frame.state.name}\n"
        f"score (levels completed): {latest_frame.levels_completed}\n"
        f"available actions: {_format_available_actions(available)}\n"
        f"current grid (latest_frame.frame[-1]):\n{frame_text}"
    )
    sections.append(state_block)

    budget_line = (
        f"You have at most {max_deliberation_turns} non-action turns to decide the next action(s)."
        if max_deliberation_turns
        else ""
    )
    sections.append(
        "## TURN\n"
        "This is a running conversation about the frame above. The frame stays fixed until you act. "
        + budget_line
        + "Aim to act within 2-3 non-action turns. Do not re-run a tool whose "
        "output is already in this conversation or in the recap above; record "
        "conclusions to memory instead of re-deriving them. Only take_actions "
        "produces new information — predict each action's effect, then commit. "
        "An action that advances the game or disproves a hypothesis are both progress; you will receive a fresh observation next."
    )
    return "\n\n".join(sections)


def build_subagent_prompt(
    *,
    task: str,
    context: dict[str, Any] | None,
    latest_frame: FrameData,
    memory_overview: str,
    skill_overview: str,
    compact_history: str,
) -> str:
    """Assemble the user-prompt half of a subagent invocation.

    The subagent's `instructions` are passed as system_instruction by the
    orchestrator; this function only builds the per-call user prompt. The
    termination cue at the bottom mirrors how the orchestrator's USER_PROMPT
    ends with `# TURN:` — keeping the action cue last (here, subagent_return).
    """
    parts: list[str] = []
    parts.append("## TASK")
    parts.append((task or "").strip() or "(no task supplied)")
    parts.append("")

    parts.append("## CONTEXT")
    if context:
        parts.append(json.dumps(context, indent=2, default=str))
    else:
        parts.append("(none)")
    parts.append("")

    parts.append(memory_overview)
    parts.append("")
    parts.append(skill_overview)
    parts.append("")

    parts.append("## RECENT STEPS")
    parts.append(compact_history)
    parts.append("")

    parts.append("## CURRENT FRAME")
    parts.append(
        f"state={latest_frame.state.name} score={latest_frame.levels_completed}"
    )
    parts.append(pretty_print_3d(latest_frame.frame))
    parts.append("")

    parts.append(
        "When you have completed your task, call "
        "subagent_return(reasoning=..., answer=..., status=...)."
    )
    return "\n".join(parts)
