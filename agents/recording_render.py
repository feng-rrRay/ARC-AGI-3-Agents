from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from PIL import Image, ImageDraw, ImageFont

RECORDING_SUFFIX = ".recording.jsonl"
TRACE_SUFFIX = ".trace.jsonl"
TRAJECTORY_SUFFIX = ".trajectory.jsonl"
VIDEO_SUFFIXES = {".gif", ".mp4"}
RenderFormat = Literal["gif", "mp4"]
CallKind = Literal["action_batch", "action_rejected", "analysis", "evolution"]
EventKind = Literal["orchestrator", "subagent", "evolution"]

# ARC-AGI 16-colour palette indexed by grid values 0..15.
ARC_PALETTE: tuple[tuple[int, int, int, int], ...] = (
    (0xFF, 0xFF, 0xFF, 0xFF),
    (0xCC, 0xCC, 0xCC, 0xFF),
    (0x99, 0x99, 0x99, 0xFF),
    (0x66, 0x66, 0x66, 0xFF),
    (0x33, 0x33, 0x33, 0xFF),
    (0x00, 0x00, 0x00, 0xFF),
    (0xE5, 0x3A, 0xA3, 0xFF),
    (0xFF, 0x7B, 0xCC, 0xFF),
    (0xF9, 0x3C, 0x31, 0xFF),
    (0x1E, 0x93, 0xFF, 0xFF),
    (0x88, 0xD8, 0xF1, 0xFF),
    (0xFF, 0xDC, 0x00, 0xFF),
    (0xFF, 0x85, 0x1B, 0xFF),
    (0x92, 0x12, 0x31, 0xFF),
    (0x4F, 0xCC, 0x30, 0xFF),
    (0xA3, 0x56, 0xD6, 0xFF),
)

_ACTION_NAMES: dict[int, str] = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}
# Action labels can be parsed from two log formats:
#  - Legacy:    "| INFO | <prefix> - ACTION1: count 7, ..."
#  - Continual: "[<game-id>] step=7 ACTION1[(args)]? src=vlm ..."
# Try both so renders of older runs keep working.
_ACTION_LOG_PATTERNS = (
    re.compile(
        r"\|\s+INFO\s+\|\s+.+?\s+-\s+(?P<action>RESET|ACTION[1-7]): "
        r"count (?P<count>\d+),"
    ),
    re.compile(
        r"\[[^\]]+\]\s+step=(?P<count>\d+)\s+"
        r"(?P<action>RESET|ACTION[1-7](?:\([^)]*\))?)(?:\s|$)"
    ),
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


JsonObject = dict[str, Any]
Grid = Sequence[Sequence[int]]


# --------------------------------------------------------------------------- #
# Visual theme (dark "blog replay" look, modelled on the ARC Prize viewer).
# --------------------------------------------------------------------------- #
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class Theme:
    bg: RGBA = (18, 18, 20, 255)
    panel_bg: RGBA = (24, 24, 27, 255)
    grid_bg: RGBA = (30, 30, 34, 255)
    card_bg: RGBA = (34, 34, 39, 255)
    card_current_bg: RGBA = (45, 45, 53, 255)
    card_border: RGBA = (62, 62, 70, 255)
    divider: RGBA = (48, 48, 54, 255)
    accent: RGBA = (255, 133, 27, 255)
    accent_soft: RGBA = (120, 78, 28, 255)
    text_primary: RGBA = (236, 237, 241, 255)
    text_secondary: RGBA = (156, 160, 170, 255)
    text_muted: RGBA = (112, 116, 126, 255)
    chip_fg: RGBA = (24, 18, 10, 255)
    tag_bg: RGBA = (52, 54, 62, 255)
    tag_fg: RGBA = (180, 200, 240, 255)
    grid_line: RGBA = (0, 0, 0, 48)
    playhead: RGBA = (240, 240, 245, 255)
    timeline_bg: RGBA = (42, 42, 48, 255)


THEME = Theme()

# Level segments cycle through warm tones so adjacent levels stay distinct.
_LEVEL_COLORS: tuple[RGBA, ...] = (
    (255, 133, 27, 255),
    (255, 196, 60, 255),
    (255, 220, 0, 255),
    (245, 110, 40, 255),
    (255, 165, 40, 255),
)

_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "regular": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ),
    "bold": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    "mono": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ),
}


@dataclass(frozen=True)
class FontSet:
    frame_big: ImageFont.FreeTypeFont
    title: ImageFont.FreeTypeFont
    header: ImageFont.FreeTypeFont
    chip: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    tiny: ImageFont.FreeTypeFont
    level: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont


def _load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in _FONT_CANDIDATES.get(kind, ()):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # Pillow >= 10 returns a TrueType-backed default at the requested size.
    return cast(ImageFont.FreeTypeFont, ImageFont.load_default(size=size))


@lru_cache(maxsize=1)
def _load_fonts() -> FontSet:
    return FontSet(
        frame_big=_load_font("bold", 19),
        title=_load_font("bold", 15),
        header=_load_font("bold", 13),
        chip=_load_font("bold", 12),
        body=_load_font("regular", 13),
        small=_load_font("regular", 11),
        tiny=_load_font("regular", 10),
        level=_load_font("bold", 10),
        mono=_load_font("mono", 11),
    )


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecordingFrame:
    index: int
    timestamp: str | None
    data: JsonObject
    action_label: str | None = None


@dataclass(frozen=True)
class RecordingGridFrame:
    event: RecordingFrame
    grid_index: int
    grid_count: int
    grid: Grid


@dataclass(frozen=True)
class CallEntry:
    name: str
    kind: CallKind
    args: JsonObject
    executed: bool
    result: JsonObject | None = None
    error: str | None = None
    actions_committed: int | None = None  # set on take_actions calls
    actions_taken_inline: int | None = None  # set on run_skill calls


@dataclass(frozen=True)
class PromptEvolutionEntry:
    generation: int
    accepted: bool
    reasoning: str | None
    validation_error: str | None


@dataclass(frozen=True)
class TraceEvent:
    """One row from the trace JSONL, normalized for the renderer."""

    file_index: int
    vlm_call: int | None
    round: int | None
    action_counter: int
    tools_exposed: str | None
    kind: EventKind
    actions_executed: int | None
    force_take_actions: bool
    reasoning: str | None
    error: str | None
    calls: tuple[CallEntry, ...]
    evolution: PromptEvolutionEntry | None
    subagent_info: JsonObject | None
    usage: JsonObject | None = None
    usage_cost: JsonObject | None = None
    timestamp: str | None = None

    @property
    def total_actions(self) -> int:
        """Direct take_actions + run_skill inline actions."""
        direct = self.actions_executed or 0
        inline = sum(
            c.actions_taken_inline
            for c in self.calls
            if c.actions_taken_inline is not None and c.actions_taken_inline > 0
        )
        return direct + inline


@dataclass(frozen=True)
class ActionPanel:
    """The frames a contiguous batch of action frames shares."""

    frame_start: int  # inclusive recording-frame index
    frame_end: int  # inclusive recording-frame index
    events: tuple[TraceEvent, ...]


@dataclass(frozen=True)
class Decision:
    """One reasoning step that produced one or more game actions.

    The unit a Reasoning-Log card represents. Built from the CH trace
    (`build_decisions_ch`) or straight from a Hermes recording's embedded
    per-action reasoning (`build_decisions_hermes`).
    """

    index: int  # 0-based ordinal across the run
    frame_start: int  # inclusive recording-frame index this decision covers
    frame_end: int  # inclusive recording-frame index
    level: int  # levels_completed at frame_end
    action_chips: tuple[
        str, ...
    ]  # compact action labels, e.g. ("A2 ×3", "CLICK (4, 9)")
    reasoning: str | None
    tool_tags: tuple[str, ...]  # tool names that produced the action (CH only)
    tokens: int | None = None
    cost_usd: float | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class LayoutSpec:
    scale: int
    canvas_w: int
    canvas_h: int
    grid_x: int
    grid_y: int
    grid_px_w: int
    grid_px_h: int
    left_w: int
    right_x: int
    right_w: int
    header_h: int
    timeline_y: int
    timeline_h: int
    total_frames: int
    level_bounds: tuple[tuple[int, int, int], ...]  # (level, start_idx, end_idx)
    reasoning: bool


@dataclass(frozen=True)
class RenderSummary:
    recording: Path
    output: Path
    frame_events: int
    grid_frames: int
    trace_log: Path | None
    decision_count: int
    style: str


# --------------------------------------------------------------------------- #
# Discovery & loading
# --------------------------------------------------------------------------- #
def discover_recording_paths(path: str | Path) -> list[Path]:
    """Return recording files from a single file or a run/recordings directory."""
    input_path = Path(path)
    if input_path.is_file():
        if not _is_recording_jsonl(input_path):
            raise ValueError(
                "Input file must be a recording JSONL file "
                f"({RECORDING_SUFFIX} or Hermes *.jsonl): {input_path}"
            )
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"No such recording file or run directory: {input_path}"
        )

    recordings_dir = input_path / "recordings"
    search_dirs = [recordings_dir] if recordings_dir.is_dir() else [input_path]
    if not recordings_dir.is_dir():
        search_dirs.extend(sorted(input_path.glob("*/recordings")))

    recordings = sorted(
        {
            recording
            for search_dir in search_dirs
            for recording in _recordings_in_dir(search_dir)
        }
    )
    if not recordings:
        detail = (
            f"{recordings_dir} or {input_path}"
            if recordings_dir.is_dir()
            else input_path
        )
        raise ValueError(
            f"No recording JSONL files found in {detail} "
            f"({RECORDING_SUFFIX} or Hermes *.jsonl)"
        )
    return recordings


def load_recording_frames(path: str | Path) -> list[RecordingFrame]:
    """Load frame-bearing events from an ARC recording JSONL file."""
    recording_path = Path(path)
    frames: list[RecordingFrame] = []

    with recording_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            event = _load_json_object(stripped, recording_path, line_number)
            data = event.get("data")
            if not isinstance(data, dict) or "frame" not in data:
                continue

            timestamp = event.get("timestamp")
            frames.append(
                RecordingFrame(
                    index=len(frames),
                    timestamp=str(timestamp) if timestamp is not None else None,
                    data=data,
                )
            )

    if not frames:
        raise ValueError(f"No frame events found in {recording_path}")

    return frames


def parse_action_log(path: str | Path) -> dict[int, str]:
    """Parse action labels from a run log keyed by Agent action count."""
    log_path = Path(path)
    labels: dict[int, str] = {}

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            for pattern in _ACTION_LOG_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue
                labels[int(match.group("count"))] = match.group("action")
                break

    return labels


def find_actions_log(
    recording_path: str | Path,
    *,
    logs_dir: str | Path = "logs",
) -> Path | None:
    """Find a run log that mentions the given recording file."""
    recording = Path(recording_path)
    run_log = _run_log_for_recording(recording)
    if run_log is not None:
        return run_log

    logs_root = Path(logs_dir)
    if not logs_root.exists():
        return None

    candidates = {recording.name, str(recording)}
    if recording.is_absolute():
        try:
            candidates.add(str(recording.relative_to(Path.cwd())))
        except ValueError:
            pass

    for log_path in sorted(logs_root.glob("*.log"), key=_path_mtime, reverse=True):
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(candidate in text for candidate in candidates):
            return log_path

    return None


def find_trace_log(
    recording_path: str | Path,
    *,
    actions_log: str | Path | None = None,
    logs_dir: str | Path = "logs",
) -> Path | None:
    """Find a VLM trace JSONL file matching the given recording."""
    recording = Path(recording_path)
    artifact_trace = _artifact_companion_for_recording(recording, TRACE_SUFFIX)
    if artifact_trace is not None:
        return artifact_trace

    run_dir = _run_dir_for_recording(recording)
    if run_dir is not None:
        sibling = run_dir / "trace.jsonl"
        if sibling.exists():
            return sibling

    if actions_log is not None:
        sibling_trace = Path(actions_log).with_suffix(TRACE_SUFFIX)
        if sibling_trace.exists():
            return sibling_trace

    logs_root = Path(logs_dir)
    if not logs_root.exists():
        return None

    agent_hint = _recording_agent_hint(recording)
    if agent_hint is None:
        return None

    for trace_path in sorted(
        logs_root.glob(f"*{TRACE_SUFFIX}"), key=_path_mtime, reverse=True
    ):
        if _trace_mentions_agent(trace_path, agent_hint):
            return trace_path

    return None


def find_trajectory_log(recording_path: str | Path) -> Path | None:
    """Find a trajectory JSONL file alongside the recording's artifacts."""
    return _artifact_companion_for_recording(Path(recording_path), TRAJECTORY_SUFFIX)


def parse_trace_events(path: str | Path) -> list[TraceEvent]:
    """Parse all VLM trace events into a single ordered list (file order kept)."""
    trace_path = Path(path)
    events: list[TraceEvent] = []

    with trace_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = _load_json_object(stripped, trace_path, line_number)
            event = _build_trace_event(raw, file_index=len(events))
            if event is not None:
                events.append(event)
    return events


def group_into_panels(events: Sequence[TraceEvent]) -> list[ActionPanel]:
    """Bucket trace events into one panel per batch of action frames."""
    panels: list[ActionPanel] = []
    buffer: list[TraceEvent] = []
    for event in events:
        buffer.append(event)
        total = event.total_actions
        if event.kind == "orchestrator" and total > 0:
            frame_end = event.action_counter - 1
            frame_start = frame_end - total + 1
            panels.append(
                ActionPanel(
                    frame_start=frame_start,
                    frame_end=frame_end,
                    events=tuple(buffer),
                )
            )
            buffer = []
    return panels


def apply_action_labels(
    frames: Sequence[RecordingFrame],
    action_labels: dict[int, str],
) -> list[RecordingFrame]:
    """Attach parsed action labels to recording frames by action count."""
    if not action_labels:
        return list(frames)
    return [
        replace(frame, action_label=action_labels.get(frame.index)) for frame in frames
    ]


# --------------------------------------------------------------------------- #
# Decision model
# --------------------------------------------------------------------------- #
def build_decisions(
    frames: Sequence[RecordingFrame],
    recording_path: str | Path | None,
    trace_events: Sequence[TraceEvent] | None,
) -> tuple[list[Decision], str]:
    """Build decision cards and report the detected style ('continual'/'hermes')."""
    if trace_events:
        return build_decisions_ch(frames, trace_events), "continual"
    trajectory = (
        find_trajectory_log(recording_path) if recording_path is not None else None
    )
    tool_by_frame = _hermes_tool_by_frame(frames, trajectory) if trajectory else None
    return build_decisions_hermes(frames, tool_by_frame), "hermes"


def build_decisions_ch(
    frames: Sequence[RecordingFrame],
    trace_events: Sequence[TraceEvent],
) -> list[Decision]:
    """One Decision per action-emitting VLM panel from the CH trace."""
    panels = group_into_panels(trace_events)
    by_index = {frame.index: frame for frame in frames}
    last_frame = max(by_index, default=-1)

    decisions: list[Decision] = []
    prev_ts: datetime | None = None
    for index, panel in enumerate(panels):
        start = max(0, panel.frame_start)
        end = min(panel.frame_end, last_frame)
        closing = panel.events[-1]

        panel_frames = [by_index[i] for i in range(start, end + 1) if i in by_index]
        chips = _chips_for_frames(panel_frames)
        reasoning = _reasoning_for_panel(panel)
        tool_tags = _tool_tags_for_panel(panel)
        level = _frame_level(by_index.get(end))

        tokens = _usage_output_tokens(closing)
        cost = _usage_cost_usd(closing)
        ts = _parse_ts(closing.timestamp)
        duration = (ts - prev_ts).total_seconds() if ts and prev_ts else None
        if ts is not None:
            prev_ts = ts

        decisions.append(
            Decision(
                index=index,
                frame_start=start,
                frame_end=end,
                level=level,
                action_chips=chips,
                reasoning=reasoning,
                tool_tags=tool_tags,
                tokens=tokens,
                cost_usd=cost,
                duration_s=duration if duration and duration > 0 else None,
            )
        )
    return decisions


def build_decisions_hermes(
    frames: Sequence[RecordingFrame],
    tool_by_frame: dict[int, str] | None = None,
) -> list[Decision]:
    """Group consecutive frames sharing (action, reasoning, tool) into one Decision.

    Hermes recordings embed per-action reasoning in `action_input.reasoning`, so
    no trace file is needed. `tool_by_frame` (recovered from `trajectory.jsonl`)
    labels each frame as ``take_actions`` or ``execute_code`` — the game server
    can't record this since both arrive via the same MCP endpoint.
    """
    tools = tool_by_frame or {}
    decisions: list[Decision] = []
    run_start = 0
    prev_key: tuple[str, str | None, str | None] | None = None

    def flush(start: int, end: int) -> None:
        frame = frames[start]
        chip = _action_chip(frame.data.get("action_input"))
        count = end - start + 1
        label = f"{chip} ×{count}" if count > 1 else chip
        tool = tools.get(start)
        decisions.append(
            Decision(
                index=len(decisions),
                frame_start=start,
                frame_end=end,
                level=_frame_level(frames[end]),
                action_chips=(label,),
                reasoning=_embedded_reasoning(frame.data.get("action_input")),
                tool_tags=(tool,) if tool else (),
            )
        )

    for i, frame in enumerate(frames):
        ai = frame.data.get("action_input")
        key = (_action_chip(ai), _embedded_reasoning(ai), tools.get(i))
        if prev_key is not None and key != prev_key:
            flush(run_start, i - 1)
            run_start = i
        prev_key = key
    if frames:
        flush(run_start, len(frames) - 1)
    return decisions


def _hermes_take_action_specs(
    trajectory_path: str | Path,
) -> list[tuple[str, str | None]]:
    """Ordered (action_name, reasoning) for every top-level take_actions action.

    These are the actions Hermes issued by calling the MCP `take_actions` tool
    directly (as opposed to from inside `execute_code`).
    """
    path = Path(trajectory_path)
    specs: list[tuple[str, str | None]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(event, dict)
                or event.get("type") != "tool_use"
                or event.get("name") != "take_actions"
            ):
                continue
            args = event.get("arguments")
            actions = args.get("actions") if isinstance(args, dict) else None
            if not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, dict):
                    continue
                specs.append(
                    (
                        _normalize_action_name(action.get("name")),
                        _clean_string(action.get("reasoning")),
                    )
                )
    return specs


def _hermes_tool_by_frame(
    frames: Sequence[RecordingFrame],
    trajectory_path: str | Path | None,
) -> dict[int, str]:
    """Label each Hermes action frame as ``take_actions`` or ``execute_code``.

    Frames whose (action, reasoning) matches a top-level take_actions call are
    tagged ``take_actions`` (consumed in recording order); every other action
    frame came from inside `execute_code`. RESET frames get no tool tag.
    """
    if trajectory_path is None:
        return {}
    remaining: Counter[tuple[str, str | None]] = Counter(
        _hermes_take_action_specs(trajectory_path)
    )
    tags: dict[int, str] = {}
    for frame in frames:
        action_input = frame.data.get("action_input")
        if not isinstance(action_input, dict):
            continue
        name = _normalize_action_name(action_input.get("id"))
        if name == "RESET":
            continue
        key = (name, _embedded_reasoning(action_input))
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            tags[frame.index] = "take_actions"
        else:
            tags[frame.index] = "execute_code"
    return tags


def frame_decision_index(
    decisions: Sequence[Decision], total_frames: int
) -> dict[int, int]:
    """Map each recording-frame index to the Decision that covers it."""
    mapping: dict[int, int] = {}
    for decision in decisions:
        for frame_index in range(decision.frame_start, decision.frame_end + 1):
            if 0 <= frame_index < total_frames:
                mapping[frame_index] = decision.index
    return mapping


def _chips_for_frames(frames: Sequence[RecordingFrame]) -> tuple[str, ...]:
    """Run-length compress the action chips across a panel's frames."""
    chips: list[str] = []
    run_label: str | None = None
    run_count = 0

    def emit() -> None:
        if run_label is None:
            return
        chips.append(f"{run_label} ×{run_count}" if run_count > 1 else run_label)

    for frame in frames:
        label = _action_chip(frame.data.get("action_input"))
        if label == run_label:
            run_count += 1
        else:
            emit()
            run_label = label
            run_count = 1
    emit()

    if not chips:
        return ("(no action)",)
    if len(chips) > 4:
        extra = len(chips) - 3
        return (*chips[:3], f"+{extra} more")
    return tuple(chips)


def _action_chip(action_input: Any) -> str:
    """Render an `action_input` as a compact chip label."""
    if not isinstance(action_input, dict):
        return "—"
    name = _normalize_action_name(action_input.get("id"))
    data = action_input.get("data")
    if name == "ACTION6" and isinstance(data, dict):
        x, y = data.get("x"), data.get("y")
        if x is not None and y is not None:
            return f"CLICK ({x}, {y})"
    if name == "RESET":
        return "RESET"
    if name.startswith("ACTION") and name[6:].isdigit():
        return f"A{name[6:]}"
    return name


def _normalize_action_name(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return _ACTION_NAMES.get(value, f"ACTION{value}")
    if isinstance(value, str):
        if value.isdecimal():
            return _ACTION_NAMES.get(int(value), value)
        return value
    return "—"


def _frame_level(frame: RecordingFrame | None) -> int:
    if frame is None:
        return 0
    value = _int_or_none(frame.data.get("levels_completed"))
    return value if value is not None else 0


def _embedded_reasoning(action_input: Any) -> str | None:
    """Pull Hermes' per-action reasoning out of `action_input.reasoning`."""
    if not isinstance(action_input, dict):
        return None
    reasoning = action_input.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("reasoning")
    return _clean_string(reasoning)


def _reasoning_for_panel(panel: ActionPanel) -> str | None:
    """Pick the reasoning text that best explains a panel's action."""
    closing = panel.events[-1]
    for call in closing.calls:
        if call.kind == "action_batch":
            text = _clean_string(call.args.get("reasoning"))
            if text:
                return text
        if call.name == "run_skill":
            text = _clean_string(call.args.get("reasoning"))
            if text:
                return text
    return _clean_string(closing.reasoning)


def _tool_tags_for_panel(panel: ActionPanel) -> tuple[str, ...]:
    """Names of the analysis/meta tools that contributed to the action."""
    tags: list[str] = []
    for event in panel.events:
        if event.kind == "subagent":
            info = event.subagent_info or {}
            name = _clean_string(info.get("name"))
            tag = f"subagent:{name}" if name else "subagent"
            if tag not in tags:
                tags.append(tag)
        for call in event.calls:
            if call.kind in {"action_batch", "action_rejected"}:
                continue
            label = _tool_tag_label(call)
            if label and label not in tags:
                tags.append(label)
    return tuple(tags[:6])


def _tool_tag_label(call: CallEntry) -> str | None:
    if call.name == "run_skill":
        skill = _clean_string((call.result or {}).get("name"))
        return f"skill:{skill}" if skill else "run_skill"
    if call.name == "process_skill":
        op = _clean_string(call.args.get("operation")) or "edit"
        return f"skill·{op}"
    if call.name == "process_memory":
        op = _clean_string(call.args.get("operation")) or "edit"
        return f"memory·{op}"
    if call.name == "process_subagent":
        return "subagent·edit"
    if call.name == "run_subagent":
        return "run_subagent"
    return _clean_string(call.name)


def _usage_output_tokens(event: TraceEvent) -> int | None:
    cost = event.usage_cost or {}
    billable = _int_or_none(cost.get("billable_output_tokens"))
    if billable is not None:
        return billable
    usage = event.usage or {}
    return _int_or_none(usage.get("output"))


def _usage_cost_usd(event: TraceEvent) -> float | None:
    cost = event.usage_cost or {}
    value = cost.get("current_usd")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Grid rendering
# --------------------------------------------------------------------------- #
def grid_to_image(grid: Grid, scale: int = 1) -> Image.Image:
    """Render a 2-D ARC grid to a crisp RGBA image."""
    if scale < 1:
        raise ValueError("scale must be >= 1")

    rows = [list(row) for row in grid]
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    if height == 0 or width == 0:
        raise ValueError("Grid must contain at least one cell")

    raw = bytearray()
    for row in rows:
        for x in range(width):
            value = row[x] if x < len(row) else 0
            raw.extend(_palette_rgba(value))

    image = Image.frombytes("RGBA", (width, height), bytes(raw))
    if scale == 1:
        return image
    return image.resize((width * scale, height * scale), Image.Resampling.NEAREST)


def expand_recording_frames(
    frames: Sequence[RecordingFrame],
) -> list[RecordingGridFrame]:
    """Expand each event's sequential grids into renderable GIF frames."""
    grid_frames: list[RecordingGridFrame] = []
    for frame in frames:
        grids = _frame_layers(frame.data)
        grid_frames.extend(
            RecordingGridFrame(
                event=frame,
                grid_index=grid_index,
                grid_count=len(grids),
                grid=grid,
            )
            for grid_index, grid in enumerate(grids)
        )
    return grid_frames


# --------------------------------------------------------------------------- #
# Web-styled frame renderer
# --------------------------------------------------------------------------- #
def render_recording_frame(
    frame: RecordingFrame,
    *,
    scale: int = 8,
    overlay: bool = True,
) -> Image.Image:
    """Render the first grid from one recording event."""
    grid_frame = expand_recording_frames([frame])[0]
    return render_recording_grid_frame(grid_frame, scale=scale, overlay=overlay)


def render_recording_grid_frame(
    grid_frame: RecordingGridFrame,
    *,
    scale: int = 8,
    overlay: bool = True,
    layout: LayoutSpec | None = None,
    decisions: Sequence[Decision] | None = None,
    decision_index: int | None = None,
) -> Image.Image:
    """Render one grid, optionally inside the full blog-replay layout."""
    grid_image = grid_to_image(grid_frame.grid, scale=scale)
    if not overlay:
        return grid_image

    if layout is None:
        layout = _compute_layout(
            [grid_frame], [grid_frame.event], scale, reasoning=True
        )
    if decisions is None:
        decisions = build_decisions_hermes([grid_frame.event])
        decision_index = 0 if decisions else None
    return _render_web_frame(grid_frame, grid_image, layout, decisions, decision_index)


def render_recording_images(
    frames: Sequence[RecordingFrame],
    *,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
    decisions: Sequence[Decision] | None = None,
    total_frames: int | None = None,
) -> list[Image.Image]:
    """Render recording events to one image per grid frame."""
    if not frames:
        raise ValueError("At least one frame is required")

    grid_frames = expand_recording_frames(frames)

    if not overlay:
        images = [grid_to_image(gf.grid, scale=scale) for gf in grid_frames]
        return _pad_to_common_size(images)

    total = total_frames if total_frames is not None else len(frames)
    if decisions is None:
        decisions = build_decisions_hermes(frames)
    dec_for_frame = frame_decision_index(decisions, total)
    layout = _compute_layout(
        grid_frames, frames, scale, reasoning=reasoning, total=total
    )

    images = [
        _render_web_frame(
            gf,
            grid_to_image(gf.grid, scale=scale),
            layout,
            decisions,
            dec_for_frame.get(gf.event.index),
        )
        for gf in grid_frames
    ]
    return _pad_to_common_size(images)


def _compute_layout(
    grid_frames: Sequence[RecordingGridFrame],
    frames: Sequence[RecordingFrame],
    scale: int,
    *,
    reasoning: bool,
    total: int | None = None,
) -> LayoutSpec:
    max_w = max((len(row) for gf in grid_frames for row in gf.grid), default=1)
    max_h = max((len(gf.grid) for gf in grid_frames), default=1)
    grid_px_w = max_w * scale
    grid_px_h = max_h * scale

    pad = 24
    header_h = 48
    gap = 14
    timeline_h = 52
    grid_x = pad
    grid_y = header_h
    left_w = grid_x + grid_px_w + pad
    timeline_y = grid_y + grid_px_h + gap
    left_h = timeline_y + timeline_h + pad

    right_w = 480 if reasoning else 0
    canvas_w = left_w + right_w
    canvas_h = max(left_h, 360 if reasoning else left_h)

    total_frames = total if total is not None else len(frames)
    return LayoutSpec(
        scale=scale,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_px_w=grid_px_w,
        grid_px_h=grid_px_h,
        left_w=left_w,
        right_x=left_w,
        right_w=right_w,
        header_h=header_h,
        timeline_y=timeline_y,
        timeline_h=timeline_h,
        total_frames=max(1, total_frames),
        level_bounds=_level_bounds(frames),
        reasoning=reasoning,
    )


def _level_bounds(
    frames: Sequence[RecordingFrame],
) -> tuple[tuple[int, int, int], ...]:
    bounds: list[tuple[int, int, int]] = []
    for frame in frames:
        level = _frame_level(frame)
        if bounds and bounds[-1][0] == level:
            prev_level, start, _ = bounds[-1]
            bounds[-1] = (prev_level, start, frame.index)
        else:
            bounds.append((level, frame.index, frame.index))
    return tuple(bounds)


def _render_web_frame(
    grid_frame: RecordingGridFrame,
    grid_image: Image.Image,
    layout: LayoutSpec,
    decisions: Sequence[Decision],
    decision_index: int | None,
) -> Image.Image:
    fonts = _load_fonts()
    canvas = Image.new("RGBA", (layout.canvas_w, layout.canvas_h), THEME.bg)
    draw = ImageDraw.Draw(canvas)

    _draw_left_header(draw, layout, grid_frame, fonts)
    _paste_grid(canvas, draw, grid_image, layout)
    _draw_timeline(draw, layout, grid_frame, fonts)

    if layout.reasoning:
        _draw_right_column(draw, layout, decisions, decision_index, fonts)
    return canvas


def _draw_left_header(
    draw: ImageDraw.ImageDraw,
    layout: LayoutSpec,
    grid_frame: RecordingGridFrame,
    fonts: FontSet,
) -> None:
    cx = layout.grid_x + layout.grid_px_w // 2
    draw.text(
        (layout.grid_x, 16),
        f"Step {grid_frame.event.index}",
        font=fonts.title,
        fill=THEME.text_secondary,
    )
    level = _frame_level(grid_frame.event) + 1
    label = f"Level: {level}"
    _draw_centered(draw, label, cx, 15, fonts.title, THEME.text_primary)

    state = _clean_string(grid_frame.event.data.get("state"))
    if state and state != "NOT_FINISHED":
        right = layout.grid_x + layout.grid_px_w
        color = THEME.accent if state == "WIN" else (240, 120, 120, 255)
        _draw_right_aligned(draw, state, right, 16, fonts.small, color)


def _paste_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    grid_image: Image.Image,
    layout: LayoutSpec,
) -> None:
    # Center a smaller grid inside the reserved area; backdrop + cell gridlines.
    area = (
        layout.grid_x,
        layout.grid_y,
        layout.grid_x + layout.grid_px_w,
        layout.grid_y + layout.grid_px_h,
    )
    _rounded_rect(draw, area, 6, fill=THEME.grid_bg)
    off_x = layout.grid_x + (layout.grid_px_w - grid_image.width) // 2
    off_y = layout.grid_y + (layout.grid_px_h - grid_image.height) // 2
    canvas.alpha_composite(_with_grid_lines(grid_image, layout.scale), (off_x, off_y))


def _with_grid_lines(image: Image.Image, scale: int) -> Image.Image:
    if scale < 4:
        return image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, image.width + 1, scale):
        draw.line((x, 0, x, image.height), fill=THEME.grid_line, width=1)
    for y in range(0, image.height + 1, scale):
        draw.line((0, y, image.width, y), fill=THEME.grid_line, width=1)
    return Image.alpha_composite(image, overlay)


def _draw_timeline(
    draw: ImageDraw.ImageDraw,
    layout: LayoutSpec,
    grid_frame: RecordingGridFrame,
    fonts: FontSet,
) -> None:
    x0 = layout.grid_x
    x1 = layout.grid_x + layout.grid_px_w
    bar_top = layout.timeline_y
    bar_h = 10
    width = max(1, x1 - x0)
    total = layout.total_frames
    head_x = x0 + round((grid_frame.event.index + 1) / total * width)

    _rounded_rect(draw, (x0, bar_top, x1, bar_top + bar_h), 5, fill=THEME.timeline_bg)

    last_label_x = -100
    for level, start, end in layout.level_bounds:
        seg_x0 = x0 + round(start / total * width)
        seg_x1 = max(x0 + round((end + 1) / total * width), seg_x0 + 1)
        color = _LEVEL_COLORS[level % len(_LEVEL_COLORS)]
        dim = _blend(color, THEME.timeline_bg, 0.6)
        # Played part keeps the bright colour; the rest is dimmed in place.
        if head_x > seg_x0:
            draw.rectangle(
                (seg_x0, bar_top, min(seg_x1, head_x), bar_top + bar_h), fill=color
            )
        if head_x < seg_x1:
            draw.rectangle(
                (max(seg_x0, head_x), bar_top, seg_x1, bar_top + bar_h), fill=dim
            )
        if start > 0:
            draw.line(
                (seg_x0, bar_top - 2, seg_x0, bar_top + bar_h + 2),
                fill=THEME.bg,
                width=2,
            )
            if seg_x0 - last_label_x >= 14:
                draw.text(
                    (seg_x0 + 2, bar_top + bar_h + 3),
                    str(level + 1),
                    font=fonts.level,
                    fill=THEME.text_muted,
                )
                last_label_x = seg_x0

    # Playhead marker.
    draw.line(
        (head_x, bar_top - 4, head_x, bar_top + bar_h + 4),
        fill=THEME.playhead,
        width=2,
    )


def _draw_right_column(
    draw: ImageDraw.ImageDraw,
    layout: LayoutSpec,
    decisions: Sequence[Decision],
    decision_index: int | None,
    fonts: FontSet,
) -> None:
    rx = layout.right_x
    draw.rectangle((rx, 0, layout.canvas_w, layout.canvas_h), fill=THEME.panel_bg)
    draw.line((rx, 0, rx, layout.canvas_h), fill=THEME.divider, width=1)

    pad = 18
    inner_x = rx + pad
    inner_w = layout.right_w - 2 * pad

    draw.text(
        (inner_x, 18), "REASONING LOG", font=fonts.header, fill=THEME.text_primary
    )
    _draw_pill(
        draw,
        inner_x + inner_w - 84,
        16,
        "Batches",
        fonts.tiny,
        THEME.text_secondary,
        THEME.tag_bg,
        width=84,
    )

    y = 48
    bottom = layout.canvas_h - 14
    if decision_index is None or not decisions:
        draw.text(
            (inner_x, y),
            "No decision for this frame.",
            font=fonts.body,
            fill=THEME.text_muted,
        )
        return

    y = _draw_expanded_card(
        draw,
        inner_x,
        y,
        inner_w,
        decisions[decision_index],
        fonts,
        bottom,
    )
    for prev in range(decision_index - 1, -1, -1):
        if y >= bottom - 36:
            break
        y = _draw_collapsed_card(draw, inner_x, y, inner_w, decisions[prev], fonts)


def _draw_expanded_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    decision: Decision,
    fonts: FontSet,
    bottom: int,
) -> int:
    pad = 12
    body_lh = _font_line_height(fonts.body)
    chip_row_h = _font_line_height(fonts.chip) + 8
    tag_row_h = _font_line_height(fonts.tiny) + 8
    text_w = width - 2 * pad
    right = x + width - pad

    label = f"Batch {decision.index + 1}"
    label_h = _font_line_height(fonts.frame_big)
    chip_first_x = x + pad + _text_width(label, fonts.frame_big) + 10
    chip_rows = _pill_rows(
        decision.action_chips,
        first_x=chip_first_x,
        wrap_x=x + pad,
        max_right=right,
        font=fonts.chip,
    )
    header_h = max(label_h, chip_rows * chip_row_h)

    tag_rows = _pill_rows(
        decision.tool_tags,
        first_x=x + pad,
        wrap_x=x + pad,
        max_right=right,
        font=fonts.tiny,
    )
    has_meta = decision.tokens is not None or decision.cost_usd is not None

    reasoning = decision.reasoning or "(no reasoning recorded)"
    lines = _wrap_text_to_width(reasoning, fonts.body, text_w)
    reserved = pad + header_h + 8 + tag_rows * tag_row_h + (18 if has_meta else 0) + pad
    max_lines = min(14, max(2, (bottom - y - reserved) // body_lh))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit_text_to_width(lines[-1], fonts.body, text_w, suffix=" …")

    card_h = (
        pad
        + header_h
        + len(lines) * body_lh
        + 8
        + tag_rows * tag_row_h
        + (18 if has_meta else 0)
        + pad
    )
    card_bottom = min(y + card_h, bottom)
    _rounded_rect(
        draw,
        (x, y, x + width, card_bottom),
        8,
        fill=THEME.card_current_bg,
        outline=THEME.accent_soft,
    )

    cy = y + pad
    draw.text((x + pad, cy), label, font=fonts.frame_big, fill=THEME.text_primary)
    _draw_pills(
        draw,
        decision.action_chips,
        first_x=chip_first_x,
        wrap_x=x + pad,
        y=cy + (label_h - chip_row_h) // 2 + 2,
        max_right=right,
        row_h=chip_row_h,
        font=fonts.chip,
        fg=THEME.chip_fg,
        bg=THEME.accent,
    )
    cy += header_h

    for line in lines:
        draw.text((x + pad, cy), line, font=fonts.body, fill=THEME.text_secondary)
        cy += body_lh
    cy += 8

    if tag_rows:
        _draw_pills(
            draw,
            decision.tool_tags,
            first_x=x + pad,
            wrap_x=x + pad,
            y=cy,
            max_right=right,
            row_h=tag_row_h,
            font=fonts.tiny,
            fg=THEME.tag_fg,
            bg=THEME.tag_bg,
        )
        cy += tag_rows * tag_row_h

    if has_meta:
        draw.text(
            (x + pad, cy),
            _format_meta(decision),
            font=fonts.small,
            fill=THEME.text_muted,
        )

    return card_bottom + 10


def _draw_collapsed_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    decision: Decision,
    fonts: FontSet,
) -> int:
    height = 34
    _rounded_rect(draw, (x, y, x + width, y + height), 7, fill=THEME.card_bg)
    label = f"Batch {decision.index + 1}"
    draw.text((x + 12, y + 9), label, font=fonts.title, fill=THEME.text_secondary)

    chip_x = x + 12 + _text_width(label, fonts.title) + 10
    right = x + width - 12
    for chip in decision.action_chips:
        if chip_x + _pill_width(chip, fonts.chip) > right:
            draw.text((chip_x, y + 9), "…", font=fonts.title, fill=THEME.text_muted)
            break
        chip_x = _draw_chip(draw, chip_x, y + 7, chip, fonts.chip) + 6
    return y + height + 8


def _format_meta(decision: Decision) -> str:
    bits: list[str] = []
    if decision.tokens is not None:
        bits.append(f"{decision.tokens:,} out tok")
    if decision.cost_usd is not None:
        bits.append(f"${decision.cost_usd:.4f}")
    if decision.duration_s is not None:
        bits.append(_format_duration(decision.duration_s))
    return "   ".join(bits)


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> int:
    return _draw_pill(draw, x, y, text, font, THEME.chip_fg, THEME.accent)


def _pill_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    return _text_width(text, font) + 16  # 8px horizontal padding each side


def _pill_rows(
    pills: Sequence[str],
    *,
    first_x: int,
    wrap_x: int,
    max_right: int,
    font: ImageFont.FreeTypeFont,
) -> int:
    """Number of lines `pills` occupy when wrapped at `max_right`."""
    if not pills:
        return 0
    x = first_x
    rows = 1
    for text in pills:
        w = _pill_width(text, font)
        if x + w > max_right and x > wrap_x:
            rows += 1
            x = wrap_x
        x += w + 6
    return rows


def _draw_pills(
    draw: ImageDraw.ImageDraw,
    pills: Sequence[str],
    *,
    first_x: int,
    wrap_x: int,
    y: int,
    max_right: int,
    row_h: int,
    font: ImageFont.FreeTypeFont,
    fg: RGBA,
    bg: RGBA,
) -> None:
    """Draw pills left-to-right, wrapping onto new rows so none overflow."""
    x = first_x
    cy = y
    for text in pills:
        w = _pill_width(text, font)
        if x + w > max_right and x > wrap_x:
            cy += row_h
            x = wrap_x
        _draw_pill(draw, x, cy, text, font, fg, bg, width=w)
        x += w + 6


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fg: RGBA,
    bg: RGBA,
    *,
    width: int | None = None,
) -> int:
    pad_x = 8
    text_w = _text_width(text, font)
    box_w = width if width is not None else text_w + 2 * pad_x
    line_h = _font_line_height(font)
    box_h = line_h + 4
    _rounded_rect(draw, (x, y, x + box_w, y + box_h), box_h // 2, fill=bg)
    tx = x + (box_w - text_w) // 2 if width is not None else x + pad_x
    draw.text((tx, y + 2), text, font=font, fill=fg)
    return x + box_w


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    *,
    fill: RGBA | None = None,
    outline: RGBA | None = None,
    width: int = 1,
) -> None:
    x0, y0, x1, y1 = box
    if x1 - x0 < 2 * radius or y1 - y0 < 2 * radius:
        radius = max(0, min((x1 - x0) // 2, (y1 - y0) // 2))
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _blend(color: RGBA, toward: RGBA, t: float) -> RGBA:
    """Blend `color` toward `toward` by fraction t (0 = color, 1 = toward)."""
    return cast(
        RGBA,
        tuple(round(a + (b - a) * t) for a, b in zip(color, toward)),
    )


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    cx: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: RGBA,
) -> None:
    draw.text((cx - _text_width(text, font) // 2, y), text, font=font, fill=fill)


def _draw_right_aligned(
    draw: ImageDraw.ImageDraw,
    text: str,
    right: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: RGBA,
) -> None:
    draw.text((right - _text_width(text, font), y), text, font=font, fill=fill)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_gif(
    frames: Sequence[RecordingFrame],
    output_path: str | Path,
    *,
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
    decisions: Sequence[Decision] | None = None,
    total_frames: int | None = None,
) -> Path:
    """Export recording frames to an animated GIF."""
    if fps < 1:
        raise ValueError("fps must be >= 1")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    images = render_recording_images(
        frames,
        scale=scale,
        overlay=overlay,
        reasoning=reasoning,
        decisions=decisions,
        total_frames=total_frames,
    )
    gif_frames = [
        image.convert("P", palette=Image.Palette.ADAPTIVE) for image in images
    ]
    first_frame = gif_frames[0]
    appended_frames = gif_frames[1:]
    duration_ms = max(1, round(1000 / fps))

    first_frame.save(
        output,
        save_all=True,
        append_images=appended_frames,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output


def export_mp4(
    frames: Sequence[RecordingFrame],
    output_path: str | Path,
    *,
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
    decisions: Sequence[Decision] | None = None,
    total_frames: int | None = None,
) -> Path:
    """Export recording frames to an MP4 via ffmpeg."""
    if fps < 1:
        raise ValueError("fps must be >= 1")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "MP4 export requires ffmpeg on PATH. Install ffmpeg or use --format gif."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    images = _pad_to_even_size(
        render_recording_images(
            frames,
            scale=scale,
            overlay=overlay,
            reasoning=reasoning,
            decisions=decisions,
            total_frames=total_frames,
        )
    )
    width, height = images[0].size
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for image in images:
            process.stdin.write(image.convert("RGB").tobytes())
    except BrokenPipeError as exc:
        _, stderr = process.communicate()
        raise RuntimeError(_ffmpeg_error(stderr)) from exc
    else:
        _, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(_ffmpeg_error(stderr))
    return output


def export_recording(
    frames: Sequence[RecordingFrame],
    output_path: str | Path,
    *,
    output_format: RenderFormat = "gif",
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
    decisions: Sequence[Decision] | None = None,
    total_frames: int | None = None,
) -> Path:
    """Export recording frames to the requested video format."""
    kwargs: dict[str, Any] = dict(
        fps=fps,
        scale=scale,
        overlay=overlay,
        reasoning=reasoning,
        decisions=decisions,
        total_frames=total_frames,
    )
    if output_format == "gif":
        return export_gif(frames, output_path, **kwargs)
    if output_format == "mp4":
        return export_mp4(frames, output_path, **kwargs)
    raise ValueError(f"Unsupported output format: {output_format}")


def default_output_path(
    path: str | Path, *, output_format: RenderFormat = "gif"
) -> Path:
    recording_path = Path(path)
    suffix = f".{output_format}"
    if recording_path.name.endswith(RECORDING_SUFFIX):
        stem = recording_path.name[: -len(RECORDING_SUFFIX)]
        return recording_path.with_name(f"{stem}{suffix}")
    return recording_path.with_suffix(suffix)


def output_path_for_recording(
    recording_path: str | Path,
    output_arg: str | Path | None,
    *,
    recording_count: int = 1,
    output_format: RenderFormat = "gif",
) -> Path:
    """Resolve the video path for one recording in file-mode or directory-mode."""
    recording = Path(recording_path)
    if output_arg is None:
        return default_output_path(recording, output_format=output_format)

    output = Path(output_arg)
    if recording_count == 1 and output.suffix.lower() in VIDEO_SUFFIXES:
        return output
    if recording_count > 1 and output.suffix.lower() in VIDEO_SUFFIXES:
        raise ValueError(
            "--output must be a directory when rendering multiple recordings"
        )
    return output / default_output_path(recording, output_format=output_format).name


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def render_recording_file(
    recording_path: str | Path,
    output_path: str | Path,
    *,
    output_format: RenderFormat = "gif",
    fps: int = 10,
    scale: int = 8,
    overlay: bool = True,
    trace_log: str | Path | None = None,
    reasoning: bool = True,
    max_frames: int | None = None,
) -> RenderSummary:
    """Render one recording file and return metadata for CLI reporting."""
    recording = Path(recording_path)
    frames = load_recording_frames(recording)
    total_frames = len(frames)

    resolved_trace_log: Path | None = None
    trace_events: list[TraceEvent] = []
    if reasoning:
        resolved_trace_log = Path(trace_log) if trace_log is not None else None
        resolved_trace_log = resolved_trace_log or find_trace_log(recording)
        if resolved_trace_log is not None:
            trace_events = parse_trace_events(resolved_trace_log)

    decisions, style = build_decisions(frames, recording, trace_events or None)

    render_frames = _downsample(frames, max_frames)
    output = export_recording(
        render_frames,
        output_path,
        output_format=output_format,
        fps=fps,
        scale=scale,
        overlay=overlay,
        reasoning=reasoning,
        decisions=decisions,
        total_frames=total_frames,
    )
    grid_frame_count = len(expand_recording_frames(render_frames))
    return RenderSummary(
        recording=recording,
        output=output,
        frame_events=len(render_frames),
        grid_frames=grid_frame_count,
        trace_log=resolved_trace_log,
        decision_count=len(decisions),
        style=style,
    )


def _downsample(
    frames: Sequence[RecordingFrame], max_frames: int | None
) -> list[RecordingFrame]:
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    stride = math.ceil(len(frames) / max_frames)
    sampled = list(frames[::stride])
    if sampled and sampled[-1].index != frames[-1].index:
        sampled.append(frames[-1])
    return sampled


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render ARC-AGI recordings to a blog-style MP4/GIF replay: game grid "
            "+ level header on the left, a Reasoning Log of decision cards "
            "(action + reasoning + tool use) on the right. Input may be one "
            ".recording.jsonl/Hermes .jsonl file or a run directory."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a recording JSONL file or run folder such as logs/<run-id> "
            "or logs/<hermes-run>/<game-id>."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file path for one recording, or output directory for a run "
            "folder. Defaults to writing each video beside its recording."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="mp4",
        help="Output video format. Defaults to mp4.",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Playback frames per second"
    )
    parser.add_argument("--scale", type=int, default=8, help="Pixel-art scale factor")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Stride-downsample to at most N rendered frames (useful for previewing "
            "very long Hermes runs). Default: render every frame."
        ),
    )
    parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help=(
            "Optional .trace.jsonl file for Continual-Harness reasoning + tool use. "
            "Defaults to auto-discovery beside the recording."
        ),
    )
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="Render only the game grid, with no header/timeline/reasoning chrome.",
    )
    # Back-compat: older invocations passed these; accept and map onto --no-panel.
    parser.add_argument("--no-overlay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-reasoning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--actions-log", type=Path, default=None, help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = cast(Path, args.input)
    output_arg = cast(Path | None, args.output)
    output_format = cast(RenderFormat, args.format)
    fps = cast(int, args.fps)
    scale = cast(int, args.scale)
    max_frames = cast("int | None", args.max_frames)
    trace_log_arg = cast(Path | None, args.trace_log)
    overlay = not (cast(bool, args.no_panel) or cast(bool, args.no_overlay))
    reasoning = overlay and not cast(bool, args.no_reasoning)

    recordings = discover_recording_paths(input_path)
    summaries: list[RenderSummary] = []
    for recording_path in recordings:
        output_path = output_path_for_recording(
            recording_path,
            output_arg,
            recording_count=len(recordings),
            output_format=output_format,
        )
        summary = render_recording_file(
            recording_path,
            output_path,
            output_format=output_format,
            fps=fps,
            scale=scale,
            overlay=overlay,
            trace_log=trace_log_arg,
            reasoning=reasoning,
            max_frames=max_frames,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary.output} from {summary.frame_events} frame events "
            f"({summary.grid_frames} rendered video frames)"
        )
        if overlay and summary.decision_count:
            source = (
                summary.trace_log
                if summary.trace_log is not None
                else "recording (embedded reasoning)"
            )
            print(
                f"  {summary.decision_count} decision cards "
                f"[{summary.style}] from {source}"
            )
    if len(summaries) > 1:
        print(f"Rendered {len(summaries)} recordings from {input_path}")
    return 0


# --------------------------------------------------------------------------- #
# Trace parsing internals
# --------------------------------------------------------------------------- #
def _build_trace_event(raw: JsonObject, *, file_index: int) -> TraceEvent | None:
    action_counter = _int_or_none(raw.get("action_counter"))
    if action_counter is None:
        return None

    tools_exposed = _clean_string(raw.get("tools_exposed"))
    kind: EventKind
    if tools_exposed == "evolution":
        kind = "evolution"
    elif tools_exposed == "subagent":
        kind = "subagent"
    else:
        kind = "orchestrator"

    reasoning = _clean_string(raw.get("reasoning"))
    error = _clean_string(raw.get("error"))
    actions_executed = _int_or_none(raw.get("actions_executed"))
    force = bool(raw.get("force_take_actions"))

    emitted = _output_function_calls(raw)
    executed_records = _executed_tool_call_records(raw)
    tc_index = 0
    calls: list[CallEntry] = []
    for fc in emitted:
        name = _clean_string(fc.get("name"))
        if name is None:
            continue
        args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
        if name == "take_actions":
            calls.append(
                CallEntry(
                    name=name,
                    kind="action_batch",
                    args=cast(JsonObject, args),
                    executed=(actions_executed or 0) > 0,
                    actions_committed=actions_executed,
                )
            )
            continue
        if _is_action_name(name):
            calls.append(
                CallEntry(
                    name=name,
                    kind="action_rejected",
                    args=cast(JsonObject, args),
                    executed=False,
                )
            )
            continue
        if name == "evolve_system_prompt":
            calls.append(
                CallEntry(
                    name=name,
                    kind="evolution",
                    args=cast(JsonObject, args),
                    executed=True,
                )
            )
            continue

        record = (
            executed_records[tc_index] if tc_index < len(executed_records) else None
        )
        tc_index += 1
        result_value = record.get("result") if isinstance(record, dict) else None
        record_error = (
            _clean_string(record.get("error")) if isinstance(record, dict) else None
        )
        actions_inline = None
        if isinstance(record, dict):
            actions_inline = _int_or_none(record.get("actions_taken_inline"))
            if not actions_inline and isinstance(result_value, dict):
                actions_inline = _int_or_none(result_value.get("actions_taken_inline"))
        calls.append(
            CallEntry(
                name=name,
                kind="analysis",
                args=cast(JsonObject, args),
                executed=record is not None,
                result=cast(JsonObject, result_value)
                if isinstance(result_value, dict)
                else None,
                error=record_error,
                actions_taken_inline=actions_inline,
            )
        )

    evolution_entry: PromptEvolutionEntry | None = None
    evolution_raw = raw.get("evolution")
    if kind == "evolution" and isinstance(evolution_raw, dict):
        generation = _int_or_none(evolution_raw.get("generation")) or 0
        evolution_entry = PromptEvolutionEntry(
            generation=generation,
            accepted=bool(evolution_raw.get("accepted")),
            reasoning=reasoning,
            validation_error=_clean_string(evolution_raw.get("validation_error")),
        )

    subagent_info: JsonObject | None = None
    subagent_raw = raw.get("subagent")
    if kind == "subagent" and isinstance(subagent_raw, dict):
        subagent_info = cast(JsonObject, subagent_raw)

    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    usage_cost = (
        raw.get("usage_cost") if isinstance(raw.get("usage_cost"), dict) else None
    )

    return TraceEvent(
        file_index=file_index,
        vlm_call=_int_or_none(raw.get("vlm_call")),
        round=_int_or_none(raw.get("round")),
        action_counter=action_counter,
        tools_exposed=tools_exposed,
        kind=kind,
        actions_executed=actions_executed,
        force_take_actions=force,
        reasoning=reasoning,
        error=error,
        calls=tuple(calls),
        evolution=evolution_entry,
        subagent_info=subagent_info,
        usage=cast("JsonObject | None", usage),
        usage_cost=cast("JsonObject | None", usage_cost),
        timestamp=_clean_string(raw.get("timestamp")),
    )


def _output_function_calls(event: JsonObject) -> list[JsonObject]:
    output = event.get("output")
    if not isinstance(output, dict):
        return []
    function_calls = output.get("function_calls")
    if not isinstance(function_calls, list):
        return []
    return [cast(JsonObject, call) for call in function_calls if isinstance(call, dict)]


def _executed_tool_call_records(event: JsonObject) -> list[JsonObject]:
    records = event.get("tool_calls")
    if not isinstance(records, list):
        return []
    return [cast(JsonObject, r) for r in records if isinstance(r, dict)]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _load_json_object(line: str, path: Path, line_number: int) -> JsonObject:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} line {line_number}: {exc.msg}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path} line {line_number}")
    return value


def _palette_rgba(value: Any) -> tuple[int, int, int, int]:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Grid value is not an integer: {value!r}") from exc

    if index < 0 or index >= len(ARC_PALETTE):
        raise ValueError(f"Grid value must be in 0..15, got {index}")
    return ARC_PALETTE[index]


def _frame_layers(data: JsonObject) -> list[Grid]:
    raw_frame = data.get("frame")
    if not isinstance(raw_frame, list) or not raw_frame:
        raise ValueError("Frame event must contain a non-empty frame list")

    layers: list[Grid] = []
    for layer_index, layer in enumerate(raw_frame):
        if not isinstance(layer, list) or not layer:
            raise ValueError(f"Frame layer {layer_index} must be a non-empty grid")
        if not all(isinstance(row, list) for row in layer):
            raise ValueError(f"Frame layer {layer_index} must contain row lists")
        layers.append(cast(Grid, layer))
    return layers


def _recording_agent_hint(recording_path: Path) -> str | None:
    name = recording_path.name
    if name.endswith(RECORDING_SUFFIX):
        stem = name[: -len(RECORDING_SUFFIX)]
    else:
        stem = recording_path.stem

    prefix, separator, suffix = stem.rpartition(".")
    if separator and _UUID_RE.fullmatch(suffix):
        stem = prefix
    return stem or None


def _is_recording_jsonl(path: Path) -> bool:
    name = path.name
    if name.endswith(RECORDING_SUFFIX):
        return True
    if path.suffix != ".jsonl":
        return False
    if (
        name.endswith(TRACE_SUFFIX)
        or name.endswith(TRAJECTORY_SUFFIX)
        or name == TRAJECTORY_SUFFIX.removeprefix(".")
        or name == "prompt_evolution.jsonl"
        or name == "observations.jsonl"
    ):
        return False
    return True


def _recordings_in_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []

    recordings = [
        candidate
        for candidate in sorted(path.iterdir())
        if candidate.is_file() and _is_recording_jsonl(candidate)
    ]
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        recordings.extend(
            candidate
            for candidate in sorted(child.iterdir())
            if candidate.is_file() and _is_recording_jsonl(candidate)
        )
    return recordings


def _run_log_for_recording(recording_path: Path) -> Path | None:
    run_dir = _run_dir_for_recording(recording_path)
    if run_dir is None:
        return None
    for candidate in (
        run_dir / "run.log",
        run_dir / "logs" / "hermes.log",
        run_dir / "hermes_memory" / "logs" / "agent.log",
    ):
        if candidate.exists():
            return candidate
    return None


def _artifact_companion_for_recording(recording_path: Path, suffix: str) -> Path | None:
    run_dir = _run_dir_for_recording(recording_path)
    if run_dir is None:
        return None
    stem = recording_path.name
    if stem.endswith(RECORDING_SUFFIX):
        stem = stem[: -len(RECORDING_SUFFIX)]
    candidates = [run_dir / "artifacts" / f"{stem}{suffix}"]
    if suffix == TRACE_SUFFIX:
        candidates.append(run_dir / "trace.jsonl")
    if suffix == TRAJECTORY_SUFFIX:
        candidates.append(run_dir / "logs" / "trajectory.jsonl")
        candidates.append(run_dir / "trajectory.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_dir_for_recording(recording_path: Path) -> Path | None:
    if recording_path.parent.name == "recordings":
        return recording_path.parent.parent
    if recording_path.parent.parent.name == "recordings":
        return recording_path.parent.parent.parent
    return None


def _trace_mentions_agent(trace_path: Path, agent_hint: str) -> bool:
    try:
        with trace_path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                if agent_hint in line:
                    return True
                if line_index >= 20:
                    break
    except OSError:
        return False
    return False


def _wrap_text_to_width(text: str, font: Any, max_width: int) -> list[str]:
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped.append("")
            continue

        current = ""
        for word in raw_line.split():
            candidate = word if not current else f"{current} {word}"
            if _text_width(candidate, font) <= max_width:
                current = candidate
                continue

            if current:
                wrapped.append(current)
            if _text_width(word, font) <= max_width:
                current = word
            else:
                pieces = _break_word_to_width(word, font, max_width)
                wrapped.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""

        if current:
            wrapped.append(current)
    return wrapped


def _break_word_to_width(word: str, font: Any, max_width: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for character in word:
        candidate = f"{current}{character}"
        if not current or _text_width(candidate, font) <= max_width:
            current = candidate
            continue
        pieces.append(current)
        current = character
    if current:
        pieces.append(current)
    return pieces


def _fit_text_to_width(text: str, font: Any, max_width: int, *, suffix: str) -> str:
    suffix_width = _text_width(suffix, font)
    if _text_width(text, font) + suffix_width <= max_width:
        return f"{text}{suffix}"
    if suffix_width >= max_width:
        return suffix

    output = text
    while output and _text_width(f"{output}{suffix}", font) > max_width:
        output = output[:-1].rstrip()
    return f"{output}{suffix}" if output else suffix


def _font_line_height(font: Any) -> int:
    try:
        bbox = font.getbbox("Ag")
    except AttributeError:
        return 14
    return int(max(12, bbox[3] - bbox[1] + 6))


def _text_width(text: str, font: Any) -> int:
    try:
        return int(round(font.getlength(text)))
    except AttributeError:
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0])


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_action_name(name: str) -> bool:
    return name == "RESET" or name in _ACTION_NAMES.values()


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _pad_to_common_size(images: Sequence[Image.Image]) -> list[Image.Image]:
    if not images:
        return []

    width = max(image.width for image in images)
    height = max(image.height for image in images)
    padded: list[Image.Image] = []
    for image in images:
        if image.size == (width, height):
            padded.append(image)
            continue
        canvas = Image.new("RGBA", (width, height), THEME.bg)
        canvas.paste(image, (0, 0))
        padded.append(canvas)
    return padded


def _pad_to_even_size(images: Sequence[Image.Image]) -> list[Image.Image]:
    """Pad video frames to even dimensions required by yuv420p MP4 output."""
    if not images:
        return []

    width, height = images[0].size
    even_width = width if width % 2 == 0 else width + 1
    even_height = height if height % 2 == 0 else height + 1
    if (even_width, even_height) == (width, height):
        return list(images)

    padded: list[Image.Image] = []
    for image in images:
        canvas = Image.new("RGBA", (even_width, even_height), THEME.bg)
        canvas.paste(image, (0, 0))
        padded.append(canvas)
    return padded


def _ffmpeg_error(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return "ffmpeg failed while exporting MP4"
    tail = "\n".join(text.splitlines()[-10:])
    return f"ffmpeg failed while exporting MP4:\n{tail}"


if __name__ == "__main__":
    raise SystemExit(main())
