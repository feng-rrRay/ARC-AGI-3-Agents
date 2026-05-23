from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from PIL import Image, ImageDraw, ImageFont

RECORDING_SUFFIX = ".recording.jsonl"
TRACE_SUFFIX = ".trace.jsonl"
TRAJECTORY_SUFFIX = ".trajectory.jsonl"
PROMPT_EVOLUTION_NAME = "prompt_evolution.jsonl"
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


@dataclass(frozen=True)
class RecordingFrame:
    index: int
    timestamp: str | None
    data: JsonObject
    action_label: str | None = None
    reasoning_trace: str | None = None


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
    """One row from the trace JSONL, normalized for the renderer.

    Each VLM call writes one event (orchestrator). Subagent inner rounds and
    prompt-evolution attempts write their own events; both appear in file
    order BEFORE the orchestrator event that triggered them, so file-order
    iteration is sufficient for grouping.
    """

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


@dataclass(frozen=True)
class ActionPanel:
    """The right-side panel that a contiguous batch of action frames shares.

    `events` holds every TraceEvent that contributed to those frames in file
    order — typically zero or more no-action / evolution / subagent events
    followed by exactly one action-emitting orchestrator event.
    """

    frame_start: int  # inclusive recording-frame index
    frame_end: int    # inclusive recording-frame index
    events: tuple[TraceEvent, ...]


@dataclass(frozen=True)
class RenderSummary:
    recording: Path
    output: Path
    frame_events: int
    grid_frames: int
    actions_log: Path | None
    action_label_count: int
    trace_log: Path | None
    reasoning_trace_count: int
    attached_reasoning_count: int
    trajectory_log: Path | None = None
    trajectory_step_count: int = 0
    prompt_evolution_log: Path | None = None
    prompt_evolution_count: int = 0
    panel_count: int = 0


def discover_recording_paths(path: str | Path) -> list[Path]:
    """Return recording files from a single file or a run/recordings directory."""
    input_path = Path(path)
    if input_path.is_file():
        if not input_path.name.endswith(RECORDING_SUFFIX):
            raise ValueError(f"Input file must end with {RECORDING_SUFFIX}: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"No such recording file or run directory: {input_path}")

    recordings_dir = input_path / "recordings"
    search_dir = recordings_dir if recordings_dir.is_dir() else input_path
    recordings = sorted(search_dir.glob(f"*{RECORDING_SUFFIX}"))
    if not recordings:
        detail = f"{recordings_dir} or {input_path}" if search_dir != input_path else input_path
        raise ValueError(f"No *{RECORDING_SUFFIX} files found in {detail}")
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
    """Parse action labels from a run log keyed by Agent action count.

    Supports both the legacy `<game> - ACTION1: count N, ...` line format and
    the continual-harness `[<game>] step=N ACTION1[(args)]? src=...` format.
    """
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
    """Find a VLM trajectory JSONL file alongside the recording's artifacts."""
    return _artifact_companion_for_recording(Path(recording_path), TRAJECTORY_SUFFIX)


def find_prompt_evolution_log(recording_path: str | Path) -> Path | None:
    """Find prompt_evolution.jsonl in the recording's run directory."""
    run_dir = _run_dir_for_recording(Path(recording_path))
    if run_dir is None:
        return None
    candidate = run_dir / PROMPT_EVOLUTION_NAME
    return candidate if candidate.exists() else None


def parse_trace_events(path: str | Path) -> list[TraceEvent]:
    """Parse all VLM trace events into a single ordered list.

    File order is preserved because the continual-harness scaffold relies on it
    for grouping: evolution / subagent rows always appear immediately before
    the orchestrator row that triggered them.
    """
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
    """Bucket trace events into one panel per batch of action frames.

    A panel closes when an orchestrator event has actions_executed > 0. Every
    preceding event since the last close (no-action orchestrator rows, prompt-
    evolution rows, subagent rows) belongs to the same panel and renders ahead
    of the action-emitting one. A trailing run of no-action events at end-of-
    file has no frames to attach to and is dropped.
    """
    panels: list[ActionPanel] = []
    buffer: list[TraceEvent] = []
    for event in events:
        buffer.append(event)
        if (
            event.kind == "orchestrator"
            and event.actions_executed is not None
            and event.actions_executed > 0
        ):
            frame_end = event.action_counter - 1
            frame_start = frame_end - event.actions_executed + 1
            panels.append(
                ActionPanel(
                    frame_start=frame_start,
                    frame_end=frame_end,
                    events=tuple(buffer),
                )
            )
            buffer = []
    return panels


def parse_trajectory_log(path: str | Path) -> dict[int, list[JsonObject]]:
    """Parse executed tool_calls per action_counter from a trajectory JSONL file."""
    trajectory_path = Path(path)
    by_step: dict[int, list[JsonObject]] = {}

    with trajectory_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            event = _load_json_object(stripped, trajectory_path, line_number)
            action_counter = _int_or_none(event.get("action_counter"))
            if action_counter is None:
                continue

            calls = event.get("tool_calls")
            if not isinstance(calls, list):
                continue
            by_step[action_counter] = [
                cast(JsonObject, call) for call in calls if isinstance(call, dict)
            ]

    return by_step


def parse_prompt_evolution_log(path: str | Path) -> dict[int, PromptEvolutionEntry]:
    """Parse prompt-evolution entries per action_counter."""
    evolution_path = Path(path)
    by_step: dict[int, PromptEvolutionEntry] = {}

    with evolution_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            event = _load_json_object(stripped, evolution_path, line_number)
            action_counter = _int_or_none(event.get("action_counter"))
            generation = _int_or_none(event.get("generation"))
            if action_counter is None or generation is None:
                continue

            by_step[action_counter] = PromptEvolutionEntry(
                generation=generation,
                accepted=bool(event.get("accepted")),
                reasoning=_clean_string(event.get("reasoning")),
                validation_error=_clean_string(event.get("validation_error")),
            )

    return by_step


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


def apply_reasoning_traces(
    frames: Sequence[RecordingFrame],
    reasoning_traces: dict[int, str],
) -> list[RecordingFrame]:
    """Attach parsed VLM reasoning traces to recording frames by action count."""
    if not reasoning_traces:
        return list(frames)
    return [
        replace(
            frame,
            reasoning_trace=reasoning_traces.get(frame.index)
            or _action_input_reasoning(frame.data),
        )
        for frame in frames
    ]


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


def render_recording_frame(
    frame: RecordingFrame,
    *,
    scale: int = 8,
    overlay: bool = True,
    reasoning_panel: bool = False,
) -> Image.Image:
    """Render the first grid from one recording event."""
    grid_frame = expand_recording_frames([frame])[0]
    return render_recording_grid_frame(
        grid_frame,
        scale=scale,
        overlay=overlay,
        reasoning_panel=reasoning_panel,
    )


def render_recording_grid_frame(
    grid_frame: RecordingGridFrame,
    *,
    scale: int = 8,
    overlay: bool = True,
    reasoning_panel: bool = False,
) -> Image.Image:
    """Render one grid. Multi-grid recording events become sequential GIF frames."""
    image = grid_to_image(grid_frame.grid, scale=scale)
    if overlay:
        image = _add_overlay(image, grid_frame)
    if reasoning_panel:
        image = _add_reasoning_panel(image, grid_frame)
    return image


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


def render_recording_images(
    frames: Sequence[RecordingFrame],
    *,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
) -> list[Image.Image]:
    """Render recording events to one image per grid frame."""
    if not frames:
        raise ValueError("At least one frame is required")

    grid_frames = expand_recording_frames(frames)
    include_reasoning_panel = reasoning and any(
        grid_frame.event.reasoning_trace for grid_frame in grid_frames
    )
    images = [
        render_recording_grid_frame(
            grid_frame,
            scale=scale,
            overlay=overlay,
            reasoning_panel=include_reasoning_panel,
        )
        for grid_frame in grid_frames
    ]
    return _pad_to_common_size(images)


def export_gif(
    frames: Sequence[RecordingFrame],
    output_path: str | Path,
    *,
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    reasoning: bool = True,
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
) -> Path:
    """Export recording frames to the requested video format."""
    if output_format == "gif":
        return export_gif(
            frames,
            output_path,
            fps=fps,
            scale=scale,
            overlay=overlay,
            reasoning=reasoning,
        )
    if output_format == "mp4":
        return export_mp4(
            frames,
            output_path,
            fps=fps,
            scale=scale,
            overlay=overlay,
            reasoning=reasoning,
        )
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


def render_recording_file(
    recording_path: str | Path,
    output_path: str | Path,
    *,
    output_format: RenderFormat = "gif",
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    actions_log: str | Path | None = None,
    trace_log: str | Path | None = None,
    trajectory_log: str | Path | None = None,
    prompt_evolution_log: str | Path | None = None,
    reasoning: bool = True,
) -> RenderSummary:
    """Render one recording file and return metadata for CLI reporting."""
    recording = Path(recording_path)
    frames = load_recording_frames(recording)

    resolved_actions_log = Path(actions_log) if actions_log is not None else None
    resolved_actions_log = resolved_actions_log or find_actions_log(recording)
    if resolved_actions_log is not None:
        action_labels = parse_action_log(resolved_actions_log)
        frames = apply_action_labels(frames, action_labels)
    else:
        action_labels = {}

    resolved_trace_log: Path | None = None
    resolved_trajectory_log: Path | None = None
    resolved_prompt_evolution_log: Path | None = None
    reasoning_traces: dict[int, str] = {}
    trajectory_step_count = 0
    prompt_evolution_count = 0
    panel_count = 0

    if reasoning:
        resolved_trace_log = Path(trace_log) if trace_log is not None else None
        resolved_trace_log = resolved_trace_log or find_trace_log(
            recording,
            actions_log=resolved_actions_log,
        )
        resolved_trajectory_log = (
            Path(trajectory_log) if trajectory_log is not None else None
        )
        resolved_trajectory_log = resolved_trajectory_log or find_trajectory_log(
            recording
        )
        resolved_prompt_evolution_log = (
            Path(prompt_evolution_log) if prompt_evolution_log is not None else None
        )
        resolved_prompt_evolution_log = (
            resolved_prompt_evolution_log or find_prompt_evolution_log(recording)
        )

        events = (
            parse_trace_events(resolved_trace_log)
            if resolved_trace_log is not None
            else []
        )
        panels = group_into_panels(events)
        panel_count = len(panels)
        reasoning_traces = build_panel_reasoning_traces(panels)

        if resolved_trajectory_log is not None:
            trajectory_step_count = len(parse_trajectory_log(resolved_trajectory_log))
        if resolved_prompt_evolution_log is not None:
            prompt_evolution_count = len(
                parse_prompt_evolution_log(resolved_prompt_evolution_log)
            )

        frames = apply_reasoning_traces(frames, reasoning_traces)

    attached_reasoning_count = sum(1 for frame in frames if frame.reasoning_trace)
    grid_frame_count = len(expand_recording_frames(frames))
    output = export_recording(
        frames,
        output_path,
        output_format=output_format,
        fps=fps,
        scale=scale,
        overlay=overlay,
        reasoning=reasoning,
    )
    return RenderSummary(
        recording=recording,
        output=output,
        frame_events=len(frames),
        grid_frames=grid_frame_count,
        actions_log=resolved_actions_log,
        action_label_count=len(action_labels),
        trace_log=resolved_trace_log,
        reasoning_trace_count=len(reasoning_traces),
        attached_reasoning_count=attached_reasoning_count,
        trajectory_log=resolved_trajectory_log,
        trajectory_step_count=trajectory_step_count,
        prompt_evolution_log=resolved_prompt_evolution_log,
        prompt_evolution_count=prompt_evolution_count,
        panel_count=panel_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render ARC-AGI recording JSONL files to GIF or MP4. "
            "Input may be one .recording.jsonl file or a run directory "
            "containing recordings/."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .recording.jsonl file or run folder such as logs/<run-id>.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file path for one recording, or output directory for a run folder. "
            "Defaults to writing each video beside its recording."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="Output video format. Defaults to gif.",
    )
    parser.add_argument("--fps", type=int, default=5, help="Playback frames per second")
    parser.add_argument("--scale", type=int, default=8, help="Pixel-art scale factor")
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable metadata overlay text.",
    )
    parser.add_argument(
        "--actions-log",
        type=Path,
        default=None,
        help=(
            "Optional run log to use for action overlay labels. "
            "Defaults to auto-discovery in logs/."
        ),
    )
    parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help=(
            "Optional .trace.jsonl file to render as a right-side reasoning panel. "
            "Defaults to auto-discovery in logs/."
        ),
    )
    parser.add_argument(
        "--trajectory-log",
        type=Path,
        default=None,
        help=(
            "Optional .trajectory.jsonl file used to enrich the panel with the "
            "results of executed analysis tools. Defaults to the sibling "
            "artifacts/*.trajectory.jsonl."
        ),
    )
    parser.add_argument(
        "--prompt-evolution-log",
        type=Path,
        default=None,
        help=(
            "Optional prompt_evolution.jsonl file used to annotate evolution "
            "steps. Defaults to the run directory's prompt_evolution.jsonl."
        ),
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable the right-side VLM reasoning panel.",
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
    overlay = not cast(bool, args.no_overlay)
    actions_log_arg = cast(Path | None, args.actions_log)
    trace_log_arg = cast(Path | None, args.trace_log)
    trajectory_log_arg = cast(Path | None, args.trajectory_log)
    prompt_evolution_log_arg = cast(Path | None, args.prompt_evolution_log)
    reasoning = not cast(bool, args.no_reasoning)

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
            actions_log=actions_log_arg,
            trace_log=trace_log_arg,
            trajectory_log=trajectory_log_arg,
            prompt_evolution_log=prompt_evolution_log_arg,
            reasoning=reasoning,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary.output} from {summary.frame_events} frame events "
            f"({summary.grid_frames} rendered video frames)"
        )
        if summary.actions_log is not None and summary.action_label_count:
            print(
                f"Loaded {summary.action_label_count} action labels "
                f"from {summary.actions_log}"
            )
        if summary.trace_log is not None and summary.reasoning_trace_count:
            print(
                f"Loaded {summary.reasoning_trace_count} reasoning traces "
                f"({summary.panel_count} batched VLM panels) "
                f"from {summary.trace_log} "
                f"({summary.attached_reasoning_count} attached to recording frames)"
            )
        if summary.trajectory_log is not None and summary.trajectory_step_count:
            print(
                f"Found {summary.trajectory_step_count} trajectory steps "
                f"in {summary.trajectory_log}"
            )
        if (
            summary.prompt_evolution_log is not None
            and summary.prompt_evolution_count
        ):
            print(
                f"Annotated {summary.prompt_evolution_count} prompt-evolution "
                f"steps from {summary.prompt_evolution_log}"
            )
    if len(summaries) > 1:
        print(f"Rendered {len(summaries)} recordings from {input_path}")
    return 0


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


def _run_log_for_recording(recording_path: Path) -> Path | None:
    run_dir = _run_dir_for_recording(recording_path)
    if run_dir is None:
        return None
    run_log = run_dir / "run.log"
    return run_log if run_log.exists() else None


def _artifact_companion_for_recording(
    recording_path: Path, suffix: str
) -> Path | None:
    run_dir = _run_dir_for_recording(recording_path)
    if run_dir is None:
        return None
    stem = recording_path.name
    if stem.endswith(RECORDING_SUFFIX):
        stem = stem[: -len(RECORDING_SUFFIX)]
    candidate = run_dir / "artifacts" / f"{stem}{suffix}"
    return candidate if candidate.exists() else None


def _run_dir_for_recording(recording_path: Path) -> Path | None:
    if recording_path.parent.name != "recordings":
        return None
    run_dir = recording_path.parent.parent
    if run_dir == recording_path.parent or run_dir.parent.name != "logs":
        return None
    return run_dir


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

    # output.function_calls is the raw list the model emitted (in order). The
    # trace's own `tool_calls` field holds executed analysis results in the
    # same order, skipping take_actions (which never produces an analysis
    # record). Walk them in parallel so each analysis call carries its result.
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

        record = executed_records[tc_index] if tc_index < len(executed_records) else None
        tc_index += 1
        result_value = record.get("result") if isinstance(record, dict) else None
        record_error = (
            _clean_string(record.get("error")) if isinstance(record, dict) else None
        )
        actions_inline = (
            _int_or_none(record.get("actions_taken_inline"))
            if isinstance(record, dict)
            else None
        )
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

    # Rejected per-action tools also land in the trace's tool_calls list. Pull
    # any leftover records (with an error) so the panel surfaces them too.
    while tc_index < len(executed_records):
        record = executed_records[tc_index]
        tc_index += 1
        rec_name = _clean_string(record.get("name"))
        if rec_name is None:
            continue
        if any(c.name == rec_name and c.kind == "action_rejected" for c in calls):
            continue
        calls.append(
            CallEntry(
                name=rec_name,
                kind="analysis",
                args=cast(JsonObject, record.get("args") or {}),
                executed=True,
                error=_clean_string(record.get("error")),
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
    )


def _output_function_calls(event: JsonObject) -> list[JsonObject]:
    output = event.get("output")
    if not isinstance(output, dict):
        return []
    function_calls = output.get("function_calls")
    if not isinstance(function_calls, list):
        return []
    return [
        cast(JsonObject, call) for call in function_calls if isinstance(call, dict)
    ]


def _executed_tool_call_records(event: JsonObject) -> list[JsonObject]:
    records = event.get("tool_calls")
    if not isinstance(records, list):
        return []
    return [cast(JsonObject, r) for r in records if isinstance(r, dict)]


def build_panel_reasoning_traces(panels: Sequence[ActionPanel]) -> dict[int, str]:
    """Map every frame index covered by a panel to that panel's rendered text."""
    output: dict[int, str] = {}
    for panel in panels:
        text = _format_action_panel(panel)
        if text is None:
            continue
        for frame_index in range(panel.frame_start, panel.frame_end + 1):
            output[frame_index] = text
    return output


def _format_action_panel(panel: ActionPanel) -> str | None:
    if not panel.events:
        return None

    closing = panel.events[-1]
    actions = closing.actions_executed or 0
    vlm_calls = sum(1 for e in panel.events if e.kind == "orchestrator")
    if panel.frame_start == panel.frame_end:
        frames_label = f"frame={panel.frame_start:03d}"
    else:
        frames_label = f"frames={panel.frame_start:03d}-{panel.frame_end:03d}"

    header = [frames_label, f"vlm_calls={vlm_calls}", f"actions={actions}"]
    if closing.force_take_actions:
        header.append("(force)")

    lines: list[str] = ["VLM reasoning", " ".join(header)]
    for event in panel.events:
        lines.extend(_format_trace_event(event))
    return "\n".join(lines)


def _format_trace_event(event: TraceEvent) -> list[str]:
    if event.kind == "evolution":
        return _format_evolution_event(event)
    if event.kind == "subagent":
        return _format_subagent_event(event)
    return _format_orchestrator_event(event)


def _format_orchestrator_event(event: TraceEvent) -> list[str]:
    call_label = event.tools_exposed or "vlm"
    suffix_bits: list[str] = []
    if event.actions_executed is not None:
        if event.actions_executed > 0:
            suffix_bits.append(f"{event.actions_executed} actions")
        else:
            suffix_bits.append("no actions")
    if event.force_take_actions:
        suffix_bits.append("force")
    suffix = f" [{', '.join(suffix_bits)}]" if suffix_bits else ""

    vlm_label = f"V{event.vlm_call}" if event.vlm_call is not None else "V?"
    lines: list[str] = ["", f"{vlm_label} ({call_label}){suffix}"]
    if event.reasoning:
        lines.append(_compact_whitespace(event.reasoning))
    if event.error:
        lines.append("error: " + _truncate(_compact_whitespace(event.error), 220))
    for call in event.calls:
        lines.extend(_format_call_lines(call))
    return lines


def _format_evolution_event(event: TraceEvent) -> list[str]:
    entry = event.evolution
    if entry is None:
        return ["", "Prompt evolution"]
    accepted = "accepted" if entry.accepted else "rejected"
    lines = ["", f"Prompt evolution gen={entry.generation} [{accepted}]"]
    if entry.reasoning:
        lines.append(_compact_whitespace(entry.reasoning))
    if entry.validation_error is not None:
        lines.append(
            "validation_error: "
            + _truncate(_compact_whitespace(entry.validation_error), 220)
        )
    return lines


def _format_subagent_event(event: TraceEvent) -> list[str]:
    info = event.subagent_info or {}
    name = _clean_string(info.get("name")) or "?"
    inner_round = _int_or_none(info.get("inner_round"))
    inner_max = _int_or_none(info.get("max_inner_rounds"))
    round_label = ""
    if inner_round is not None and inner_max is not None:
        round_label = f" round {inner_round}/{inner_max}"
    elif inner_round is not None:
        round_label = f" round {inner_round}"

    lines = ["", f"subagent {name}{round_label}"]
    if event.error:
        lines.append("error: " + _truncate(_compact_whitespace(event.error), 220))
    for call in event.calls:
        lines.extend(_format_call_lines(call))
    return lines


def _format_call_lines(call: CallEntry) -> list[str]:
    if call.kind == "action_batch":
        return _format_take_actions(call)
    if call.kind == "action_rejected":
        suffix = f" [rejected: {call.error}]" if call.error else " [rejected]"
        return [f"> {call.name}{suffix}"]

    formatter = _CALL_FORMATTERS.get(call.name, _format_generic_call)
    return formatter(call)


def _format_take_actions(call: CallEntry) -> list[str]:
    actions_label = _format_action_specs(call.args.get("actions"))
    committed = call.actions_committed
    head = f"> take_actions {actions_label}"
    if committed is not None:
        head += f" = {committed}"
    if call.error:
        head += " [error]"
    lines = [head]
    _append_detail(lines, "reasoning", call.args.get("reasoning"))
    if call.error:
        _append_detail(lines, "error", call.error)
    return lines


def _format_action_specs(specs: Any) -> str:
    """Render a take_actions `actions` arg as a compact [A1,A6(12,30),...] label."""
    if not isinstance(specs, list):
        return "[?]"
    parts: list[str] = []
    for item in specs:
        if not isinstance(item, dict):
            parts.append("?")
            continue
        name = str(item.get("name") or "?")
        short = name
        if name.startswith("ACTION") and name[6:].isdigit():
            short = f"A{name[6:]}"
        if "x" in item and "y" in item:
            parts.append(f"{short}({item['x']},{item['y']})")
        else:
            parts.append(short)
    return "[" + ",".join(parts) + "]"


def _status_suffix(call: CallEntry) -> str:
    if not call.executed:
        return " [skipped]"
    if call.error:
        return " [error]"
    return ""


_DETAIL_LIMITS: dict[str, int] = {
    "reasoning": 400,
    "description": 400,
    "body": 600,
    "instructions": 600,
    "code": 600,
    "task": 400,
    "answer": 400,
    "query": 200,
    "args": 200,
    "context": 200,
    "stdout": 400,
    "stderr": 400,
    "result": 400,
}


def _append_detail(lines: list[str], label: str, value: Any) -> None:
    text = _stringify_detail_value(value)
    if text is None:
        return
    limit = _DETAIL_LIMITS.get(label, 300)
    lines.append(f"    {label}: " + _truncate(text, limit))


def _stringify_detail_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _compact_whitespace(value).strip() or None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return None
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return str(value)
    return _compact_whitespace(str(value))


def _tag_suffix(args: JsonObject, result: JsonObject) -> str:
    tags = args.get("tags") if isinstance(args.get("tags"), list) else result.get("tags")
    if not isinstance(tags, list) or not tags:
        return ""
    return " [" + ",".join(str(tag) for tag in tags) + "]"


def _format_process_skill(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    op = _clean_string(args.get("operation")) or "?"
    skill_name = _clean_string(args.get("name")) or _clean_string(result.get("name"))
    skill_id = _clean_string(result.get("id")) or _clean_string(args.get("id")) or "?"
    head = f"- process_skill {op} {skill_id}"
    if skill_name:
        head += f' "{skill_name}"'
    head += _tag_suffix(args, result) + _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    if op in {"add", "edit"}:
        _append_detail(lines, "description", args.get("description"))
        _append_detail(lines, "code", args.get("code"))
    if op == "search":
        _append_detail(lines, "query", args.get("query"))
    _append_detail(lines, "error", call.error)
    return lines


def _format_run_skill(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    skill_id = _clean_string(args.get("id")) or _clean_string(result.get("id")) or "?"
    skill_name = _clean_string(result.get("name"))
    head = f"- run_skill {skill_id}"
    if skill_name:
        head += f" ({skill_name})"
    head += _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    _append_detail(lines, "args", args.get("args"))
    _append_detail(lines, "result", result.get("result"))
    _append_detail(lines, "stdout", result.get("stdout"))
    _append_detail(lines, "stderr", result.get("stderr"))
    _append_detail(lines, "error", call.error or result.get("error"))
    return lines


def _format_process_memory(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    op = _clean_string(args.get("operation")) or "?"
    mem_id = _clean_string(result.get("id")) or _clean_string(args.get("id")) or "?"
    title = _clean_string(args.get("title")) or _clean_string(result.get("title"))
    head = f"- process_memory {op} {mem_id}"
    if title:
        head += f' "{title}"'
    head += _tag_suffix(args, result) + _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    if op in {"add", "edit"}:
        _append_detail(lines, "body", args.get("body"))
    if op == "search":
        _append_detail(lines, "query", args.get("query"))
    _append_detail(lines, "error", call.error)
    return lines


def _format_process_subagent(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    op = _clean_string(args.get("operation")) or "?"
    sub_name = _clean_string(args.get("name")) or _clean_string(result.get("name"))
    sub_id = _clean_string(result.get("id")) or _clean_string(args.get("id")) or "?"
    head = f"- process_subagent {op} {sub_id}"
    if sub_name:
        head += f' "{sub_name}"'
    head += _tag_suffix(args, result) + _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    if op in {"add", "edit"}:
        _append_detail(lines, "description", args.get("description"))
        _append_detail(lines, "instructions", args.get("instructions"))
        _append_detail(lines, "allowed_tools", args.get("allowed_tools"))
    if op == "search":
        _append_detail(lines, "query", args.get("query"))
    _append_detail(lines, "error", call.error)
    return lines


def _format_run_subagent(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    sub_id = _clean_string(args.get("id")) or _clean_string(result.get("id")) or "?"
    sub_name = _clean_string(result.get("name"))
    head = f"- run_subagent {sub_id}"
    if sub_name:
        head += f" ({sub_name})"
    rounds_used = result.get("rounds_used")
    if rounds_used is not None:
        head += f" rounds={rounds_used}"
    head += _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    _append_detail(lines, "task", args.get("task"))
    _append_detail(lines, "context", args.get("context"))
    _append_detail(lines, "result", result.get("result"))
    _append_detail(lines, "warning", result.get("warning"))
    _append_detail(lines, "error", call.error or result.get("error"))
    return lines


def _format_get_recent_trajectory(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    limit = args.get("limit")
    count = result.get("count")
    head = "- get_recent_trajectory"
    if limit is not None:
        head += f"(limit={limit})"
    if count is not None:
        head += f" -> {count} steps"
    head += _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    _append_detail(lines, "error", call.error)
    return lines


def _format_run_code(call: CallEntry) -> list[str]:
    args, result = call.args, call.result or {}
    head = "- run_code" + _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", args.get("reasoning"))
    _append_detail(lines, "code", args.get("code"))
    _append_detail(lines, "args", args.get("args"))
    _append_detail(lines, "result", result.get("result"))
    _append_detail(lines, "stdout", result.get("stdout"))
    _append_detail(lines, "stderr", result.get("stderr"))
    _append_detail(lines, "error", call.error or result.get("error"))
    return lines


def _format_evolve_system_prompt(call: CallEntry) -> list[str]:
    return ["- evolve_system_prompt" + _status_suffix(call)]


def _format_generic_call(call: CallEntry) -> list[str]:
    op = _clean_string(call.args.get("operation"))
    head = f"- {call.name}"
    if op is not None:
        head += f" {op}"
    head += _status_suffix(call)
    lines = [head]
    _append_detail(lines, "reasoning", call.args.get("reasoning"))
    _append_detail(lines, "error", call.error)
    return lines


_CALL_FORMATTERS: dict[str, Any] = {
    "process_skill": _format_process_skill,
    "run_skill": _format_run_skill,
    "process_memory": _format_process_memory,
    "process_subagent": _format_process_subagent,
    "run_subagent": _format_run_subagent,
    "get_recent_trajectory": _format_get_recent_trajectory,
    "run_code": _format_run_code,
    "evolve_system_prompt": _format_evolve_system_prompt,
    "take_actions": _format_take_actions,
}


def _action_input_reasoning(data: JsonObject) -> str | None:
    action_input = data.get("action_input")
    if not isinstance(action_input, dict):
        return None

    reasoning = _clean_string(action_input.get("reasoning"))
    if reasoning is not None:
        return reasoning

    action_data = action_input.get("data")
    if isinstance(action_data, dict):
        return _clean_string(action_data.get("reasoning"))
    return None


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


def _add_reasoning_panel(
    image: Image.Image,
    grid_frame: RecordingGridFrame,
) -> Image.Image:
    panel_width = image.width
    output = Image.new(
        "RGBA",
        (image.width + panel_width, image.height),
        (15, 18, 22, 255),
    )
    output.paste(image, (0, 0))

    panel_left = image.width
    draw = ImageDraw.Draw(output)
    draw.rectangle(
        (panel_left, 0, output.width, output.height),
        fill=(15, 18, 22, 255),
    )
    draw.line(
        (panel_left, 0, panel_left, output.height),
        fill=(72, 78, 88, 255),
        width=1,
    )

    text = (
        grid_frame.event.reasoning_trace
        or f"VLM reasoning\nstep={grid_frame.event.index:03d}\n\nNo trace for this step."
    )
    _draw_panel_text(
        draw,
        text,
        x=panel_left + 8,
        y=8,
        width=panel_width - 16,
        height=image.height - 16,
    )
    return output


def _draw_panel_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    font = ImageFont.load_default()
    line_height = _font_line_height(font)
    lines = _wrap_text_to_width(text, font, width)
    max_lines = max(1, height // line_height)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit_text_to_width(lines[-1], font, width, suffix="...")

    for line_index, line in enumerate(lines):
        fill = _panel_line_color(line_index, line)
        draw.text((x, y + line_index * line_height), line, fill=fill, font=font)


_VLM_HEADER_RE = re.compile(r"^V[\d?]+ \([^()]+\)(?: \[[^\]]+\])?$")
_SUBAGENT_HEADER_RE = re.compile(r"^subagent ")


def _panel_line_color(line_index: int, line: str) -> tuple[int, int, int, int]:
    if line_index == 0 and line == "VLM reasoning":
        return (255, 255, 255, 255)
    if (
        line.startswith("step=")
        or line.startswith("frame=")
        or line.startswith("frames=")
        or line.startswith("Prompt evolution")
    ):
        return (188, 207, 255, 255)
    if _VLM_HEADER_RE.match(line) or _SUBAGENT_HEADER_RE.match(line):
        return (148, 199, 247, 255)
    if line.startswith("> "):
        return (255, 196, 138, 255)
    if "[skipped]" in line or "[no actions]" in line:
        return (140, 144, 152, 255)
    if (
        "[error]" in line
        or "[rejected" in line
        or line.startswith("error:")
        or line.startswith("validation_error:")
    ):
        return (240, 130, 130, 255)
    return (226, 232, 240, 255)


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


def _break_word_to_width(
    word: str,
    font: Any,
    max_width: int,
) -> list[str]:
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


def _fit_text_to_width(
    text: str,
    font: Any,
    max_width: int,
    *,
    suffix: str,
) -> str:
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
    return int(max(12, bbox[3] - bbox[1] + 4))


def _text_width(text: str, font: Any) -> int:
    try:
        return int(round(font.getlength(text)))
    except AttributeError:
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0])


def _compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def _add_overlay(image: Image.Image, grid_frame: RecordingGridFrame) -> Image.Image:
    overlay_height = 40
    output = Image.new(
        "RGBA",
        (image.width, image.height + overlay_height),
        ARC_PALETTE[0],
    )
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, image.width, overlay_height), fill=(20, 20, 20, 255))

    font = ImageFont.load_default()
    line_1, line_2 = _metadata_lines(grid_frame)
    draw.text((4, 4), line_1, fill=(255, 255, 255, 255), font=font)
    draw.text((4, 20), line_2, fill=(255, 255, 255, 255), font=font)
    output.paste(image, (0, overlay_height))
    return output


def _metadata_lines(grid_frame: RecordingGridFrame) -> tuple[str, str]:
    data = grid_frame.event.data
    game_id = str(data.get("game_id", "?"))
    state = str(data.get("state", "?"))
    levels = f"{data.get('levels_completed', '?')}/{data.get('win_levels', '?')}"
    subframe = ""
    if grid_frame.grid_count > 1:
        subframe = f" grid={grid_frame.grid_index + 1}/{grid_frame.grid_count}"
    line_1 = (
        f"step={grid_frame.event.index:03d}{subframe} "
        f"game={game_id} state={state} levels={levels}"
    )

    action = grid_frame.event.action_label or _format_action(data.get("action_input"))
    available = _compact_json(data.get("available_actions", []), limit=48)
    line_2 = f"action={action} available={available}"
    return _truncate(line_1, 120), _truncate(line_2, 120)


def _format_action(value: Any) -> str:
    if not isinstance(value, dict):
        return "?"

    action_id = value.get("id", "?")
    action_name = _format_action_id(action_id)
    action_data = _compact_json(value.get("data", {}), limit=40)
    return f"{action_name} data={action_data}"


def _format_action_id(value: Any) -> str:
    if isinstance(value, int):
        return _ACTION_NAMES.get(value, str(value))
    if isinstance(value, str):
        if value.isdecimal():
            return _ACTION_NAMES.get(int(value), value)
        return value
    return str(value)


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _compact_json(value: Any, *, limit: int) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        text = str(value)
    return _truncate(text, limit)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


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
        canvas = Image.new("RGBA", (width, height), (20, 20, 20, 255))
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
        canvas = Image.new("RGBA", (even_width, even_height), (20, 20, 20, 255))
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
