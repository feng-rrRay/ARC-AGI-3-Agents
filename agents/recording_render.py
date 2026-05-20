from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

from PIL import Image, ImageDraw, ImageFont

RECORDING_SUFFIX = ".recording.jsonl"
TRACE_SUFFIX = ".trace.jsonl"

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
_ACTION_LOG_RE = re.compile(
    r"\|\s+INFO\s+\|\s+.+?\s+-\s+(?P<action>RESET|ACTION[1-7]): "
    r"count (?P<count>\d+),"
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
    """Parse action labels from a run log keyed by Agent action count."""
    log_path = Path(path)
    labels: dict[int, str] = {}

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = _ACTION_LOG_RE.search(line)
            if match is None:
                continue
            labels[int(match.group("count"))] = match.group("action")

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
    artifact_trace = _artifact_trace_for_recording(recording)
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


def parse_trace_log(path: str | Path) -> dict[int, str]:
    """Parse per-action VLM reasoning from a continual-harness trace JSONL file."""
    trace_path = Path(path)
    accumulators: dict[int, JsonObject] = {}

    with trace_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            event = _load_json_object(stripped, trace_path, line_number)
            action_counter = _int_or_none(event.get("action_counter"))
            if action_counter is None:
                continue

            accumulator = accumulators.setdefault(
                action_counter,
                {"analysis_calls": [], "errors": []},
            )
            _merge_trace_event(accumulator, event)

    traces: dict[int, str] = {}
    for action_counter, accumulator in sorted(accumulators.items()):
        trace_text = _format_trace_text(action_counter, accumulator)
        if trace_text is not None:
            traces[action_counter] = trace_text
    return traces


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
    if not frames:
        raise ValueError("At least one frame is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

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
    images = _pad_to_common_size(images)
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


def default_output_path(path: str | Path) -> Path:
    recording_path = Path(path)
    if recording_path.name.endswith(RECORDING_SUFFIX):
        stem = recording_path.name[: -len(RECORDING_SUFFIX)]
        return recording_path.with_name(f"{stem}.gif")
    return recording_path.with_suffix(".gif")


def output_path_for_recording(
    recording_path: str | Path,
    output_arg: str | Path | None,
    *,
    recording_count: int = 1,
) -> Path:
    """Resolve the GIF path for one recording in file-mode or directory-mode."""
    recording = Path(recording_path)
    if output_arg is None:
        return default_output_path(recording)

    output = Path(output_arg)
    if recording_count == 1 and output.suffix.lower() == ".gif":
        return output
    if recording_count > 1 and output.suffix.lower() == ".gif":
        raise ValueError(
            "--output must be a directory when rendering multiple recordings"
        )
    return output / default_output_path(recording).name


def render_recording_file(
    recording_path: str | Path,
    output_path: str | Path,
    *,
    fps: int = 5,
    scale: int = 8,
    overlay: bool = True,
    actions_log: str | Path | None = None,
    trace_log: str | Path | None = None,
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
    if reasoning:
        resolved_trace_log = Path(trace_log) if trace_log is not None else None
        resolved_trace_log = resolved_trace_log or find_trace_log(
            recording,
            actions_log=resolved_actions_log,
        )
        if resolved_trace_log is not None:
            reasoning_traces = parse_trace_log(resolved_trace_log)
            frames = apply_reasoning_traces(frames, reasoning_traces)
        else:
            reasoning_traces = {}
    else:
        reasoning_traces = {}

    attached_reasoning_count = sum(1 for frame in frames if frame.reasoning_trace)
    grid_frame_count = len(expand_recording_frames(frames))
    output = export_gif(
        frames,
        output_path,
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
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render ARC-AGI recording JSONL files to animated GIFs. "
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
            "Output GIF path for one recording, or output directory for a run folder. "
            "Defaults to writing each GIF beside its recording."
        ),
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
    fps = cast(int, args.fps)
    scale = cast(int, args.scale)
    overlay = not cast(bool, args.no_overlay)
    actions_log_arg = cast(Path | None, args.actions_log)
    trace_log_arg = cast(Path | None, args.trace_log)
    reasoning = not cast(bool, args.no_reasoning)

    recordings = discover_recording_paths(input_path)
    summaries: list[RenderSummary] = []
    for recording_path in recordings:
        output_path = output_path_for_recording(
            recording_path,
            output_arg,
            recording_count=len(recordings),
        )
        summary = render_recording_file(
            recording_path,
            output_path,
            fps=fps,
            scale=scale,
            overlay=overlay,
            actions_log=actions_log_arg,
            trace_log=trace_log_arg,
            reasoning=reasoning,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary.output} from {summary.frame_events} frame events "
            f"({summary.grid_frames} rendered GIF frames)"
        )
        if summary.actions_log is not None and summary.action_label_count:
            print(
                f"Loaded {summary.action_label_count} action labels "
                f"from {summary.actions_log}"
            )
        if summary.trace_log is not None and summary.reasoning_trace_count:
            print(
                f"Loaded {summary.reasoning_trace_count} reasoning traces "
                f"from {summary.trace_log} "
                f"({summary.attached_reasoning_count} attached to recording frames)"
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


def _artifact_trace_for_recording(recording_path: Path) -> Path | None:
    run_dir = _run_dir_for_recording(recording_path)
    if run_dir is None:
        return None
    stem = recording_path.name
    if stem.endswith(RECORDING_SUFFIX):
        stem = stem[: -len(RECORDING_SUFFIX)]
    trace_path = run_dir / "artifacts" / f"{stem}{TRACE_SUFFIX}"
    return trace_path if trace_path.exists() else None


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


def _merge_trace_event(accumulator: JsonObject, event: JsonObject) -> None:
    chosen_action = _clean_string(event.get("chosen_action"))
    round_value = _int_or_none(event.get("round"))
    reasoning = _clean_string(event.get("reasoning"))

    if chosen_action is not None:
        accumulator["chosen_action"] = chosen_action
        if round_value is not None:
            accumulator["round"] = round_value
        if reasoning is None:
            reasoning = _reasoning_from_output_action(event, chosen_action)
        if reasoning is not None:
            accumulator["reasoning"] = reasoning
    elif reasoning is not None and accumulator.get("reasoning") is None:
        accumulator["reasoning"] = reasoning

    for summary in _analysis_call_summaries(event):
        _append_unique(accumulator, "analysis_calls", summary)

    error = _clean_string(event.get("error"))
    if error is not None:
        _append_unique(
            accumulator, "errors", _truncate(_compact_whitespace(error), 220)
        )


def _analysis_call_summaries(event: JsonObject) -> list[str]:
    summaries: list[str] = []
    for call in _output_function_calls(event):
        name = _clean_string(call.get("name"))
        if name is None or _is_action_name(name):
            continue

        args = call.get("args")
        reasoning = args.get("reasoning") if isinstance(args, dict) else None
        reason_text = _clean_string(reasoning)
        if reason_text is None:
            summaries.append(name)
        else:
            summaries.append(f"{name}: {_compact_whitespace(reason_text)}")

    tool_calls = event.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = _clean_string(
                call.get("name") or call.get("tool") or call.get("function")
            )
            if name is None or _is_action_name(name):
                continue

            result = _clean_string(
                call.get("result") or call.get("output") or call.get("content")
            )
            if result is None:
                summaries.append(name)
            else:
                summaries.append(f"{name} -> {_compact_whitespace(result)}")
    return [_truncate(summary, 260) for summary in summaries]


def _format_trace_text(action_counter: int, accumulator: JsonObject) -> str | None:
    action = _clean_string(accumulator.get("chosen_action"))
    round_value = _int_or_none(accumulator.get("round"))
    reasoning = _clean_string(accumulator.get("reasoning"))
    analysis_calls = _string_list(accumulator.get("analysis_calls"))
    errors = _string_list(accumulator.get("errors"))

    if action is None and reasoning is None and not analysis_calls and not errors:
        return None

    details = [f"step={action_counter:03d}"]
    if action is not None:
        details.append(f"action={action}")
    if round_value is not None:
        details.append(f"round={round_value}")

    lines = ["VLM reasoning", " ".join(details)]
    if reasoning is not None:
        lines.extend(("", reasoning))
    if analysis_calls:
        lines.extend(("", "Analysis tools:"))
        lines.extend(f"- {call}" for call in analysis_calls)
    if errors:
        lines.extend(("", "Errors:"))
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def _reasoning_from_output_action(event: JsonObject, action_name: str) -> str | None:
    for call in _output_function_calls(event):
        if _clean_string(call.get("name")) != action_name:
            continue
        args = call.get("args")
        if isinstance(args, dict):
            return _clean_string(args.get("reasoning"))
    return None


def _output_function_calls(event: JsonObject) -> list[JsonObject]:
    output = event.get("output")
    if not isinstance(output, dict):
        return []
    function_calls = output.get("function_calls")
    if not isinstance(function_calls, list):
        return []
    return [cast(JsonObject, call) for call in function_calls if isinstance(call, dict)]


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _append_unique(accumulator: JsonObject, key: str, value: str) -> None:
    values = accumulator.setdefault(key, [])
    if not isinstance(values, list):
        return
    if value not in values:
        values.append(value)


def _is_action_name(name: str) -> bool:
    return name in _ACTION_NAMES.values()


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
        if line_index == 0 and line == "VLM reasoning":
            fill = (255, 255, 255, 255)
        elif line.startswith("step=") or line in {"Analysis tools:", "Errors:"}:
            fill = (188, 207, 255, 255)
        else:
            fill = (226, 232, 240, 255)
        draw.text((x, y + line_index * line_height), line, fill=fill, font=font)


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


if __name__ == "__main__":
    raise SystemExit(main())
