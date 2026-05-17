from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

from PIL import Image, ImageDraw, ImageFont

RECORDING_SUFFIX = ".recording.jsonl"

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


JsonObject = dict[str, Any]
Grid = Sequence[Sequence[int]]


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
) -> Image.Image:
    """Render the first grid from one recording event."""
    grid_frame = expand_recording_frames([frame])[0]
    return render_recording_grid_frame(grid_frame, scale=scale, overlay=overlay)


def render_recording_grid_frame(
    grid_frame: RecordingGridFrame,
    *,
    scale: int = 8,
    overlay: bool = True,
) -> Image.Image:
    """Render one grid. Multi-grid recording events become sequential GIF frames."""
    image = grid_to_image(grid_frame.grid, scale=scale)
    if overlay:
        return _add_overlay(image, grid_frame)
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
) -> Path:
    """Export recording frames to an animated GIF."""
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if not frames:
        raise ValueError("At least one frame is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    images = [
        render_recording_grid_frame(grid_frame, scale=scale, overlay=overlay)
        for grid_frame in expand_recording_frames(frames)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an ARC-AGI recording JSONL file to an animated GIF."
    )
    parser.add_argument("recording", type=Path, help="Path to a .recording.jsonl file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to the input name with .gif.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    recording_path = cast(Path, args.recording)
    output_arg = cast(Path | None, args.output)
    output_path = (
        output_arg if output_arg is not None else default_output_path(recording_path)
    )
    fps = cast(int, args.fps)
    scale = cast(int, args.scale)
    overlay = not cast(bool, args.no_overlay)
    actions_log_arg = cast(Path | None, args.actions_log)

    frames = load_recording_frames(recording_path)
    actions_log = actions_log_arg or find_actions_log(recording_path)
    if actions_log is not None:
        action_labels = parse_action_log(actions_log)
        frames = apply_action_labels(frames, action_labels)
    else:
        action_labels = {}

    grid_frame_count = len(expand_recording_frames(frames))
    output = export_gif(
        frames,
        output_path,
        fps=fps,
        scale=scale,
        overlay=overlay,
    )
    print(
        f"Wrote {output} from {len(frames)} frame events "
        f"({grid_frame_count} rendered GIF frames)"
    )
    if actions_log is not None and action_labels:
        print(f"Loaded {len(action_labels)} action labels from {actions_log}")
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
