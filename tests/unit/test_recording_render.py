import json
from pathlib import Path

import pytest
from PIL import Image, ImageSequence

from agents.recording_render import (
    ARC_PALETTE,
    RecordingFrame,
    apply_action_labels,
    default_output_path,
    expand_recording_frames,
    export_gif,
    find_actions_log,
    grid_to_image,
    load_recording_frames,
    parse_action_log,
    render_recording_frame,
)


def write_jsonl(path: Path, events: list[object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event))
            file.write("\n")


def frame_event(frame: list[list[list[int]]]) -> dict[str, object]:
    return {
        "timestamp": "2026-05-17T00:00:00+00:00",
        "data": {
            "game_id": "test-game",
            "frame": frame,
            "state": "NOT_FINISHED",
            "levels_completed": 1,
            "win_levels": 3,
            "action_input": {"id": 2, "data": {}},
            "available_actions": [1, 2, 3, 4],
        },
    }


@pytest.mark.unit
def test_load_recording_frames_filters_non_frame_events(tmp_path: Path) -> None:
    recording = tmp_path / "test.recording.jsonl"
    write_jsonl(
        recording,
        [
            {"timestamp": "ignored", "data": {"tokens": 10}},
            frame_event([[[0, 1], [2, 3]]]),
        ],
    )

    frames = load_recording_frames(recording)

    assert len(frames) == 1
    assert frames[0].index == 0
    assert frames[0].timestamp == "2026-05-17T00:00:00+00:00"
    assert frames[0].data["game_id"] == "test-game"


@pytest.mark.unit
def test_export_gif_creates_expected_frame_count(tmp_path: Path) -> None:
    recording = tmp_path / "test.recording.jsonl"
    output = tmp_path / "out.gif"
    write_jsonl(
        recording,
        [
            frame_event([[[0, 0], [0, 0]]]),
            frame_event([[[8, 8], [8, 8]]]),
        ],
    )

    exported = export_gif(
        load_recording_frames(recording),
        output,
        fps=2,
        scale=2,
        overlay=False,
    )

    with Image.open(exported) as gif:
        assert gif.format == "GIF"
        assert len(list(ImageSequence.Iterator(gif))) == 2
        assert gif.info["duration"] == 500


@pytest.mark.unit
def test_grid_to_image_preserves_palette_with_nearest_scaling() -> None:
    image = grid_to_image([[0, 8], [14, 15]], scale=3)

    assert image.size == (6, 6)
    assert image.getpixel((0, 0)) == ARC_PALETTE[0]
    assert image.getpixel((3, 0)) == ARC_PALETTE[8]
    assert image.getpixel((5, 5)) == ARC_PALETTE[15]


@pytest.mark.unit
def test_multi_grid_event_expands_into_sequential_frames() -> None:
    recording_frame = RecordingFrame(
        index=0,
        timestamp=None,
        data={
            "frame": [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
            ],
        },
    )

    grid_frames = expand_recording_frames([recording_frame])
    image = render_recording_frame(recording_frame, scale=4, overlay=False)

    assert len(grid_frames) == 2
    assert grid_frames[0].grid_index == 0
    assert grid_frames[0].grid_count == 2
    assert grid_frames[1].grid_index == 1
    assert image.size == (8, 8)


@pytest.mark.unit
def test_multi_grid_event_exports_as_sequential_gif_frames(tmp_path: Path) -> None:
    recording = tmp_path / "test.recording.jsonl"
    output = tmp_path / "out.gif"
    write_jsonl(
        recording,
        [
            frame_event(
                [
                    [[0, 0], [0, 0]],
                    [[8, 8], [8, 8]],
                ]
            ),
        ],
    )

    exported = export_gif(
        load_recording_frames(recording),
        output,
        fps=2,
        scale=2,
        overlay=False,
    )

    with Image.open(exported) as gif:
        gif_frames = list(ImageSequence.Iterator(gif))
        assert len(gif_frames) == 2
        assert gif_frames[0].size == (4, 4)
        assert gif_frames[1].size == (4, 4)


@pytest.mark.unit
def test_frame_less_recording_raises_clear_error(tmp_path: Path) -> None:
    recording = tmp_path / "empty.recording.jsonl"
    write_jsonl(recording, [{"timestamp": "ignored", "data": {"tokens": 10}}])

    with pytest.raises(ValueError, match="No frame events found"):
        load_recording_frames(recording)


@pytest.mark.unit
def test_malformed_jsonl_raises_clear_error(tmp_path: Path) -> None:
    recording = tmp_path / "bad.recording.jsonl"
    recording.write_text('{"timestamp": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_recording_frames(recording)


@pytest.mark.unit
def test_default_output_path_removes_recording_suffix() -> None:
    output = default_output_path("recordings/test.agent.guid.recording.jsonl")

    assert output == Path("recordings/test.agent.guid.gif")


@pytest.mark.unit
def test_parse_action_log_reads_labels_by_action_count(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-17 | INFO | test-game - ACTION2: count 0, levels completed 0, avg fps 0.0)",
                "2026-05-17 | INFO | test-game - ACTION4: count 1, levels completed 0, avg fps 0.0)",
            ]
        ),
        encoding="utf-8",
    )

    labels = parse_action_log(log)

    assert labels == {0: "ACTION2", 1: "ACTION4"}


@pytest.mark.unit
def test_apply_action_labels_overrides_reset_metadata() -> None:
    frame = RecordingFrame(
        index=0,
        timestamp=None,
        data={
            "frame": [[[0]]],
            "action_input": {"id": 0, "data": {}},
        },
    )

    labeled = apply_action_labels([frame], {0: "ACTION2"})

    assert labeled[0].action_label == "ACTION2"
    assert frame.action_label is None


@pytest.mark.unit
def test_find_actions_log_matches_recording_name(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    recording = tmp_path / "recordings" / "test.agent.guid.recording.jsonl"
    log = logs_dir / "run.log"
    log.write_text(
        "created new recording for test into "
        "recordings/test.agent.guid.recording.jsonl\n",
        encoding="utf-8",
    )

    assert find_actions_log(recording, logs_dir=logs_dir) == log
