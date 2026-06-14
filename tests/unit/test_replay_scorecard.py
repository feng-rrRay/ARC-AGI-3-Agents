import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arc_agi
import pytest
from arcengine import GameAction

from cli_agents.server import replay_scorecard


class FakeRaw:
    def __init__(self, levels_completed: int = 0, state: str = "NOT_FINISHED") -> None:
        self.levels_completed = levels_completed
        self.state = SimpleNamespace(name=state)


class FakeEnv:
    def __init__(self) -> None:
        self.observation_space = FakeRaw()
        self.steps: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    def reset(self) -> FakeRaw:
        return FakeRaw()

    def step(
        self, action: Any, data: dict[str, Any], reasoning: dict[str, Any]
    ) -> FakeRaw:
        self.steps.append((action, data, reasoning))
        return FakeRaw()


class FakeScorecard:
    def __init__(self, card_id: str) -> None:
        self.card_id = card_id

    def model_dump(self) -> dict[str, Any]:
        return {"card_id": self.card_id, "score": 12.5}


class FakeArcade:
    instances: list["FakeArcade"] = []

    def __init__(self) -> None:
        self.operation_mode = arc_agi.OperationMode.ONLINE
        self.open_tags: list[list[str]] = []
        self.make_calls: list[tuple[str, str, bool]] = []
        self.closed_cards: list[str] = []
        self.envs: list[FakeEnv] = []
        FakeArcade.instances.append(self)

    def open_scorecard(self, tags: list[str]) -> str:
        self.open_tags.append(tags)
        return f"card-{len(self.open_tags)}"

    def make(
        self, game_id: str, scorecard_id: str, save_recording: bool = False
    ) -> FakeEnv:
        self.make_calls.append((game_id, scorecard_id, save_recording))
        env = FakeEnv()
        self.envs.append(env)
        return env

    def close_scorecard(self, card_id: str) -> FakeScorecard:
        self.closed_cards.append(card_id)
        return FakeScorecard(card_id)


def _write_run(
    root: Path,
    name: str,
    game_id: str,
    action_inputs: list[dict[str, Any]],
) -> Path:
    run_dir = root / name
    recordings_dir = run_dir / "recordings"
    recordings_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"card_id": f"source-{game_id}"}),
        encoding="utf-8",
    )
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {
                "game_id": game_id,
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "action_input": {"id": 0, "data": {}},
            },
        }
    ]
    for action_input in action_inputs:
        rows.append(
            {
                "timestamp": "2026-01-01T00:00:01+00:00",
                "data": {
                    "game_id": game_id,
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "action_input": action_input,
                },
            }
        )
    rec_path = recordings_dir / f"{game_id}.recording.jsonl"
    rec_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


@pytest.mark.unit
@pytest.mark.parametrize(
    ("action_id", "expected"),
    [
        (6, GameAction.ACTION6),
        ("6", GameAction.ACTION6),
        ("ACTION6", GameAction.ACTION6),
        ("action6", GameAction.ACTION6),
        (0, GameAction.RESET),
        ("RESET", GameAction.RESET),
    ],
)
def test_decode_action_input_accepts_legacy_and_named_ids(
    action_id: int | str,
    expected: GameAction,
) -> None:
    action, data = replay_scorecard._decode_action_input(
        {"id": action_id, "data": {"x": 3, "y": 4, "ignored": 99}}
    )

    assert action == expected
    assert data == {"x": 3, "y": 4}


@pytest.mark.unit
def test_decode_action_input_rejects_boolean_id() -> None:
    with pytest.raises(ValueError, match="boolean action id"):
        replay_scorecard._decode_action_input({"id": True, "data": {}})


@pytest.mark.unit
def test_resolve_operation_mode_honors_uppercase_competition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPERATION_MODE", "COMPETITION")

    assert replay_scorecard._resolve_operation_mode() == "competition"


@pytest.mark.unit
def test_resolve_operation_mode_defaults_to_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPERATION_MODE", raising=False)

    assert replay_scorecard._resolve_operation_mode() == "online"


@pytest.mark.unit
def test_resolve_operation_mode_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPERATION_MODE", "invalid")

    with pytest.raises(SystemExit, match="OPERATION_MODE must be one of"):
        replay_scorecard._resolve_operation_mode()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["offline", "normal"])
def test_resolve_operation_mode_rejects_non_api_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("OPERATION_MODE", mode)

    with pytest.raises(SystemExit, match="competition, online"):
        replay_scorecard._resolve_operation_mode()


@pytest.mark.unit
def test_scorecard_payload_includes_url_in_competition_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_BASE_URL", "https://three.arcprize.org")
    arc = SimpleNamespace(operation_mode=arc_agi.OperationMode.COMPETITION)

    payload = replay_scorecard._scorecard_payload(
        arc,
        FakeScorecard("card-competition"),
        "card-competition",
        "https://fallback.example",
        {},
    )

    assert (
        payload["scorecard_url"]
        == "https://three.arcprize.org/scorecards/card-competition"
    )


@pytest.mark.unit
def test_replay_one_decodes_numeric_actions_and_writes_scorecard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeArcade.instances = []
    monkeypatch.setattr(arc_agi, "Arcade", FakeArcade)
    monkeypatch.setenv("ARC_BASE_URL", "https://three.arcprize.org")
    run_dir = _write_run(
        tmp_path,
        "run-one",
        "ka59-38d34dbb",
        [
            {"id": 6, "data": {"x": 7, "y": 8}},
            {"id": "ACTION1", "data": {}},
        ],
    )

    ok = replay_scorecard.replay_one(
        run_dir,
        "https://three.arcprize.org",
        force=True,
    )

    assert ok is True
    arcade = FakeArcade.instances[0]
    assert arcade.open_tags == [["agent", "ContinualHarness"]]
    assert arcade.make_calls == [("ka59-38d34dbb", "card-1", False)]
    assert arcade.closed_cards == ["card-1"]
    assert [(action.name, data) for action, data, _ in arcade.envs[0].steps] == [
        ("ACTION6", {"x": 7, "y": 8}),
        ("ACTION1", {}),
    ]

    payload = json.loads((run_dir / "scorecard.json").read_text())
    assert payload["scorecard_url"] == "https://three.arcprize.org/scorecards/card-1"
    assert payload["replayed_from_card_id"] == "source-ka59-38d34dbb"
    assert payload["replay_result"]["applied"] == 2
    assert payload["replay_result"]["success"] is True


@pytest.mark.unit
def test_single_scorecard_replays_multiple_runs_into_one_scorecard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeArcade.instances = []
    monkeypatch.setattr(arc_agi, "Arcade", FakeArcade)
    monkeypatch.setenv("ARC_BASE_URL", "https://three.arcprize.org")
    run_one = _write_run(
        tmp_path,
        "run-one",
        "ka59-38d34dbb",
        [{"id": 1, "data": {}}],
    )
    run_two = _write_run(
        tmp_path,
        "run-two",
        "ls20-9607627b",
        [{"id": "ACTION2", "data": {}}],
    )
    output_path = tmp_path / "scorecard.online.json"

    ok = replay_scorecard.replay_single_scorecard(
        [run_one, run_two],
        "https://three.arcprize.org",
        output_path,
        force=True,
    )

    assert ok is True
    arcade = FakeArcade.instances[0]
    assert arcade.open_tags == [["agent", "ContinualHarness"]]
    assert arcade.make_calls == [
        ("ka59-38d34dbb", "card-1", False),
        ("ls20-9607627b", "card-1", False),
    ]
    assert arcade.closed_cards == ["card-1"]

    payload = json.loads(output_path.read_text())
    assert payload["scorecard_url"] == "https://three.arcprize.org/scorecards/card-1"
    assert payload["replay_source_count"] == 2
    assert [result["game_id"] for result in payload["replay_results"]] == [
        "ka59-38d34dbb",
        "ls20-9607627b",
    ]
