from __future__ import annotations

import asyncio
import json

import pytest
from arcengine import GameAction, GameState

from cli_agents.server import app as server


class _Request:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


class _Layer:
    def __init__(self, grid: list[list[int]]) -> None:
        self._grid = grid

    def tolist(self) -> list[list[int]]:
        return self._grid


class _RawFrame:
    def __init__(
        self,
        *,
        state: GameState = GameState.NOT_FINISHED,
        levels_completed: int = 0,
        full_reset: bool = False,
    ) -> None:
        self.game_id = "test-game"
        self.frame = [_Layer([[1, 1], [1, 0]])]
        self.state = state
        self.levels_completed = levels_completed
        self.win_levels = 7
        self.available_actions = [1, 2, 3, 4, 6]
        self.guid = "test-guid"
        self.full_reset = full_reset


class _FakeEnv:
    def __init__(self, responses: list[_RawFrame] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[GameAction, dict, dict]] = []

    def step(
        self,
        action: GameAction,
        data: dict | None = None,
        reasoning: dict | None = None,
    ) -> _RawFrame:
        self.calls.append((action, data or {}, reasoning or {}))
        if not self.responses:
            raise AssertionError(f"unexpected env.step({action.name})")
        return self.responses.pop(0)


def _cached_frame(state: str = GameState.NOT_FINISHED.name) -> dict:
    return {
        "game_id": "test-game",
        "frame": [[[1, 1], [1, 0]]],
        "state": state,
        "levels_completed": 0,
        "win_levels": 7,
        "available_actions": [1, 2, 3, 4, 6],
        "guid": "test-guid",
        "full_reset": False,
    }


def _decode_response(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    server._arc = None
    server._card_id = None
    server._env = None
    server._latest = None
    server._budget = 0
    server._win_levels = None
    server._game_id = "test-game"
    server._tags = []
    server._run_dir = None
    server._recordings_dir = None
    server._observations_dir = None
    server._state_file = None
    server._scorecard_closed = False
    server._scorecard_closing = False
    server._scorecard_payload = None
    server._pending_obs = []
    server._obs_window = []
    server._batch_counter = 0
    server._action_counter = 0
    server._observe_counter = 0


def test_take_actions_rejects_reset_without_stepping_env() -> None:
    env = _FakeEnv()
    server._env = env
    server._latest = _cached_frame()
    server._budget = 10

    response = asyncio.run(
        server.mcp_take_actions(
            _Request({"actions": [{"name": "RESET", "reasoning": "start over"}]})
        )
    )

    payload = _decode_response(response)
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["applied_count"] == 0
    assert payload["applied_actions"] == []
    assert payload["rejected"][0]["raw"]["name"] == "RESET"
    assert "managed by the server/harness" in payload["rejected"][0]["reason"]
    assert env.calls == []


def test_get_game_state_auto_resets_before_first_playable_observation() -> None:
    env = _FakeEnv([
        _RawFrame(state=GameState.NOT_FINISHED, full_reset=True),
    ])
    server._env = env
    server._latest = _cached_frame(GameState.NOT_PLAYED.name)
    server._budget = 10

    response = asyncio.run(server.mcp_get_game_state())

    payload = _decode_response(response)
    assert response.status_code == 200
    assert payload["state"] == GameState.NOT_FINISHED.name
    assert payload["auto_reset"] is True
    assert "RESET" not in payload["available_actions"]
    assert server._budget == 9
    assert [call[0] for call in env.calls] == [GameAction.RESET]


def test_take_actions_auto_resets_initial_state_without_applying_stale_batch() -> None:
    env = _FakeEnv([
        _RawFrame(state=GameState.NOT_FINISHED, full_reset=True),
    ])
    server._env = env
    server._latest = _cached_frame(GameState.NOT_PLAYED.name)
    server._budget = 10

    response = asyncio.run(
        server.mcp_take_actions(
            _Request({"actions": [{"name": "ACTION1", "reasoning": "move up"}]})
        )
    )

    payload = _decode_response(response)
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["auto_reset"] is True
    assert payload["applied_count"] == 0
    assert payload["state"] == GameState.NOT_FINISHED.name
    assert server._budget == 9
    assert [call[0] for call in env.calls] == [GameAction.RESET]
