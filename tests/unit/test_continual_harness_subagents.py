from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.templates.continual_harness.subagents import (
    DEFAULT_SUBAGENT_ALLOWED_TOOLS,
    INSTRUCTIONS_MAX_CHARS,
    MAX_SUBAGENTS,
    SEARCH_MAX_MATCHES,
    SubagentEntry,
    SubagentStore,
    active_subagent_path,
    bootstrap_subagent_path,
    format_subagent_overview,
)


def _store(tmp_path: Path, game_id: str = "ls20") -> SubagentStore:
    return SubagentStore(tmp_path / "subagents.json", game_id=game_id)


def _add_basic(
    store: SubagentStore,
    *,
    name: str = "summarizer",
    description: str = "Summarize the last N steps.",
    instructions: str = "Read recent steps and call subagent_return with a digest.",
    allowed_tools: list[str] | None = None,
    tags: list[str] | None = None,
) -> SubagentEntry:
    return store.add(
        name=name,
        description=description,
        instructions=instructions,
        allowed_tools=allowed_tools
        if allowed_tools is not None
        else ["get_recent_trajectory"],
        tags=tags,
    )


@pytest.mark.unit
class TestBootstrapSubagentPath:
    def test_returns_none_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", raising=False)
        assert bootstrap_subagent_path() is None

    def test_returns_set_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "subagents.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", str(target))
        assert bootstrap_subagent_path() == target


@pytest.mark.unit
class TestActiveSubagentPath:
    def test_prefers_bootstrap_over_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bootstrap = tmp_path / "bootstrap.json"
        run_subagents = tmp_path / "run" / "subagents.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", str(bootstrap))
        monkeypatch.setenv("RUN_SUBAGENTS_PATH", str(run_subagents))
        assert active_subagent_path() == bootstrap

    def test_falls_through_to_run_subagents_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", raising=False)
        target = tmp_path / "run" / "subagents.json"
        monkeypatch.setenv("RUN_SUBAGENTS_PATH", str(target))
        assert active_subagent_path() == target

    def test_falls_through_to_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", raising=False)
        monkeypatch.delenv("RUN_SUBAGENTS_PATH", raising=False)
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        assert active_subagent_path() == run_dir / "subagents.json"

    def test_falls_through_to_run_log_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS", raising=False)
        monkeypatch.delenv("RUN_SUBAGENTS_PATH", raising=False)
        monkeypatch.delenv("RUN_DIR", raising=False)
        log_path = tmp_path / "logs" / "agent.log"
        monkeypatch.setenv("RUN_LOG_PATH", str(log_path))
        assert active_subagent_path() == log_path.with_name("subagents.json")


@pytest.mark.unit
class TestSubagentStoreCRUD:
    def test_store_loads_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.all_entries() == []

    def test_add_creates_entry_with_sequential_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        first = _add_basic(store, name="summarizer")
        second = _add_basic(store, name="planner")
        assert first.id == "subagent_001"
        assert second.id == "subagent_002"

    def test_add_stamps_game_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path, game_id="some-game")
        entry = _add_basic(store)
        assert entry.game_id == "some-game"

    def test_add_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "subagents.json"
        SubagentStore(path, game_id="g").add(
            name="summarizer",
            description="d",
            instructions="do thing",
            allowed_tools=["get_recent_trajectory"],
        )
        reopened = SubagentStore(path, game_id="g")
        entries = reopened.all_entries()
        assert len(entries) == 1
        assert entries[0].name == "summarizer"
        assert entries[0].allowed_tools == ["get_recent_trajectory"]

    def test_add_rejects_invalid_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add(name="", description="d", instructions="i", allowed_tools=[])
        with pytest.raises(ValueError):
            store.add(name="1bad", description="d", instructions="i", allowed_tools=[])
        with pytest.raises(ValueError):
            store.add(
                name="has space", description="d", instructions="i", allowed_tools=[]
            )

    def test_add_rejects_duplicate_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store, name="summarizer")
        with pytest.raises(ValueError, match="already in use"):
            _add_basic(store, name="summarizer")

    def test_add_rejects_empty_instructions(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add(name="s", description="d", instructions="", allowed_tools=[])

    def test_add_rejects_oversize_instructions(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        big = "x" * (INSTRUCTIONS_MAX_CHARS + 1)
        with pytest.raises(ValueError):
            store.add(name="s", description="d", instructions=big, allowed_tools=[])

    def test_add_rejects_when_full(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(MAX_SUBAGENTS):
            _add_basic(store, name=f"sub_n_{i}")
        with pytest.raises(ValueError, match="subagents full"):
            _add_basic(store, name="overflow")

    def test_delete_returns_false_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store)
        assert store.delete("subagent_999") is False

    def test_delete_does_not_reuse_ids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store, name="a")
        _add_basic(store, name="b")
        _add_basic(store, name="c")
        assert store.delete("subagent_002") is True
        new = _add_basic(store, name="d_sub")
        assert new.id == "subagent_004"

    def test_edit_updates_specific_fields_and_bumps_version(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        original = _add_basic(
            store,
            name="summarizer",
            description="old desc",
            instructions="old instructions",
            tags=["a"],
        )
        edited = store.edit(original.id, instructions="new instructions")
        assert edited is not None
        assert edited.name == "summarizer"  # unchanged
        assert edited.description == "old desc"  # unchanged
        assert edited.instructions == "new instructions"
        assert edited.tags == ["a"]
        assert edited.version == 2
        assert edited.updated_at >= original.updated_at

    def test_edit_can_change_allowed_tools(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(store, allowed_tools=["get_recent_trajectory"])
        edited = store.edit(entry.id, allowed_tools=["process_memory", "run_code"])
        assert edited is not None
        assert edited.allowed_tools == ["process_memory", "run_code"]

    def test_edit_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.edit("subagent_999", instructions="x") is None

    def test_edit_requires_at_least_one_field(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(store)
        with pytest.raises(ValueError):
            store.edit(entry.id)

    def test_edit_rejects_name_collision(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store, name="first")
        second = _add_basic(store, name="second")
        with pytest.raises(ValueError, match="already in use"):
            store.edit(second.id, name="first")


@pytest.mark.unit
class TestAllowedToolsValidation:
    def test_accepts_empty_list(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(store, allowed_tools=[])
        assert entry.allowed_tools == []

    def test_defaults_when_allowed_tools_omitted(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add(
            name="defaulted",
            description="Uses the default safe tool set.",
            instructions="Do the task and return.",
        )
        assert entry.allowed_tools == list(DEFAULT_SUBAGENT_ALLOWED_TOOLS)

    def test_accepts_full_enum(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(
            store,
            allowed_tools=[
                "get_recent_trajectory",
                "process_memory",
                "process_skill",
                "run_skill",
                "run_code",
            ],
        )
        assert set(entry.allowed_tools) == {
            "get_recent_trajectory",
            "process_memory",
            "process_skill",
            "run_skill",
            "run_code",
        }

    def test_rejects_unknown_tool_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="not in"):
            _add_basic(store, allowed_tools=["bogus_tool"])

    def test_rejects_run_subagent_recursion(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="not in"):
            _add_basic(store, allowed_tools=["run_subagent"])

    def test_rejects_process_subagent_recursion(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="not in"):
            _add_basic(store, allowed_tools=["process_subagent"])

    def test_deduplicates(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(
            store, allowed_tools=["run_code", "run_code", "process_memory"]
        )
        assert entry.allowed_tools == ["run_code", "process_memory"]

    def test_edit_rejects_unknown_tool(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = _add_basic(store)
        with pytest.raises(ValueError, match="not in"):
            store.edit(entry.id, allowed_tools=["bogus_tool"])


@pytest.mark.unit
class TestSubagentStoreSearch:
    def test_substring_over_name_description_instructions_tags(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        _add_basic(
            store,
            name="summarizer",
            description="Compact the trajectory.",
            instructions="Read trajectory; call subagent_return with a digest.",
            tags=["history"],
        )
        _add_basic(
            store,
            name="coord_proposer",
            description="Suggest x/y coordinates for ACTION6.",
            instructions="Inspect frame; propose 3 click points.",
            tags=["geo"],
        )

        name_hits, _ = store.search("summarizer")
        desc_hits, _ = store.search("ACTION6")
        instr_hits, _ = store.search("click points")
        tag_hits, _ = store.search("history")

        assert [e.name for e in name_hits] == ["summarizer"]
        assert [e.name for e in desc_hits] == ["coord_proposer"]
        assert [e.name for e in instr_hits] == ["coord_proposer"]
        assert [e.name for e in tag_hits] == ["summarizer"]

    def test_caps_returned_at_max_matches(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(15):
            _add_basic(
                store,
                name=f"common_{i}",
                description="shared description text",
                allowed_tools=[],
            )
        matches, total = store.search("shared")
        assert len(matches) == SEARCH_MAX_MATCHES
        assert total == 15

    def test_empty_query_returns_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store, name="a")
        _add_basic(store, name="b")
        matches, total = store.search("")
        assert {m.name for m in matches} == {"a", "b"}
        assert total == 2


@pytest.mark.unit
class TestFormatSubagentOverview:
    def test_empty_has_hint(self) -> None:
        out = format_subagent_overview([])
        assert "## SUBAGENTS (0 saved)" in out
        assert "No subagents saved yet" in out
        assert 'process_subagent(operation="add"' in out

    def test_lists_id_name_tools_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(
            store,
            name="summarizer",
            description="Compact the trajectory and save key insights.",
            instructions="SECRET_INSTRUCTION_TEXT do not leak in overview",
            allowed_tools=["get_recent_trajectory", "process_memory"],
        )
        out = format_subagent_overview(store.all_entries())
        assert "[subagent_001] summarizer" in out
        assert "[get_recent_trajectory, process_memory]" in out
        assert "Compact the trajectory" in out  # description IS shown
        assert "SECRET_INSTRUCTION_TEXT" not in out  # instructions never leak

    def test_shows_no_tools_marker(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add_basic(store, allowed_tools=[])
        out = format_subagent_overview(store.all_entries())
        assert "[no tools]" in out


@pytest.mark.unit
class TestStoreRobustness:
    def test_corrupt_json_self_heals(self, tmp_path: Path) -> None:
        path = tmp_path / "subagents.json"
        path.write_text("{this is not valid")
        store = SubagentStore(path, game_id="g")
        assert store.all_entries() == []
        entry = _add_basic(store)
        assert entry.id == "subagent_001"

    def test_load_advances_next_id_past_hand_edited_entries(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "subagents.json"
        path.write_text(
            json.dumps(
                {
                    "next_id": 1,
                    "entries": [
                        {
                            "id": "subagent_010",
                            "game_id": "g",
                            "name": "seed",
                            "description": "seeded sub",
                            "instructions": "do thing",
                            "allowed_tools": ["get_recent_trajectory"],
                            "tags": [],
                            "version": 3,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        store = SubagentStore(path, game_id="g")
        new = _add_basic(store, name="after_seed")
        assert new.id == "subagent_011"
