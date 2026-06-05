from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.templates.continual_harness.skills import (
    CODE_MAX_CHARS,
    MAX_SKILLS,
    SEARCH_MAX_MATCHES,
    SkillStore,
    active_skill_path,
    format_skill_overview,
)


def _store(tmp_path: Path, game_id: str = "ls20") -> SkillStore:
    return SkillStore(tmp_path / "skills.json", game_id=game_id)


@pytest.mark.unit
class TestActiveSkillPath:
    def test_per_game_under_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))

        assert active_skill_path("ls20") == run_dir / "ls20" / "skills.json"
        assert active_skill_path("ls20") != active_skill_path("vc33")

    def test_falls_back_to_run_dir_without_game_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        monkeypatch.delenv("RUN_LOG_PATH", raising=False)
        assert active_skill_path() == run_dir / "skills.json"

    def test_falls_back_to_run_log_sibling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("RUN_DIR", raising=False)
        monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "run.log"))
        assert active_skill_path("ls20") == tmp_path / "skills.json"


@pytest.mark.unit
class TestSkillStoreCRUD:
    def test_store_loads_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.all_entries() == []

    def test_add_creates_entry_with_sequential_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        first = store.add("find_player", "Find the player cell.", "result = 1")
        second = store.add("count_walls", "Count walls.", "result = 2")
        assert first.id == "skill_001"
        assert second.id == "skill_002"

    def test_add_stamps_game_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path, game_id="some-game")
        entry = store.add("find_player", "desc", "result = 1")
        assert entry.game_id == "some-game"

    def test_add_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "skills.json"
        SkillStore(path, game_id="g").add("find_player", "desc", "result = 1")
        reopened = SkillStore(path, game_id="g")
        entries = reopened.all_entries()
        assert len(entries) == 1
        assert entries[0].name == "find_player"

    def test_add_rejects_invalid_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add("", "desc", "result = 1")
        with pytest.raises(ValueError):
            store.add("1bad_start", "desc", "result = 1")
        with pytest.raises(ValueError):
            store.add("has space", "desc", "result = 1")

    def test_add_with_existing_name_upserts(self, tmp_path: Path) -> None:
        """Same-name `add` updates the existing entry in place."""
        store = _store(tmp_path)
        first = store.add("find_player", "desc1", "result = 1")
        second = store.add(
            "find_player", "desc2", "result = 2", tags=["geo"]
        )
        # Same id; bumped version; fresh code/description/tags.
        assert second.id == first.id
        assert second.version == first.version + 1
        assert second.description == "desc2"
        assert second.code == "result = 2"
        assert second.tags == ["geo"]
        # Store still has only one entry.
        assert len(store.all_entries()) == 1

    def test_add_rejects_empty_code(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add("find_player", "desc", "")

    def test_add_rejects_oversize_code(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        big = "x = 1\n" * (CODE_MAX_CHARS // 5)
        assert len(big) > CODE_MAX_CHARS
        with pytest.raises(ValueError):
            store.add("find_player", "desc", big)

    def test_add_rejects_when_full(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(MAX_SKILLS):
            store.add(f"skill_n_{i}", "desc", "result = 1")
        with pytest.raises(ValueError, match="skills full"):
            store.add("overflow", "desc", "result = 1")

    def test_delete_returns_false_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("find_player", "desc", "result = 1")
        assert store.delete("skill_999") is False

    def test_delete_does_not_reuse_ids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("a", "d", "result = 1")
        store.add("b", "d", "result = 2")
        store.add("c", "d", "result = 3")
        assert store.delete("skill_002") is True
        new = store.add("d_skill", "d", "result = 4")
        assert new.id == "skill_004"

    def test_edit_updates_specific_fields_and_bumps_version(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        original = store.add("find_player", "old desc", "result = 1", tags=["a"])
        edited = store.edit(original.id, code="result = 99")
        assert edited is not None
        assert edited.name == "find_player"  # unchanged
        assert edited.description == "old desc"  # unchanged
        assert edited.code == "result = 99"
        assert edited.tags == ["a"]
        assert edited.version == 2
        assert edited.updated_at >= original.updated_at

    def test_edit_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.edit("skill_999", code="result = 1") is None

    def test_edit_requires_at_least_one_field(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("find_player", "d", "result = 1")
        with pytest.raises(ValueError):
            store.edit(entry.id)

    def test_edit_rejects_name_collision_with_another_entry(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        store.add("first", "d", "result = 1")
        second = store.add("second", "d", "result = 2")
        with pytest.raises(ValueError, match="already in use"):
            store.edit(second.id, name="first")


@pytest.mark.unit
class TestSkillStoreSearch:
    def test_substring_over_name_description_code_tags(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        store.add("find_player", "Locate the player cell.", "result = 1", tags=["geo"])
        store.add("count_walls", "Count color-3 cells.", "WALL = 3\nresult = 0")
        store.add("util_misc", "Misc helpers.", "result = None", tags=["mISC"])

        name_hits, _ = store.search("find_player")
        desc_hits, _ = store.search("player cell")
        code_hits, _ = store.search("WALL = 3")
        tag_hits, _ = store.search("misc")

        assert [e.name for e in name_hits] == ["find_player"]
        assert [e.name for e in desc_hits] == ["find_player"]
        assert [e.name for e in code_hits] == ["count_walls"]
        assert [e.name for e in tag_hits] == ["util_misc"]

    def test_caps_returned_at_max_matches(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(15):
            store.add(f"common_{i}", "shared description", "result = 1")
        matches, total = store.search("shared")
        assert len(matches) == SEARCH_MAX_MATCHES
        assert total == 15

    def test_empty_query_returns_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("a", "d", "result = 1")
        store.add("b", "d", "result = 2")
        matches, total = store.search("")
        assert {m.name for m in matches} == {"a", "b"}
        assert total == 2


@pytest.mark.unit
class TestFormatSkillOverview:
    def test_empty_has_hint(self) -> None:
        out = format_skill_overview([])
        assert "## SKILLS (0 saved)" in out
        assert "No skills saved yet" in out
        assert 'process_skill(operation="add"' in out

    def test_lists_id_name_tags_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(
            "find_player",
            "Locate the player cell using bounding box scan.",
            "SECRET_CODE_PATTERN = 'do not leak'",
            tags=["geo", "analysis"],
        )
        out = format_skill_overview(store.all_entries())
        # The overview labels both id and name explicitly so the model
        # cannot conflate them when calling run_skill / process_skill.
        assert "id=skill_001" in out
        assert "name=find_player" in out
        assert "tags=geo,analysis" in out
        assert "Locate the player cell" in out  # first line of description IS shown
        assert "SECRET_CODE_PATTERN" not in out  # code never leaks into overview


@pytest.mark.unit
class TestStoreRobustness:
    def test_corrupt_json_self_heals(self, tmp_path: Path) -> None:
        path = tmp_path / "skills.json"
        path.write_text("{this is not valid")
        store = SkillStore(path, game_id="g")
        assert store.all_entries() == []
        entry = store.add("find_player", "d", "result = 1")
        assert entry.id == "skill_001"

    def test_load_advances_next_id_past_hand_edited_entries(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "skills.json"
        path.write_text(
            json.dumps(
                {
                    "next_id": 1,
                    "entries": [
                        {
                            "id": "skill_010",
                            "game_id": "g",
                            "name": "seed",
                            "description": "seeded skill",
                            "code": "result = 1",
                            "tags": [],
                            "version": 3,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        store = SkillStore(path, game_id="g")
        new = store.add("after_seed", "d", "result = 2")
        assert new.id == "skill_011"


@pytest.mark.unit
class TestGetByIdOrName:
    """Covers the lookup-side fix for the model confusing skill id and name."""

    def test_returns_entry_by_canonical_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("eval_python", "Run arithmetic", "result = 1+1")
        entry = store.get_by_id_or_name("skill_001")
        assert entry is not None
        assert entry.id == "skill_001"
        assert entry.name == "eval_python"

    def test_returns_entry_by_exact_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("eval_python", "Run arithmetic", "result = 1+1")
        entry = store.get_by_id_or_name("eval_python")
        assert entry is not None
        assert entry.id == "skill_001"

    def test_name_lookup_is_case_insensitive(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("Eval_Python", "Run arithmetic", "result = 1+1")
        assert store.get_by_id_or_name("EVAL_PYTHON") is not None
        assert store.get_by_id_or_name("eval_python") is not None

    def test_unknown_key_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("eval_python", "Run arithmetic", "result = 1+1")
        assert store.get_by_id_or_name("missing") is None
        assert store.get_by_id_or_name("") is None

    def test_id_match_wins_over_name_match(self, tmp_path: Path) -> None:
        # Pathological but possible after hand-editing: an entry's name
        # collides with another entry's id. The id match should win.
        path = tmp_path / "skills.json"
        path.write_text(
            json.dumps(
                {
                    "next_id": 3,
                    "entries": [
                        {
                            "id": "skill_001",
                            "game_id": "g",
                            "name": "alpha",
                            "description": "d",
                            "code": "result = 0",
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        },
                        {
                            "id": "skill_002",
                            "game_id": "g",
                            "name": "skill_001",  # collision
                            "description": "d",
                            "code": "result = 0",
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        },
                    ],
                }
            )
        )
        store = SkillStore(path, game_id="g")
        hit = store.get_by_id_or_name("skill_001")
        assert hit is not None
        assert hit.id == "skill_001"  # id match, not the name collision
