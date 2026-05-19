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
    bootstrap_skill_path,
    format_skill_overview,
)


def _store(tmp_path: Path, game_id: str = "ls20") -> SkillStore:
    return SkillStore(tmp_path / "skills.json", game_id=game_id)


@pytest.mark.unit
class TestBootstrapSkillPath:
    def test_returns_none_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", raising=False)
        assert bootstrap_skill_path() is None

    def test_returns_set_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "skills.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", str(target))
        assert bootstrap_skill_path() == target


@pytest.mark.unit
class TestActiveSkillPath:
    def test_prefers_bootstrap_over_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bootstrap = tmp_path / "bootstrap.json"
        run_skills = tmp_path / "run" / "skills.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", str(bootstrap))
        monkeypatch.setenv("RUN_SKILLS_PATH", str(run_skills))
        assert active_skill_path() == bootstrap

    def test_falls_through_to_run_skills_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", raising=False)
        target = tmp_path / "run" / "skills.json"
        monkeypatch.setenv("RUN_SKILLS_PATH", str(target))
        assert active_skill_path() == target

    def test_falls_through_to_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", raising=False)
        monkeypatch.delenv("RUN_SKILLS_PATH", raising=False)
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        assert active_skill_path() == run_dir / "skills.json"


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

    def test_add_rejects_duplicate_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("find_player", "desc1", "result = 1")
        with pytest.raises(ValueError, match="already in use"):
            store.add("find_player", "desc2", "result = 2")

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
        assert "[skill_001] find_player (geo, analysis)" in out
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
