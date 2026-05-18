from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.templates.continual_harness.memory import (
    MAX_ENTRIES,
    SEARCH_MAX_MATCHES,
    MemoryStore,
    bootstrap_memory_path,
    format_memory_overview,
)


def _store(tmp_path: Path, game_id: str = "ls20") -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json", game_id=game_id)


@pytest.mark.unit
class TestBootstrapMemoryPath:
    def test_returns_none_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", raising=False)
        assert bootstrap_memory_path() is None

    def test_returns_set_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "mem.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", str(target))
        assert bootstrap_memory_path() == target


@pytest.mark.unit
class TestMemoryStoreCRUD:
    def test_store_loads_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.all_entries() == []

    def test_add_creates_entry_with_sequential_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        first = store.add("title 1", "body 1")
        second = store.add("title 2", "body 2")
        assert first.id == "mem_001"
        assert second.id == "mem_002"

    def test_add_stamps_game_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path, game_id="some-game")
        entry = store.add("t", "b")
        assert entry.game_id == "some-game"

    def test_add_persists_across_store_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        MemoryStore(path, game_id="g").add("t", "b")
        reopened = MemoryStore(path, game_id="g")
        entries = reopened.all_entries()
        assert len(entries) == 1
        assert entries[0].title == "t"

    def test_add_rejects_empty_title_or_body(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add("", "body")
        with pytest.raises(ValueError):
            store.add("title", "")

    def test_add_truncates_oversize_body(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("t", "x" * 5000)
        assert len(entry.body) < 5000
        assert entry.body.endswith("…")

    def test_add_rejects_when_full(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(MAX_ENTRIES):
            store.add(f"t{i}", "b")
        with pytest.raises(ValueError, match="memory full"):
            store.add("overflow", "b")

    def test_delete_returns_false_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("t", "b")
        assert store.delete("mem_999") is False

    def test_delete_does_not_reuse_ids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("t1", "b")
        store.add("t2", "b")
        store.add("t3", "b")
        assert store.delete("mem_002") is True
        new = store.add("t4", "b")
        assert new.id == "mem_004"

    def test_edit_updates_specific_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        original = store.add("orig title", "orig body", tags=["a"])
        edited = store.edit(original.id, body="new body")
        assert edited is not None
        assert edited.title == "orig title"
        assert edited.body == "new body"
        assert edited.tags == ["a"]
        assert edited.updated_at >= original.updated_at

    def test_edit_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.edit("mem_999", title="new") is None

    def test_edit_requires_at_least_one_field(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("t", "b")
        with pytest.raises(ValueError):
            store.edit(entry.id)


@pytest.mark.unit
class TestMemoryStoreSearch:
    def test_substring_over_title_body_tags(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("about apples", "body has nothing", tags=["fruit"])
        store.add("nothing here", "talks about BANANAS", tags=["fruit"])
        store.add("third", "third body", tags=["MISC", "important"])

        title_hits, _ = store.search("apple")
        body_hits, _ = store.search("banana")
        tag_hits, _ = store.search("important")
        case_hits, _ = store.search("MISC")

        assert [e.id for e in title_hits] == ["mem_001"]
        assert [e.id for e in body_hits] == ["mem_002"]
        assert [e.id for e in tag_hits] == ["mem_003"]
        assert [e.id for e in case_hits] == ["mem_003"]

    def test_caps_returned_at_max_matches(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(15):
            store.add(f"common title {i}", "body")
        matches, total = store.search("common")
        assert len(matches) == SEARCH_MAX_MATCHES
        assert total == 15

    def test_empty_query_returns_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("a", "x")
        store.add("b", "y")
        matches, total = store.search("")
        assert {m.id for m in matches} == {"mem_001", "mem_002"}
        assert total == 2


@pytest.mark.unit
class TestFormatMemoryOverview:
    def test_empty_has_hint(self) -> None:
        out = format_memory_overview([])
        assert "## LONG-TERM MEMORY (0 entries)" in out
        assert "No memories saved yet" in out
        assert 'process_memory(operation="add"' in out

    def test_lists_id_title_tags_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(
            "title only",
            "VERY SECRET BODY",
            tags=["alpha", "beta"],
        )
        out = format_memory_overview(store.all_entries())
        assert "[mem_001] title only (alpha, beta)" in out
        assert "VERY SECRET BODY" not in out  # body never leaks into overview


@pytest.mark.unit
class TestStoreRobustness:
    def test_corrupt_json_self_heals_to_empty_state(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        path.write_text("{this is not valid json")
        store = MemoryStore(path, game_id="g")
        assert store.all_entries() == []
        # Subsequent add must still work and overwrite the bad file.
        entry = store.add("t", "b")
        assert entry.id == "mem_001"
        reloaded = json.loads(path.read_text())
        assert reloaded["entries"][0]["title"] == "t"

    def test_load_advances_next_id_past_hand_edited_entries(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "memory.json"
        # Pretend a human curated the file with mem_010 but forgot to bump next_id.
        path.write_text(
            json.dumps(
                {
                    "next_id": 1,
                    "entries": [
                        {
                            "id": "mem_010",
                            "game_id": "g",
                            "title": "seed",
                            "body": "seed body",
                            "tags": [],
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        store = MemoryStore(path, game_id="g")
        new = store.add("after seed", "body")
        assert new.id == "mem_011"
