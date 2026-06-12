from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.templates.continual_harness.memory import (
    MAX_ENTRIES,
    SEARCH_MAX_MATCHES,
    MemoryStore,
    active_memory_path,
    format_memory_full,
    format_memory_overview,
)


def _store(tmp_path: Path, game_id: str = "ls20") -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json", game_id=game_id)


@pytest.mark.unit
class TestActiveMemoryPath:
    def test_per_game_under_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))

        assert active_memory_path("ls20") == run_dir / "ls20" / "memory.json"
        # Different games must never resolve to the same file.
        assert active_memory_path("ls20") != active_memory_path("vc33")

    def test_falls_back_to_run_dir_without_game_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        monkeypatch.delenv("RUN_LOG_PATH", raising=False)

        assert active_memory_path() == run_dir / "memory.json"

    def test_falls_back_to_run_log_sibling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("RUN_DIR", raising=False)
        monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "run.log"))

        assert active_memory_path("ls20") == tmp_path / "memory.json"


@pytest.mark.unit
class TestMemoryStoreCRUD:
    def test_store_loads_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.all_entries() == []

    def test_add_creates_entry_with_sequential_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        first = store.add("title 1", "body 1", confidence=3)
        second = store.add("title 2", "body 2", confidence=3)
        assert first.id == "mem_001"
        assert second.id == "mem_002"

    def test_add_stamps_game_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path, game_id="some-game")
        entry = store.add("t", "b", confidence=3)
        assert entry.game_id == "some-game"

    def test_add_persists_across_store_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        MemoryStore(path, game_id="g").add("t", "b", confidence=3)
        reopened = MemoryStore(path, game_id="g")
        entries = reopened.all_entries()
        assert len(entries) == 1
        assert entries[0].title == "t"

    def test_add_rejects_empty_title_or_body(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.add("", "body", confidence=3)
        with pytest.raises(ValueError):
            store.add("title", "", confidence=3)

    def test_add_truncates_oversize_body(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("t", "x" * 5000, confidence=3)
        assert len(entry.body) < 5000
        assert entry.body.endswith("…")

    def test_add_rejects_when_full(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(MAX_ENTRIES):
            store.add(f"t{i}", "b", confidence=3)
        with pytest.raises(ValueError, match="memory full"):
            store.add("overflow", "b", confidence=3)

    def test_delete_returns_false_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("t", "b", confidence=3)
        assert store.delete("mem_999") is False

    def test_delete_does_not_reuse_ids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("t1", "b", confidence=3)
        store.add("t2", "b", confidence=3)
        store.add("t3", "b", confidence=3)
        assert store.delete("mem_002") is True
        new = store.add("t4", "b", confidence=3)
        assert new.id == "mem_004"

    def test_edit_updates_specific_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        original = store.add("orig title", "orig body", tags=["a"], confidence=3)
        edited = store.edit(original.id, body="new body")
        assert edited is not None
        assert edited.title == "orig title"
        assert edited.body == "new body"
        assert edited.tags == ["a"]
        assert edited.confidence == 3
        assert edited.updated_at >= original.updated_at

    def test_edit_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.edit("mem_999", title="new") is None

    def test_edit_requires_at_least_one_field(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("t", "b", confidence=3)
        with pytest.raises(ValueError):
            store.edit(entry.id)


@pytest.mark.unit
class TestMemoryConfidence:
    def test_add_round_trips_confidence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("player is orange", "ACTION1 moved it up.", confidence=5)
        assert entry.confidence == 5
        reopened = MemoryStore(store.path, game_id="ls20")
        assert reopened.all_entries()[0].confidence == 5
        matches, _ = reopened.search("orange")
        assert matches[0].confidence == 5

    def test_add_requires_confidence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="add requires confidence"):
            store.add("t", "b")

    def test_add_rejects_out_of_range_or_non_integer(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="between 1 and 5"):
            store.add("t", "b", confidence=0)
        with pytest.raises(ValueError, match="between 1 and 5"):
            store.add("t", "b", confidence=6)
        with pytest.raises(ValueError, match="integer 1-5"):
            store.add("t", "b", confidence="high")

    def test_add_coerces_integral_strings(self, tmp_path: Path) -> None:
        # Gemini occasionally sends numbers as strings; int("4") is accepted.
        store = _store(tmp_path)
        entry = store.add("t", "b", confidence="4")
        assert entry.confidence == 4

    def test_add_rounds_numeric_confidence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        low = store.add("low", "b", confidence=1.2)
        high = store.add("high", "b", confidence=4.9)

        assert low.confidence == 1
        assert high.confidence == 5

    def test_add_rejects_boolean_confidence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="integer 1-5"):
            store.add("t", "b", confidence=True)

    def test_edit_confidence_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        original = store.add("t", "b", confidence=1)
        edited = store.edit(original.id, confidence=5)
        assert edited is not None
        assert edited.confidence == 5
        assert edited.body == "b"
        assert edited.updated_at >= original.updated_at

    def test_edit_rejects_out_of_range_confidence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        entry = store.add("t", "b", confidence=3)
        with pytest.raises(ValueError, match="between 1 and 5"):
            store.edit(entry.id, confidence=9)

    def test_legacy_entries_without_confidence_default_to_3(
        self, tmp_path: Path
    ) -> None:
        # Bootstrapped memory files written before the confidence field existed.
        path = tmp_path / "memory.json"
        path.write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "mem_001",
                            "game_id": "g",
                            "title": "legacy",
                            "body": "legacy body",
                            "tags": [],
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        store = MemoryStore(path, game_id="g")
        entries = store.all_entries()
        assert entries[0].confidence == 3
        matches, _ = store.search("legacy")
        assert matches[0].confidence == 3


@pytest.mark.unit
class TestMemoryStoreSearch:
    def test_substring_over_title_body_tags(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("about apples", "body has nothing", tags=["fruit"], confidence=3)
        store.add("nothing here", "talks about BANANAS", tags=["fruit"], confidence=3)
        store.add("third", "third body", tags=["MISC", "important"], confidence=3)

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
            store.add(f"common title {i}", "body", confidence=3)
        matches, total = store.search("common")
        assert len(matches) == SEARCH_MAX_MATCHES
        assert total == 15

    def test_empty_query_returns_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("a", "x", confidence=3)
        store.add("b", "y", confidence=3)
        matches, total = store.search("")
        assert {m.id for m in matches} == {"mem_001", "mem_002"}
        assert total == 2


@pytest.mark.unit
class TestFormatMemoryOverview:
    def test_empty_has_hint(self) -> None:
        out = format_memory_overview([])
        assert "## LONG-TERM MEMORY (0 entries)" in out
        assert "No facts recorded yet" in out
        assert 'process_memory(operation="add"' in out
        assert "confidence" in out

    def test_lists_id_confidence_title_tags_and_first_line(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        store.add(
            "title only",
            "first line preview\nREST OF BODY STAYS OUT",
            tags=["alpha", "beta"],
            confidence=4,
        )
        out = format_memory_overview(store.all_entries())
        assert "[mem_001][c4] title only (alpha, beta): first line preview..." in out
        # Only the first line is previewed; the rest never leaks into the index.
        assert "REST OF BODY STAYS OUT" not in out

    def test_full_dump_renders_confidence_heading(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add("guess", "maybe walls block", confidence=1)
        store.add("rule", "walls block movement", confidence=5)
        out = format_memory_full(store.all_entries())
        assert "### [mem_001] guess — confidence 1/5" in out
        assert "### [mem_002] rule — confidence 5/5" in out
        assert "maybe walls block" in out  # full dump includes bodies


@pytest.mark.unit
class TestStoreRobustness:
    def test_corrupt_json_self_heals_to_empty_state(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        path.write_text("{this is not valid json")
        store = MemoryStore(path, game_id="g")
        assert store.all_entries() == []
        # Subsequent add must still work and overwrite the bad file.
        entry = store.add("t", "b", confidence=3)
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
        new = store.add("after seed", "body", confidence=3)
        assert new.id == "mem_011"
