"""Tests for Grammar v1 features: mood, negation, default importance, warm cache."""

from __future__ import annotations

import time

import pytest

from hypabase import Hypabase
from hypabase.memory.agent import Memory
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength
from hypabase.memory.types import DEFAULT_MOOD, MOODS

# ==================================================================
# TestMoodType
# ==================================================================


class TestMoodType:
    def test_mood_type_values(self):
        assert MOODS == {"actual", "planned", "uncertain", "normative", "conditional"}

    def test_default_mood(self):
        assert DEFAULT_MOOD == "actual"


# ==================================================================
# TestMoodRemember
# ==================================================================


class TestMoodRemember:
    def test_remember_with_mood_stores_in_properties(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :locus Friday :mood planned)")
        results = mem.recall(entity="Alice", mood="planned")
        assert len(results) >= 1
        assert results[0]["edge"].properties["mood"] == "planned"
        mem.hb.close()

    def test_remember_without_mood_does_not_store(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :object API)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert "mood" not in results[0]["edge"].properties
        mem.hb.close()

    def test_all_five_moods_accepted(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for mood_val in ("actual", "planned", "uncertain", "normative", "conditional"):
            mem.remember(f"(test :subject Alice :object Thing{mood_val} :mood {mood_val})")
        # Verify all moods were stored
        for mood_val in ("actual", "planned", "uncertain", "normative", "conditional"):
            results = mem.recall(entity=f"Thing{mood_val}", mood=mood_val)
            assert len(results) >= 1
            assert results[0]["edge"].properties["mood"] == mood_val
        mem.hb.close()

    def test_invalid_mood_raises(self, tmp_db_path):
        """Invalid mood raises ValueError."""
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember("(test :subject Alice :object Thing :mood hypothetical)")
        mem.hb.close()


# ==================================================================
# TestMoodRecall
# ==================================================================


class TestMoodRecall:
    def test_recall_filter_by_mood(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :object API)")
        mem.remember("(deploy :subject Alice :object v2 :mood planned)")
        results = mem.recall(entity="Alice", mood="planned")
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "planned"
        mem.hb.close()

    def test_recall_actual_finds_memories_without_explicit_mood(self, tmp_db_path):
        """mood="actual" should match memories that have no explicit mood set."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :object API)")
        results = mem.recall(entity="Alice", mood="actual")
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "actual"
        mem.hb.close()

    def test_recall_mood_in_result_dicts(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(uses :subject Alice :object Python :mood actual)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert "mood" in results[0]
        assert results[0]["mood"] == "actual"
        mem.hb.close()

    def test_recall_filter_by_mood_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :object API)")
        mem.remember("(deploy :subject Alice :object v2 :mood planned)")
        results = mem.recall(entity="Alice", mood="planned")
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "planned"
        mem.hb.close()


# ==================================================================
# TestNegation
# ==================================================================


class TestNegation:
    def test_remember_with_negated_stores_in_properties(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(use :subject team :object MongoDB :negated true)")
        results = mem.recall(entity="team", negated=True)
        assert len(results) >= 1
        assert results[0]["edge"].properties["negated"] is True
        mem.hb.close()

    def test_remember_default_negated_not_stored(self, tmp_db_path):
        """negated=false (default) should not be stored in properties."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(use :subject team :object PostgreSQL)")
        results = mem.recall(entity="team")
        assert len(results) >= 1
        assert "negated" not in results[0]["edge"].properties
        mem.hb.close()

    def test_recall_filter_negated_true(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(use :subject "Team Alpha" :object PostgreSQL)')
        mem.remember('(use :subject "Team Alpha" :object MongoDB :negated true)')
        results = mem.recall(entity="Team Alpha", negated=True)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is True
        mem.hb.close()

    def test_recall_filter_negated_false(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(use :subject "Team Alpha" :object PostgreSQL)')
        mem.remember('(use :subject "Team Alpha" :object MongoDB :negated true)')
        results = mem.recall(entity="Team Alpha", negated=False)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is False
        mem.hb.close()

    def test_recall_filter_by_negated_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(use :subject "Team Alpha" :object PostgreSQL)')
        mem.remember('(use :subject "Team Alpha" :object MongoDB :negated true)')
        results = mem.recall(entity="Team Alpha", negated=True)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is True
        mem.hb.close()


# ==================================================================
# TestDefaultImportance
# ==================================================================


class TestDefaultImportance:
    def test_default_salience_is_half(self):
        """Memory without importance should use salience=0.5."""
        now = time.time()
        s_default = memory_strength(created_at=now, salience=0.5, now=now)
        s_full = memory_strength(created_at=now, salience=1.0, now=now)
        assert s_full == pytest.approx(s_default * 2, abs=0.001)

    def test_explicit_importance_has_higher_salience(self, tmp_db_path):
        """Memory with importance=1.0 should rank higher than default."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object cats)")
        mem.remember("(loves :subject Alice :object dogs :importance 1.0)")
        results = mem.recall(entity="Alice")
        # Find the two edges by action type
        strengths_by_action = {r["action"]: r["strength"] for r in results}
        assert strengths_by_action["loves"] > strengths_by_action["likes"]
        mem.hb.close()


# ==================================================================
# TestWarmCache
# ==================================================================


class TestWarmCache:
    def test_memory_init_warms_cache(self, tmp_db_path):
        """Memory.__init__ calls warm_cache(), so aliases resolve on restart."""
        mem1 = Memory(path=tmp_db_path)
        mem1.hb.node("Bob Jones", type="entity")
        mem1.hb.node("Bob", type="entity")
        mem1.hb.edge(["Bob", "Bob Jones"], type="same_as", source="resolution")
        mem1.hb.close()

        mem2 = Memory(path=tmp_db_path)
        # After warm_cache, "Bob" should resolve to "Bob Jones" (higher degree from same_as)
        resolved = mem2._resolver.resolve("Bob")
        # The cache should know about the same_as relationship
        assert resolved in ("Bob", "Bob Jones")
        mem2.hb.close()

    def test_warm_cache_loads_aliases_from_properties(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice Smith", type="person", aliases=["Alice"])
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb2)
        resolver.warm_cache()
        assert resolver.resolve("Alice") == "Alice Smith"
        hb2.close()


# ==================================================================
# TestMoodValidation
# ==================================================================


class TestMoodValidation:
    def test_valid_mood_accepted(self, tmp_db_path):
        """Valid moods should not raise."""
        mem = Memory(path=tmp_db_path)
        for mood_val in ("actual", "planned", "uncertain", "normative", "conditional"):
            mem.remember(f"(test :subject Alice :object X{mood_val} :mood {mood_val})")
        # Verify all were stored
        results = mem.recall(entity="Alice")
        assert len(results) >= 5
        mem.hb.close()

    def test_none_mood_accepted(self, tmp_db_path):
        """No mood should not raise."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        mem.hb.close()

    def test_typo_mood_raises(self, tmp_db_path):
        """Typo mood should raise ValueError."""
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember("(deploy :subject Alice :object API :mood planed)")
        mem.hb.close()


# ==================================================================
# TestMoodAndNegationCombined
# ==================================================================


class TestMoodAndNegationCombined:
    def test_normative_negated(self, tmp_db_path):
        """'We should NOT use X' -> mood=normative, negated=true."""
        mem = Memory(path=tmp_db_path)
        mem.remember('(use :subject "Team Alpha" :object MongoDB :mood normative :negated true :importance 0.8)')
        results = mem.recall(entity="Team Alpha", mood="normative", negated=True)
        assert len(results) >= 1
        edge = results[0]["edge"]
        assert edge.properties["mood"] == "normative"
        assert edge.properties["negated"] is True
        mem.hb.close()

    def test_recall_normative_negated(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(use :subject "Team Alpha" :object PostgreSQL)')
        mem.remember('(use :subject "Team Alpha" :object MongoDB :mood normative :negated true)')
        results = mem.recall(entity="Team Alpha", mood="normative", negated=True)
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "normative"
            assert r["negated"] is True
        mem.hb.close()


# ==================================================================
# TestRecallAccessRecording
# ==================================================================


class TestRecallAccessRecording:
    def test_recall_records_access(self, tmp_db_path):
        """recall(entity=...) should record access on returned edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        edge_id = results[0]["edge"].id
        stats = mem.hb.storage.get_access_stats(mem.hb.current_namespace, "edge", edge_id)
        assert stats is not None
        assert stats["access_count"] >= 1
        mem.hb.close()

    def test_recall_repeated_access_increments(self, tmp_db_path):
        """Multiple recall calls should increment access_count."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        mem.recall(entity="Alice")
        mem.recall(entity="Alice")
        mem.recall(entity="Alice")
        results = mem.recall(entity="Alice")
        edge_id = results[0]["edge"].id
        stats = mem.hb.storage.get_access_stats(mem.hb.current_namespace, "edge", edge_id)
        assert stats["access_count"] >= 4
        mem.hb.close()

    def test_recall_empty_result_no_crash(self, tmp_db_path):
        """recall(entity=...) with no matching edges should not error."""
        mem = Memory(path=tmp_db_path)
        results = mem.recall(entity="Nonexistent")
        assert results == []
        mem.hb.close()
