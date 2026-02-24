"""Tests for Grammar v1 features: mood, negation, same_as edges, default importance, warm cache."""

from __future__ import annotations

import time

import pytest

from tests.conftest import MockEmbedder
from hypabase import Hypabase
from hypabase.memory.agent import Memory
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength
from hypabase.memory.types import MOODS, DEFAULT_MOOD, Mood


# ==================================================================
# TestMoodType
# ==================================================================


class TestMoodType:
    def test_mood_type_values(self):
        assert MOODS == {"actual", "planned", "uncertain", "normative"}

    def test_default_mood(self):
        assert DEFAULT_MOOD == "actual"


# ==================================================================
# TestMoodRemember
# ==================================================================


class TestMoodRemember:
    def test_remember_with_mood_stores_in_properties(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Friday", "type": "time", "role": "locus"},
            ],
            text="Alice will deploy on Friday",
            mood="planned",
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.properties["mood"] == "planned"
        mem.hb.close()

    def test_remember_without_mood_does_not_store(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert "mood" not in edge.properties
        mem.hb.close()

    def test_all_four_moods_accepted(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for mood_val in ("actual", "planned", "uncertain", "normative"):
            result = mem.remember(
                action="test",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": f"Thing{mood_val}", "type": "thing", "role": "object"},
                ],
                mood=mood_val,
            )
            edge = mem.hb.get_edge(result["edge_id"])
            assert edge.properties["mood"] == mood_val
        mem.hb.close()

    def test_invalid_mood_raises(self, tmp_db_path):
        """Invalid mood raises ValueError."""
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember(
                action="test",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": "Thing", "type": "thing", "role": "object"},
                ],
                mood="hypothetical",
            )
        mem.hb.close()


# ==================================================================
# TestMoodRecall
# ==================================================================


class TestMoodRecall:
    def test_recall_filter_by_mood(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        results = mem.recall(entity="Alice", mood="planned")
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "planned"
        mem.hb.close()

    def test_recall_actual_finds_memories_without_explicit_mood(self, tmp_db_path):
        """mood="actual" should match memories that have no explicit mood set."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
            # No mood set — should be treated as "actual"
        )
        results = mem.recall(entity="Alice", mood="actual")
        assert len(results) >= 1
        for r in results:
            assert r["mood"] == "actual"
        mem.hb.close()

    def test_recall_mood_in_result_dicts(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="uses",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            mood="actual",
        )
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert "mood" in results[0]
        assert results[0]["mood"] == "actual"
        mem.hb.close()

    def test_recall_filter_by_mood_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
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
        result = mem.remember(
            action="use",
            entities=[
                {"name": "team", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            negated=True,
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.properties["negated"] is True
        mem.hb.close()

    def test_remember_default_negated_not_stored(self, tmp_db_path):
        """negated=False (default) should not be stored in properties."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="use",
            entities=[
                {"name": "team", "type": "group", "role": "agent"},
                {"name": "PostgreSQL", "type": "technology", "role": "object"},
            ],
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert "negated" not in edge.properties
        mem.hb.close()

    def test_recall_filter_negated_true(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "PostgreSQL", "type": "technology", "role": "object"},
            ],
        )
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            negated=True,
        )
        results = mem.recall(entity="Team Alpha", negated=True)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is True
        mem.hb.close()

    def test_recall_filter_negated_false(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "PostgreSQL", "type": "technology", "role": "object"},
            ],
        )
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            negated=True,
        )
        results = mem.recall(entity="Team Alpha", negated=False)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is False
        mem.hb.close()

    def test_recall_filter_by_negated_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "PostgreSQL", "type": "technology", "role": "object"},
            ],
        )
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            negated=True,
        )
        results = mem.recall(entity="Team Alpha", negated=True)
        assert len(results) >= 1
        for r in results:
            assert r["negated"] is True
        mem.hb.close()


# ==================================================================
# TestSameAsEdges
# ==================================================================


class TestSameAsEdges:
    def test_alias_creates_same_as_edge(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        resolver.resolve("Bob Jones", "person")
        hb.node("Bob Jones", type="person")
        resolver.resolve("Bob", "person")
        # Check that a same_as edge exists
        edges = hb.edges_by_vertex_set(["Bob", "Bob Jones"])
        same_as = [e for e in edges if e.type == "same_as"]
        assert len(same_as) >= 1
        assert same_as[0].source == "entity_resolution"
        hb.close()

    def test_upsert_prevents_duplicate_same_as(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        resolver.resolve("Bob Jones", "person")
        hb.node("Bob Jones", type="person")
        # Force re-resolution by clearing cache and re-adding
        resolver._cache.clear()
        resolver.resolve("Bob Jones", "person")
        resolver.resolve("Bob", "person")
        resolver._cache.clear()
        resolver._cache["bob jones"] = "Bob Jones"
        resolver.resolve("Bob", "person")
        # Should still be only one same_as edge
        edges = hb.edges_by_vertex_set(["Bob", "Bob Jones"])
        same_as = [e for e in edges if e.type == "same_as"]
        assert len(same_as) == 1
        hb.close()

    def test_same_as_surfaces_as_alias(self, tmp_db_path):
        """connections() should surface same_as endpoints as aliases, not neighbors."""
        mem = Memory(path=tmp_db_path)
        # Create a memory about "Bob Jones"
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        # Resolve "Bob" to "Bob Jones" — creates same_as edge
        mem._resolver.resolve("Bob", "person")
        # BFS from "Bob" should report "Bob Jones" as an alias, not a neighbor
        result = mem.connections("Bob", max_hops=2)
        assert "Bob Jones" in result["aliases"]
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "Bob Jones" not in neighbor_ids
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
        # 1.0 salience should be exactly double 0.5 salience
        assert s_full == pytest.approx(s_default * 2, abs=0.001)

    def test_explicit_importance_has_higher_salience(self, tmp_db_path):
        """Memory with importance=1.0 should rank higher than default."""
        mem = Memory(path=tmp_db_path)
        r_default = mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "cats", "type": "animal", "role": "object"},
            ],
            # No importance → defaults to 0.5 in strength calc
        )
        r_important = mem.remember(
            action="loves",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "dogs", "type": "animal", "role": "object"},
            ],
            importance=1.0,
        )
        results = mem.recall(entity="Alice")
        # The important memory should rank higher
        strengths = {r["edge"].id: r["strength"] for r in results}
        assert strengths[r_important["edge_id"]] > strengths[r_default["edge_id"]]
        mem.hb.close()


# ==================================================================
# TestWarmCache
# ==================================================================


class TestWarmCache:
    def test_memory_init_warms_cache(self, tmp_db_path):
        """Memory.__init__ calls warm_cache(), so aliases resolve on restart."""
        mem1 = Memory(path=tmp_db_path)
        mem1.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        # Force alias by resolving "Bob" → "Bob Jones"
        mem1._resolver.resolve("Bob", "person")
        mem1.hb.close()

        # Reopen — warm_cache should load the aliases
        mem2 = Memory(path=tmp_db_path)
        assert mem2._resolver.resolve("Bob", "person") == "Bob Jones"
        mem2.hb.close()

    def test_warm_cache_loads_aliases_from_properties(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice Smith", type="person", aliases=["Alice"])
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb2)
        resolver.warm_cache()
        assert resolver.resolve("Alice", "person") == "Alice Smith"
        hb2.close()


# ==================================================================
# TestForgetWithMood
# ==================================================================


class TestForgetWithMood:
    def test_forget_by_mood(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        # Forget only planned memories
        result = mem.forget(mood="planned")
        assert result["expired_count"] == 1
        # Actual memory should survive
        active = mem.hb.edges(active=True)
        moods = [e.properties.get("mood") for e in active]
        assert "planned" not in moods
        mem.hb.close()

    def test_forget_actual_does_not_sweep_legacy_memories(self, tmp_db_path):
        """forget(mood="actual") must NOT expire memories without explicit mood."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
            # No mood set — legacy memory
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v3", "type": "system", "role": "object"},
            ],
            mood="actual",
        )
        result = mem.forget(mood="actual")
        # Only the explicitly "actual" memory should be expired
        assert result["expired_count"] == 1
        assert result["skipped_untagged"] == 1
        active = [e for e in mem.hb.edges(active=True) if e.type != "same_as"]
        # Legacy (no mood) and planned should survive
        assert len(active) == 2
        moods = {e.properties.get("mood") for e in active}
        assert None in moods      # legacy memory survives
        assert "planned" in moods  # planned memory survives
        mem.hb.close()


# ==================================================================
# TestMoodAndNegationCombined
# ==================================================================


class TestMoodAndNegationCombined:
    def test_normative_negated(self, tmp_db_path):
        """'We should NOT use X' → mood=normative, negated=true."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            mood="normative",
            negated=True,
            importance=0.8,
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.properties["mood"] == "normative"
        assert edge.properties["negated"] is True
        mem.hb.close()

    def test_recall_normative_negated(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "PostgreSQL", "type": "technology", "role": "object"},
            ],
        )
        mem.remember(
            action="use",
            entities=[
                {"name": "Team Alpha", "type": "group", "role": "agent"},
                {"name": "MongoDB", "type": "technology", "role": "object"},
            ],
            mood="normative",
            negated=True,
        )
        # "What should we NOT do?"
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
        """recall(entity=...) should record access on returned edges (retrieval practice)."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        # First recall — should record access
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        edge_id = results[0]["edge"].id

        stats = mem.hb.storage.get_access_stats(mem.hb.current_namespace, "edge", edge_id)
        assert stats is not None
        assert stats["access_count"] >= 1

        mem.hb.close()

    def test_recall_repeated_access_increments(self, tmp_db_path):
        """Multiple recall(entity=...) calls should increment access_count."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem.recall(entity="Alice")
        mem.recall(entity="Alice")
        mem.recall(entity="Alice")

        results = mem.recall(entity="Alice")
        edge_id = results[0]["edge"].id
        stats = mem.hb.storage.get_access_stats(mem.hb.current_namespace, "edge", edge_id)
        assert stats["access_count"] >= 4  # 3 previous + 1 from last call

        mem.hb.close()

    def test_recall_empty_result_no_crash(self, tmp_db_path):
        """recall(entity=...) with no matching edges should not error."""
        mem = Memory(path=tmp_db_path)
        results = mem.recall(entity="Nonexistent")
        assert results == []
        mem.hb.close()


# ==================================================================
# TestConnectionsAliases
# ==================================================================


class TestConnectionsAliases:
    def test_connections_returns_aliases_field(self, tmp_db_path):
        """connections() result must always include an 'aliases' field."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        result = mem.connections("Alice")
        assert "aliases" in result
        assert isinstance(result["aliases"], list)
        mem.hb.close()

    def test_connections_aliases_from_same_as(self, tmp_db_path):
        """Aliases are collected from same_as edges, not traversed as neighbors."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem._resolver.resolve("Bob", "person")
        result = mem.connections("Bob Jones", max_hops=2)
        assert "Bob" in result["aliases"]
        # "Bob" should NOT be in neighbors
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "Bob" not in neighbor_ids
        mem.hb.close()

    def test_connections_same_as_edges_not_in_edges_list(self, tmp_db_path):
        """same_as edges should not appear in the edges list."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem._resolver.resolve("Bob", "person")
        result = mem.connections("Bob Jones", max_hops=2)
        edge_types = {e["type"] for e in result["edges"]}
        assert "same_as" not in edge_types
        mem.hb.close()

    def test_connections_explicit_edge_types_same_as_traverses(self, tmp_db_path):
        """When edge_types=["same_as"] is explicit, same_as edges ARE traversed."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem._resolver.resolve("Bob", "person")
        result = mem.connections("Bob", max_hops=2, edge_types=["same_as"])
        # With explicit edge_types, same_as IS traversed normally
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "Bob Jones" in neighbor_ids
        mem.hb.close()


# ==================================================================
# TestAutoPruning
# ==================================================================


class TestAutoPruning:
    def test_forget_prunes_orphaned_same_as(self, tmp_db_path):
        """After forget(), orphaned same_as edges should be auto-expired."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem._resolver.resolve("Bob", "person")

        # Verify same_as edge exists
        same_as_edges = [
            e for e in mem.hb.edges_by_vertex_set(["Bob", "Bob Jones"])
            if e.type == "same_as"
        ]
        assert len(same_as_edges) >= 1

        # Forget all memories involving Bob Jones
        count = mem.forget(entity="Bob Jones")["expired_count"]
        assert count >= 1

        # same_as edge should now be expired (orphaned)
        same_as_edges = [
            e for e in mem.hb.edges_by_vertex_set(["Bob", "Bob Jones"])
            if e.type == "same_as" and e.is_active
        ]
        assert len(same_as_edges) == 0
        mem.hb.close()

    def test_forget_does_not_prune_same_as_with_remaining_memories(self, tmp_db_path):
        """same_as edges should survive if endpoints still have active memories."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem.remember(
            action="likes",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
        )
        mem._resolver.resolve("Bob", "person")

        # Forget only Python memory by entity filter
        count = mem.forget(entity="Python")["expired_count"]
        assert count >= 1

        # same_as edge should still be active (Bob Jones still has Java memory)
        same_as_edges = [
            e for e in mem.hb.edges_by_vertex_set(["Bob", "Bob Jones"])
            if e.type == "same_as" and e.is_active
        ]
        assert len(same_as_edges) >= 1
        mem.hb.close()


# ==================================================================
# TestMoodValidation
# ==================================================================


class TestMoodValidation:
    def test_valid_mood_accepted(self, tmp_db_path):
        """Valid moods should not raise."""
        mem = Memory(path=tmp_db_path)
        for mood_val in ("actual", "planned", "uncertain", "normative"):
            result = mem.remember(
                action="test",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": f"X{mood_val}", "type": "thing", "role": "object"},
                ],
                mood=mood_val,
            )
            assert result["edge_id"] is not None
        mem.hb.close()

    def test_none_mood_accepted(self, tmp_db_path):
        """No mood (None) should not raise."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        assert result["edge_id"] is not None
        mem.hb.close()

    def test_typo_mood_raises(self, tmp_db_path):
        """Typo mood should raise ValueError."""
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember(
                action="deploy",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": "API", "type": "system", "role": "object"},
                ],
                mood="planed",
            )
        mem.hb.close()


# ==================================================================
# TestForgetMoodSafety
# ==================================================================


class TestForgetMoodSafety:
    def test_forget_planned_only_hits_planned(self, tmp_db_path):
        """forget(mood="planned") should only expire explicitly planned memories."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v1", "type": "system", "role": "object"},
            ],
            mood="actual",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v0", "type": "system", "role": "object"},
            ],
        )
        result = mem.forget(mood="planned")
        assert result["expired_count"] == 1
        assert result["skipped_untagged"] == 1  # legacy memory protected
        active = [e for e in mem.hb.edges(active=True) if e.type != "same_as"]
        assert len(active) == 2  # actual + legacy survive
        mem.hb.close()

    def test_forget_without_mood_does_not_filter_by_mood(self, tmp_db_path):
        """forget() without mood parameter should not apply mood filtering."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v1", "type": "system", "role": "object"},
            ],
            mood="actual",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        # forget by entity without mood — should expire both
        count = mem.forget(entity="Alice")["expired_count"]
        assert count == 2
        mem.hb.close()
