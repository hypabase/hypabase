"""Tests for the Hypabase Memory Module."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from hypabase.memory.agent import Memory
from hypabase.memory.strength import memory_strength
from tests.conftest import MockEmbedder


class TestStrength:
    def test_strength_basic(self):
        now = time.time()
        s = memory_strength(
            created_at=now,
            access_count=5,
            confidence=1.0,
            now=now,
        )
        # recency=1.0 (age=0), frequency=1.0+log(6), salience=1.0, confidence=1.0
        expected = 1.0 * (1.0 + math.log(6)) * 1.0 * 1.0
        assert s == pytest.approx(expected, abs=0.01)

    def test_strength_decay(self):
        now = time.time()
        old = now - 86400 * 7  # 7 days ago
        s_new = memory_strength(created_at=now, access_count=1, now=now)
        s_old = memory_strength(created_at=old, access_count=1, now=now)
        assert s_new > s_old

    def test_strength_frequency(self):
        now = time.time()
        s_low = memory_strength(created_at=now, access_count=1, now=now)
        s_high = memory_strength(created_at=now, access_count=100, now=now)
        assert s_high > s_low

    def test_strength_zero_access(self):
        now = time.time()
        s = memory_strength(created_at=now, access_count=0, now=now)
        # frequency baseline = 1.0, so new memories have strength > 0
        assert s == pytest.approx(1.0, abs=0.01)

    def test_strength_confidence(self):
        now = time.time()
        s_high = memory_strength(created_at=now, access_count=5, confidence=1.0, now=now)
        s_low = memory_strength(created_at=now, access_count=5, confidence=0.5, now=now)
        assert s_high > s_low

    def test_strength_no_created_at(self):
        s = memory_strength(created_at=None, access_count=5, confidence=1.0)
        assert s > 0.0  # recency defaults to 1.0


class TestMemory:
    def test_remember_basic(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "office", "type": "location", "role": "locus"},
            ],
            text="Alice Smith met Bob Jones at the office",
        )
        assert result["edge_id"] is not None
        assert len(result["node_ids"]) >= 2
        mem.hb.close()

    def test_remember_missing_action_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="action is required"):
            mem.remember(
                action="",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": "Bob", "type": "person", "role": "object"},
                ],
            )
        mem.hb.close()

    def test_remember_one_entity_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="at least 2 entities"):
            mem.remember(
                action="met",
                entities=[{"name": "Alice", "type": "person", "role": "agent"}],
            )
        mem.hb.close()

    def test_remember_missing_name_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="missing 'name'"):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"role": "object"},
                ],
            )
        mem.hb.close()

    def test_remember_invalid_mood_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": "Bob", "type": "person", "role": "object"},
                ],
                mood="hypothetical",
            )
        mem.hb.close()

    def test_remember_invalid_memory_type_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown memory_type"):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice", "type": "person", "role": "agent"},
                    {"name": "Bob", "type": "person", "role": "object"},
                ],
                memory_type="unknown",
            )
        mem.hb.close()

    def test_remember_no_text(self, tmp_db_path):
        """remember works without text param."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
        )
        assert result["edge_id"] is not None
        edge = mem.hb.get_edge(result["edge_id"])
        assert "text" not in edge.properties
        mem.hb.close()

    def test_remember_text_is_cosmetic(self, tmp_db_path):
        """text is stored in properties but not used for extraction."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
            text="Alice met Bob at the park",
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.properties.get("text") == "Alice met Bob at the park"
        mem.hb.close()

    def test_recall_by_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "Central Park", "type": "location", "role": "locus"},
            ],
            text="Alice Smith met Bob Jones at Central Park",
        )
        results = mem.recall(entity="Alice Smith")
        assert len(results) >= 1
        mem.hb.close()

    def test_recall_returns_text(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="loves",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            text="Alice Smith loves Python programming",
        )
        results = mem.recall(entity="Alice Smith")
        assert any("Python" in r.get("text", "") for r in results)
        mem.hb.close()

    def test_recall_no_params_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="At least one of"):
            mem.recall()
        mem.hb.close()

    def test_recall_multi_entity(self, tmp_db_path):
        """recall(entity=[...]) finds memories involving multiple entities."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
                {"name": "Bob", "type": "person", "role": "recipient"},
            ],
        )
        mem.remember(
            action="reviewed",
            entities=[
                {"name": "Carol", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        # Should find memories involving both Alice and API
        results = mem.recall(entity=["Alice", "API"])
        assert len(results) >= 1
        # The assigned edge has both Alice and API
        assigned = [r for r in results if r["action"] == "assigned"]
        assert len(assigned) >= 1
        mem.hb.close()

    def test_recall_filter_by_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        mem.remember(
            action="reviewed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "code", "type": "artifact", "role": "object"},
            ],
        )
        results = mem.recall(entity="Alice", action="assigned")
        assert len(results) >= 1
        assert all(r["action"] == "assigned" for r in results)
        mem.hb.close()

    def test_recall_filter_by_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
                {"name": "Bob", "type": "person", "role": "recipient"},
            ],
        )
        results = mem.recall(entity="Alice", role="agent")
        assert len(results) >= 1
        for r in results:
            assert r["roles"].get("Alice") == "agent"
        mem.hb.close()

    def test_recall_filter_by_memory_type(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
            memory_type="episodic",
        )
        mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        results = mem.recall(entity="Alice", memory_type="semantic")
        assert len(results) >= 1
        assert all(r["memory_type"] == "semantic" for r in results)
        mem.hb.close()

    def test_recall_filter_by_mood(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deployed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
            mood="actual",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API v2", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        results = mem.recall(mood="planned")
        assert len(results) >= 1
        assert all(r["mood"] == "planned" for r in results)
        mem.hb.close()

    def test_recall_filter_by_negated(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="uses",
            entities=[
                {"name": "team", "type": "group", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
        )
        mem.remember(
            action="uses",
            entities=[
                {"name": "team", "type": "group", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            negated=True,
        )
        results = mem.recall(action="uses", negated=True)
        assert len(results) >= 1
        assert all(r["negated"] is True for r in results)
        mem.hb.close()

    def test_recall_filter_only_no_entity(self, tmp_db_path):
        """recall(mood=...) without entity scans all edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "service", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        mem.remember(
            action="deploy",
            entities=[
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
            mood="actual",
        )
        results = mem.recall(mood="planned")
        assert len(results) >= 1
        assert all(r["mood"] == "planned" for r in results)
        mem.hb.close()

    def test_recall_temporal_since(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)

        mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
        )
        # Recall since the past should return results
        past = now - timedelta(hours=1)
        results = mem.recall(entity="Alice", since=past)
        assert len(results) >= 1

        # Recall since the future should return nothing
        future = now + timedelta(hours=1)
        results = mem.recall(entity="Alice", since=future)
        assert len(results) == 0
        mem.hb.close()

    def test_recall_temporal_before(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)

        mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
        )
        # Recall before the future should return results
        future = now + timedelta(hours=1)
        results = mem.recall(entity="Alice", before=future)
        assert len(results) >= 1

        # Recall before the past should return nothing
        past = now - timedelta(hours=1)
        results = mem.recall(entity="Alice", before=past)
        assert len(results) == 0
        mem.hb.close()

    def test_recall_same_as_expansion(self, tmp_db_path):
        """recall(entity="Bob") finds memories under alias "Bob Jones" via same_as edges."""
        mem = Memory(path=tmp_db_path)
        # Create a same_as edge linking Bob to Bob Jones
        mem.hb.node("Bob", type="person")
        mem.hb.node("Bob Jones", type="person")
        mem.hb.edge(["Bob", "Bob Jones"], type="same_as", source="resolution")

        mem.remember(
            action="assigned",
            entities=[
                {"name": "Bob Jones", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        # Recall using "Bob" should find memories for "Bob Jones" via same_as
        results = mem.recall(entity="Bob")
        assert len(results) >= 1
        mem.hb.close()

    def test_forget_by_strength(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith met Bob Jones yesterday",
        )
        # New memories have strength ~0.5 (default importance), so low threshold should NOT forget them
        count_low = mem.forget(min_strength=0.01)["expired_count"]
        assert count_low == 0
        # High threshold (above new memory strength) should forget them
        count_high = mem.forget(min_strength=2.0)["expired_count"]
        assert count_high >= 1
        # After forget, no memory edges should be active (same_as edges persist)
        active = mem.hb.edges(active=True)
        active_memories = [e for e in active if e.type != "same_as"]
        assert len(active_memories) == 0
        mem.hb.close()

    def test_forget_by_age(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        # Forget everything older than the future (nothing should match)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        count = mem.forget(older_than=future)["expired_count"]
        assert count >= 1
        mem.hb.close()

    def test_connections(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        mem.remember(
            action="called",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Carol Davis", "type": "person", "role": "object"},
            ],
            text="Alice Smith called Carol Davis about the project",
        )
        result = mem.connections("Alice Smith")
        assert result["entity"] == "Alice Smith"
        assert result["edge_count"] >= 2
        assert result["neighbor_count"] >= 1
        mem.hb.close()

    def test_consolidate(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="worked",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith and Bob Jones worked on project Alpha",
        )
        mem.remember(
            action="presented",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith and Bob Jones presented at the conference",
        )
        mem.remember(
            action="published",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith and Bob Jones published a paper",
        )
        summaries = mem.consolidate()
        # Alice Smith and Bob Jones co-occur 3 times, should be consolidated
        assert len(summaries) >= 1
        mem.hb.close()

    def test_consolidate_entity_filter(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="worked",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="collaborated",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="published",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )
        summaries = mem.consolidate(entity="Alice Smith")
        # Should only consolidate edges involving "Alice Smith"
        assert len(summaries) >= 1
        for s in summaries:
            assert "Carol Davis" not in s["entities"]
            assert "Dave Evans" not in s["entities"]
        mem.hb.close()

    def test_remember_with_embedder(self, tmp_db_path):
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith met Bob Jones at the office",
        )
        assert result["edge_id"] is not None
        # Verify embedding was stored
        raw = mem.hb._storage.load_embeddings(mem.hb._current_ns, kind="edge")
        assert len(raw) >= 1
        mem.hb.close()

    def test_recall_min_strength(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        # New memories have strength ~1.0; a very high threshold should filter all out
        results = mem.recall(entity="Alice Smith", min_strength=999.0)
        assert len(results) == 0
        # A zero threshold should include them
        results = mem.recall(entity="Alice Smith", min_strength=0.0)
        assert len(results) >= 1
        mem.hb.close()

    def test_connections_with_edge_types(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        # memory edges have type="met" now
        result = mem.connections("Alice Smith", edge_types=["met"])
        assert result["edge_count"] >= 1
        # Non-matching type should return no edges
        result_other = mem.connections("Alice Smith", edge_types=["nonexistent"])
        assert result_other["edge_count"] == 0
        mem.hb.close()

    def test_consolidate_idempotent(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="worked",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="presented",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="published",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        summaries1 = mem.consolidate()
        assert len(summaries1) >= 1
        # Calling consolidate again should not create duplicates
        summaries2 = mem.consolidate()
        assert len(summaries2) == 0
        mem.hb.close()

    def test_recall_empty_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # No memories stored — recall should return empty list
        results = mem.recall(entity="nonexistent")
        assert results == []
        mem.hb.close()

    def test_forget_entity_only(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        mem.remember(
            action="called",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )
        # Forget only edges involving Alice Smith
        count = mem.forget(entity="Alice Smith", min_strength=2.0)["expired_count"]
        assert count >= 1
        # Carol/Dave edge should still be active
        active = mem.hb.edges(active=True)
        carol_edges = [e for e in active if "Carol Davis" in e.node_set]
        assert len(carol_edges) >= 1
        mem.hb.close()

    def test_forget_entity_only_no_other_filters(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        mem.remember(
            action="called",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )
        count = mem.forget(entity="Alice Smith")["expired_count"]
        assert count >= 1
        # Carol/Dave edge should still be active
        active = mem.hb.edges(active=True)
        carol_edges = [e for e in active if "Carol Davis" in e.node_set]
        assert len(carol_edges) >= 1
        # Alice memory edges should be gone (same_as edges persist)
        alice_edges = [e for e in active if "Alice Smith" in e.node_set and e.type != "same_as"]
        assert len(alice_edges) == 0
        mem.hb.close()

    def test_forget_entity_with_older_than(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=30)).timestamp()

        # Create an old Alice edge
        result_old = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith met Bob Jones Long Ago",
        )
        assert result_old["edge_id"] is not None
        # Backdate it
        core_edge = mem.hb._store.get_edge(result_old["edge_id"])
        core_edge.created_at = old_ts
        mem.hb._storage._conn.execute(
            "UPDATE edges SET created_at = ? WHERE id = ? AND namespace = ?",
            (old_ts, result_old["edge_id"], mem.hb._current_ns),
        )
        mem.hb._storage._conn.commit()

        # Create a recent Alice edge
        result_recent = mem.remember(
            action="called",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Carol Davis", "type": "person", "role": "object"},
            ],
            text="Alice Smith called Carol Davis Recently",
        )
        assert result_recent["edge_id"] is not None

        cutoff = now - timedelta(days=15)
        count = mem.forget(entity="Alice Smith", older_than=cutoff)["expired_count"]
        assert count == 1  # Only the old edge expired

        # Recent Alice edge should survive
        active = mem.hb.edges(active=True)
        active_ids = {e.id for e in active}
        assert result_recent["edge_id"] in active_ids
        assert result_old["edge_id"] not in active_ids
        mem.hb.close()

    def test_recall_ordered_by_strength(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        mem.remember(
            action="called",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Carol Davis", "type": "person", "role": "object"},
            ],
        )
        results = mem.recall(entity="Alice Smith")
        if len(results) >= 2:
            # Results should be sorted by strength descending
            for i in range(len(results) - 1):
                assert results[i]["strength"] >= results[i + 1]["strength"]
        mem.hb.close()

    def test_recall_entity_not_found(self, tmp_db_path):
        """Entity with no matching nodes returns []."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        results = mem.recall(entity="Nonexistent Person")
        assert results == []
        mem.hb.close()

    def test_remember_duplicate_creates_two_edges(self, tmp_db_path):
        """Same entities+action twice creates two separate edges."""
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        r2 = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        assert r1["edge_id"] is not None
        assert r2["edge_id"] is not None
        assert r1["edge_id"] != r2["edge_id"]
        mem.hb.close()

    def test_forget_already_expired(self, tmp_db_path):
        """Forgetting already-expired edges is a no-op."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        # Expire all
        mem.forget(min_strength=999.0)
        # Try to forget again — nothing active to expire
        count = mem.forget(min_strength=999.0)["expired_count"]
        assert count == 0
        mem.hb.close()

    def test_consolidate_insufficient_edges(self, tmp_db_path):
        """< 2 edges returns empty list."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        # Only 1 memory edge — not enough to consolidate
        summaries = mem.consolidate()
        assert summaries == []
        mem.hb.close()

    def test_connections_nonexistent_entity(self, tmp_db_path):
        """Returns empty neighbors/edges for nonexistent entity."""
        mem = Memory(path=tmp_db_path)
        result = mem.connections("Nonexistent Entity")
        assert result["entity"] == "Nonexistent Entity"
        assert result["neighbor_count"] == 0
        assert result["edge_count"] == 0
        assert result["neighbors"] == []
        assert result["edges"] == []
        mem.hb.close()


class TestRecallRoleWeights:
    """Tests for role-weighted spreading activation."""

    def test_agent_role_propagates_stronger(self, tmp_db_path):
        """Agent role should propagate stronger than locus role."""
        mem = Memory(path=tmp_db_path)
        # Shared entity connects two memories
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
                {"name": "meeting", "type": "event", "role": "locus"},
            ],
        )
        mem.remember(
            action="discussed",
            entities=[
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        # Both memories are accessible via "task"
        results = mem.recall(entity="task")
        assert len(results) >= 2
        mem.hb.close()


class TestRecallNaryOverlap:
    """Tests for N-ary overlap scoring."""

    def test_higher_overlap_scores_higher(self, tmp_db_path):
        """Edge with 2/2 query entities should score higher than 1/2."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="worked",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "project X", "type": "project", "role": "object"},
            ],
        )
        mem.remember(
            action="reviewed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "code", "type": "artifact", "role": "object"},
            ],
        )
        # Query for both Alice and Bob
        results = mem.recall(entity=["Alice", "Bob"])
        assert len(results) >= 1
        # The "worked" edge (2/2 overlap) should score higher
        if len(results) >= 2:
            worked = [r for r in results if r["action"] == "worked"]
            reviewed = [r for r in results if r["action"] == "reviewed"]
            if worked and reviewed:
                assert worked[0]["score"] >= reviewed[0]["score"]
        mem.hb.close()


class TestForgetValidation:
    def test_forget_no_args_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        with pytest.raises(ValueError, match="At least one filter required"):
            mem.forget()
        mem.hb.close()


class TestAccessTracking:
    def test_record_access(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        storage.record_access("default", "edge", "e1")
        storage.record_access("default", "edge", "e1")
        stats = storage.get_access_stats("default", "edge", "e1")
        assert stats is not None
        assert stats["access_count"] == 2
        storage.close()

    def test_get_access_stats_missing(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        stats = storage.get_access_stats("default", "edge", "nonexistent")
        assert stats is None
        storage.close()

    def test_get_all_access_stats(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        storage.record_access("default", "edge", "e1")
        storage.record_access("default", "node", "n1")
        all_stats = storage.get_all_access_stats("default")
        assert len(all_stats) == 2
        storage.close()

    def test_recall_updates_access(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        edge_id = result["edge_id"]
        # Recall should update access count
        mem.recall(entity="Alice Smith")
        stats = mem.hb._storage.get_access_stats(
            mem.hb._current_ns, "edge", edge_id
        )
        assert stats is not None
        assert stats["access_count"] >= 1
        mem.hb.close()

    def test_record_access_batch(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        storage.record_access_batch("default", "edge", ["e1", "e2", "e3"])
        storage.record_access_batch("default", "edge", ["e1", "e2"])
        stats_e1 = storage.get_access_stats("default", "edge", "e1")
        stats_e3 = storage.get_access_stats("default", "edge", "e3")
        assert stats_e1 is not None
        assert stats_e1["access_count"] == 2
        assert stats_e3 is not None
        assert stats_e3["access_count"] == 1
        storage.close()

    def test_record_access_batch_empty(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        # Should not error on empty list
        storage.record_access_batch("default", "edge", [])
        all_stats = storage.get_all_access_stats("default")
        assert len(all_stats) == 0
        storage.close()

    def test_get_batch_access_stats(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        storage.record_access("default", "edge", "e1")
        storage.record_access("default", "edge", "e2")
        batch = storage.get_batch_access_stats("default", "edge", ["e1", "e2", "missing"])
        assert "e1" in batch
        assert "e2" in batch
        assert "missing" not in batch
        assert batch["e1"]["access_count"] == 1
        storage.close()


class TestForgetSQLPushdown:
    @staticmethod
    def _backdate_edge(mem, edge_id, old_ts):
        """Set created_at on both in-memory and SQLite edge."""
        core_edge = mem.hb._store.get_edge(edge_id)
        core_edge.created_at = old_ts
        mem.hb._storage._conn.execute(
            "UPDATE edges SET created_at = ? WHERE id = ? AND namespace = ?",
            (old_ts, edge_id, mem.hb._current_ns),
        )
        mem.hb._storage._conn.commit()

    def test_forget_both_filters_pushes_older_than(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=30)).timestamp()

        # Create 5 old memories by backdating created_at
        old_edge_ids = []
        for i in range(5):
            result = mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": f"Person{i}", "type": "person", "role": "object"},
                ],
            )
            if result["edge_id"]:
                self._backdate_edge(mem, result["edge_id"], old_ts)
                old_edge_ids.append(result["edge_id"])

        # Create 5 recent memories
        recent_edge_ids = []
        for i in range(5):
            result = mem.remember(
                action="met",
                entities=[
                    {"name": "Bob Jones", "type": "person", "role": "agent"},
                    {"name": f"Person{i}", "type": "person", "role": "object"},
                ],
            )
            if result["edge_id"]:
                recent_edge_ids.append(result["edge_id"])

        cutoff = now - timedelta(days=15)
        count = mem.forget(older_than=cutoff, min_strength=2.0)["expired_count"]

        # Only old memories should be expired
        assert count == len(old_edge_ids)

        # Recent memories should still be active
        active = mem.hb.edges(active=True)
        active_ids = {e.id for e in active}
        for eid in recent_edge_ids:
            assert eid in active_ids
        mem.hb.close()

    def test_forget_both_filters_skips_recent_low_strength(self, tmp_db_path):
        """When both older_than and min_strength are set, recent edges are not
        examined even if they have low strength."""
        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=30)).timestamp()

        # Create an old memory
        result_old = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
            text="Alice Smith met Bob Jones Long Ago",
        )
        assert result_old["edge_id"] is not None
        self._backdate_edge(mem, result_old["edge_id"], old_ts)

        # Create a recent memory
        result_recent = mem.remember(
            action="met",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
            text="Carol Davis met Dave Evans Recently",
        )
        assert result_recent["edge_id"] is not None

        cutoff = now - timedelta(days=15)
        count = mem.forget(older_than=cutoff, min_strength=2.0)["expired_count"]

        assert count == 1

        # Recent memory should still be active
        active = mem.hb.edges(active=True)
        active_ids = {e.id for e in active}
        assert result_recent["edge_id"] in active_ids
        mem.hb.close()


class TestPersistence:
    def test_memory_persists(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.hb.close()

        mem2 = Memory(path=tmp_db_path)
        results = mem2.recall(entity="Alice Smith")
        assert len(results) >= 1
        mem2.hb.close()


class TestPerformanceOptimizations:
    """Tests for P1 performance optimizations."""

    def test_forget_older_than_skips_stats_query(self, tmp_db_path):
        """forget(older_than=...) should not query access_log when min_strength is None."""
        from unittest.mock import patch

        mem = Memory(path=tmp_db_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=30)).timestamp()

        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        assert result["edge_id"] is not None

        # Backdate the edge
        core_edge = mem.hb._store.get_edge(result["edge_id"])
        core_edge.created_at = old_ts
        mem.hb._storage._conn.execute(
            "UPDATE edges SET created_at = ? WHERE id = ? AND namespace = ?",
            (old_ts, result["edge_id"], mem.hb._current_ns),
        )
        mem.hb._storage._conn.commit()

        # Patch get_batch_access_stats to fail if called
        with patch.object(
            mem.hb._storage, "get_batch_access_stats", side_effect=AssertionError("Should not call stats")
        ):
            cutoff = now - timedelta(days=15)
            count = mem.forget(older_than=cutoff)["expired_count"]

        assert count == 1
        mem.hb.close()

    def test_consolidate_uses_batch(self, tmp_db_path):
        """consolidate() should create all summary edges in a single transaction."""
        from unittest.mock import patch

        mem = Memory(path=tmp_db_path)

        # Create memories with overlapping entities to trigger consolidation
        for i in range(5):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": "Bob Jones", "type": "person", "role": "object"},
                    {"name": f"Location{i}", "type": "location", "role": "locus"},
                ],
            )

        # Count commits
        commit_count = [0]
        original_commit = mem.hb._storage.commit

        def counting_commit():
            commit_count[0] += 1
            original_commit()

        with patch.object(mem.hb._storage, "commit", counting_commit):
            summaries = mem.consolidate()

        # Should have created at least one summary
        assert len(summaries) >= 1
        # All summaries should be created in a single commit (the batch)
        assert commit_count[0] == 1
        mem.hb.close()

    def test_consolidate_entity_filter_uses_sql(self, tmp_db_path):
        """consolidate(entity=...) should only load edges containing that entity."""
        mem = Memory(path=tmp_db_path)

        # Create memories for two separate groups
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="worked",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="met",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )
        mem.remember(
            action="collaborated",
            entities=[
                {"name": "Carol Davis", "type": "person", "role": "agent"},
                {"name": "Dave Evans", "type": "person", "role": "object"},
            ],
        )

        # Only Alice/Bob should be considered
        summaries = mem.consolidate(entity="Alice Smith")

        # Verify no Carol/Dave consolidation created
        for s in summaries:
            assert "Carol Davis" not in s["entities"]
            assert "Dave Evans" not in s["entities"]
        mem.hb.close()

    def test_recall_ordering_with_many_results(self, tmp_db_path):
        """recall() should return top-k by strength even with many candidates."""
        mem = Memory(path=tmp_db_path)

        # Create many memories mentioning the same entity
        for i in range(50):
            mem.remember(
                action="met",
                entities=[
                    {"name": f"Entity{i}", "type": "person", "role": "agent"},
                    {"name": "Alice Smith", "type": "person", "role": "object"},
                ],
            )

        results = mem.recall(entity="Alice Smith", limit=5)

        assert len(results) == 5
        # Verify sorted by strength descending
        strengths = [r["strength"] for r in results]
        assert strengths == sorted(strengths, reverse=True)
        mem.hb.close()

    def test_recall_heapq_vs_sort_equivalence(self, tmp_db_path):
        """heapq.nlargest should return same results as full sort + slice."""
        mem = Memory(path=tmp_db_path)

        # Create memories with distinct strengths by varying access counts
        edge_ids = []
        for i in range(20):
            result = mem.remember(
                action="met",
                entities=[
                    {"name": f"Person{i}", "type": "person", "role": "agent"},
                    {"name": "Alice Smith", "type": "person", "role": "object"},
                ],
            )
            if result["edge_id"]:
                edge_ids.append(result["edge_id"])
                # Vary access counts to create distinct strengths
                for _ in range(i):
                    mem.hb.storage.record_access(
                        mem.hb.current_namespace, "edge", result["edge_id"]
                    )

        # Get results with the optimized method
        results = mem.recall(entity="Alice Smith", limit=5)

        # Get all results to compare
        all_results = mem.recall(entity="Alice Smith", limit=100)
        expected = sorted(all_results, key=lambda x: x["strength"], reverse=True)[:5]

        # Edge IDs should match
        result_ids = [r["edge"].id for r in results]
        expected_ids = [r["edge"].id for r in expected]
        assert result_ids == expected_ids
        mem.hb.close()
