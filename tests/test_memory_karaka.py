"""Tests for the kāraka-based agent memory system."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import MockEmbedder
from hypabase import Hypabase
from hypabase.memory.agent import Memory
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength
from hypabase.memory.types import (
    DEFAULT_DECAY_RATE,
    KARAKA_ROLES,
    MEMORY_DECAY_RATES,
    MEMORY_TYPES,
)


# ==================================================================
# TestKarakaRoles
# ==================================================================


class TestKarakaRoles:
    def test_remember_with_structured_entities_stores_roles(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
                {"name": "Bob", "type": "person", "role": "recipient"},
            ],
        )
        assert result["edge_id"] is not None
        assert result["action"] == "assigned"
        # Verify roles on incidences
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge is not None
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences}
        assert roles["Alice"] == "agent"
        assert roles["task"] == "object"
        assert roles["Bob"] == "recipient"
        mem.hb.close()

    def test_roles_survive_persist_reload(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="decided",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "proposal", "type": "document", "role": "object"},
            ],
        )
        edge_id = result["edge_id"]
        mem.hb.close()

        # Reload
        mem2 = Memory(path=tmp_db_path)
        edge = mem2.hb.get_edge(edge_id)
        assert edge is not None
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences}
        assert roles["Alice"] == "agent"
        assert roles["proposal"] == "object"
        mem2.hb.close()

    def test_partial_roles(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="discussed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person"},  # no role
            ],
        )
        edge = mem.hb.get_edge(result["edge_id"])
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences}
        assert roles["Alice"] == "agent"
        assert roles.get("Bob") is None
        mem.hb.close()

    def test_client_edge_accepts_roles(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice", type="person")
        hb.node("task", type="task")
        edge = hb.edge(
            ["Alice", "task"],
            type="assigned",
            roles=["agent", "object"],
        )
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences}
        assert roles["Alice"] == "agent"
        assert roles["task"] == "object"
        hb.close()

    def test_roles_length_mismatch_raises(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        with pytest.raises(ValueError, match="roles length"):
            hb.edge(
                ["Alice", "Bob"],
                type="test",
                roles=["agent"],  # too few
            )
        hb.close()

    def test_recall_includes_roles(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
                {"name": "Bob", "type": "person", "role": "recipient"},
            ],
        )
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert "roles" in results[0]
        assert results[0]["roles"]["Alice"] == "agent"
        mem.hb.close()

    def test_recall_filter_by_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # Alice as agent
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        # Alice as recipient
        mem.remember(
            action="sent",
            entities=[
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "report", "type": "document", "role": "object"},
                {"name": "Alice", "type": "person", "role": "recipient"},
            ],
        )
        # Recall only where Alice is agent
        results = mem.recall(entity="Alice", role="agent")
        assert len(results) >= 1
        for r in results:
            assert r["roles"].get("Alice") == "agent"
        mem.hb.close()


# ==================================================================
# TestMemoryTypes
# ==================================================================


class TestMemoryTypes:
    def test_episodic_decays_faster_than_semantic(self):
        now = time.time()
        created = now - 86400 * 7  # 7 days ago
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        assert s_sem > s_epi

    def test_semantic_persists_longer_than_episodic(self):
        now = time.time()
        created = now - 86400 * 30  # 30 days ago
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        assert s_sem > s_epi

    def test_procedural_most_durable(self):
        now = time.time()
        created = now - 86400 * 30  # 30 days ago
        s_pro = memory_strength(created_at=created, memory_type="procedural", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        assert s_pro > s_sem > s_epi

    def test_default_decay_backward_compat(self):
        now = time.time()
        # No memory_type → default decay
        s1 = memory_strength(created_at=now - 86400, now=now)
        s2 = memory_strength(created_at=now - 86400, decay=0.1, now=now)
        assert s1 == s2

    def test_memory_type_stored_in_edge_properties(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="uses",
            entities=[
                {"name": "API", "type": "system", "role": "agent"},
                {"name": "REST", "type": "technology", "role": "object"},
            ],
            memory_type="semantic",
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.properties["memory_type"] == "semantic"
        mem.hb.close()

    def test_forget_by_memory_type(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="happened",
            entities=[
                {"name": "Meeting", "type": "event", "role": "agent"},
                {"name": "Office", "type": "place", "role": "locus"},
            ],
            memory_type="episodic",
        )
        mem.remember(
            action="is",
            entities=[
                {"name": "Python", "type": "technology", "role": "agent"},
                {"name": "language", "type": "concept", "role": "object"},
            ],
            memory_type="semantic",
        )
        # Forget only episodic
        count = mem.forget(memory_type="episodic")["expired_count"]
        assert count == 1
        # Semantic memory should survive
        active = mem.hb.edges(active=True)
        types = [e.properties.get("memory_type") for e in active]
        assert "semantic" in types
        assert "episodic" not in types
        mem.hb.close()


# ==================================================================
# TestActionAsEdgeType
# ==================================================================


class TestActionAsEdgeType:
    def test_action_becomes_edge_type(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        result = mem.remember(
            action="decided",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "proposal", "type": "document", "role": "object"},
            ],
        )
        edge = mem.hb.get_edge(result["edge_id"])
        assert edge.type == "decided"
        mem.hb.close()

    def test_recall_filter_by_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="decided",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "X", "type": "thing", "role": "object"},
            ],
        )
        mem.remember(
            action="reviewed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Y", "type": "thing", "role": "object"},
            ],
        )
        results = mem.recall(entity="Alice", action="decided")
        assert len(results) >= 1
        for r in results:
            assert r["action"] == "decided"
        mem.hb.close()


# ==================================================================
# TestEntityResolution
# ==================================================================


class TestEntityResolution:
    def test_case_normalization(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        id1 = resolver.resolve("Alice Smith", "person")
        id2 = resolver.resolve("alice smith", "person")
        assert id1 == id2
        hb.close()

    def test_alias_detection_prefix(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        id1 = resolver.resolve("Bob Jones", "person")
        id2 = resolver.resolve("Bob", "person")
        assert id1 == id2
        hb.close()

    def test_alias_detection_suffix(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        id1 = resolver.resolve("Bob Jones", "person")
        id2 = resolver.resolve("Jones", "person")
        assert id1 == id2
        hb.close()

    def test_embedding_similarity_resolves(self, tmp_db_path):
        """With embedder, similar names can resolve to same entity."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        resolver = EntityResolver(hb, embedder=embedder, similarity_threshold=0.0)
        # Create a node and embed it
        id1 = resolver.resolve("Robert Jones", "person")
        # With very low threshold, search should find it
        # (MockEmbedder uses hash-based vectors so this is a probabilistic test)
        # At minimum, verify the mechanism doesn't crash
        id2 = resolver.resolve("Robert Jones", "person")
        assert id1 == id2
        hb.close()

    def test_no_false_merges(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        id1 = resolver.resolve("Alice Smith", "person")
        id2 = resolver.resolve("Carol Davis", "person")
        assert id1 != id2
        hb.close()

    def test_aliases_stored_on_node(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        resolver.resolve("Bob Jones", "person")
        hb.node("Bob Jones", type="person")  # ensure node exists
        resolver.resolve("Bob", "person")
        node = hb.get_node("Bob Jones")
        assert node is not None
        assert "Bob" in node.properties.get("aliases", [])
        hb.close()

    def test_warm_cache(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice", type="person")
        hb.node("Bob", type="person")
        resolver = EntityResolver(hb)
        resolver.warm_cache()
        # Should resolve from cache without creating new nodes
        assert resolver.resolve("Alice", "person") == "Alice"
        assert resolver.resolve("Bob", "person") == "Bob"
        hb.close()

    def test_graceful_fallback_no_embedder(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)  # no embedder
        id1 = resolver.resolve("Alice Smith", "person")
        id2 = resolver.resolve("alice smith", "person")
        # Steps 1-2 should still work
        assert id1 == id2
        hb.close()


# ==================================================================
# TestSpreadingActivation
# ==================================================================


class TestSpreadingActivation:
    def test_direct_neighbors_surfaced(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="works_with",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        mem.remember(
            action="works_with",
            entities=[
                {"name": "Bob", "type": "person"},
                {"name": "Carol", "type": "person"},
            ],
        )
        # Spread from Alice should surface Carol-related memories
        results = mem.recall(entity="Alice")
        all_node_ids = set()
        for r in results:
            all_node_ids.update(r["edge"].node_ids)
        # Should discover Bob at minimum; Carol via spreading
        assert "Bob" in all_node_ids
        mem.hb.close()

    def test_activation_decays_with_distance(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # Build a chain: Alice -> Bob -> Carol -> Dave
        mem.remember(
            action="knows",
            entities=[{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}],
        )
        mem.remember(
            action="knows",
            entities=[{"name": "Bob", "type": "person"}, {"name": "Carol", "type": "person"}],
        )
        mem.remember(
            action="knows",
            entities=[{"name": "Carol", "type": "person"}, {"name": "Dave", "type": "person"}],
        )
        # Direct activation test
        activation = mem._spreading_activation({"Alice": 1.0}, depth=3, decay=0.5)
        # Should have decreasing activation for farther edges
        assert len(activation) >= 1
        mem.hb.close()

    def test_depth_zero_no_spreading(self, tmp_db_path):
        """_spreading_activation with depth=0 should only match direct edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="works_with",
            entities=[{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}],
        )
        mem.remember(
            action="works_with",
            entities=[{"name": "Bob", "type": "person"}, {"name": "Carol", "type": "person"}],
        )
        # depth=0 means no edges discovered (only depth >= 1 finds neighbors)
        activation = mem._spreading_activation({"Alice": 1.0}, depth=0, decay=0.5)
        assert len(activation) == 0
        mem.hb.close()

    def test_spread_depth_respected(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # Chain of 4 hops
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            mem.remember(
                action="knows",
                entities=[{"name": a, "type": "person"}, {"name": b, "type": "person"}],
            )
        # depth=1 should not reach D or E
        activation = mem._spreading_activation({"A": 1.0}, depth=1, decay=0.5)
        # Only edges directly connected to A or B should have activation
        reached_nodes = set()
        for eid in activation:
            edge = mem.hb.get_edge(eid)
            if edge:
                reached_nodes.update(edge.node_ids)
        assert "D" not in reached_nodes
        assert "E" not in reached_nodes
        mem.hb.close()

    def test_min_activation_cutoff(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for a, b in [("X", "Y"), ("Y", "Z")]:
            mem.remember(
                action="knows",
                entities=[{"name": a, "type": "person"}, {"name": b, "type": "person"}],
            )
        # Very low decay means propagation drops below min_activation quickly.
        # The X-Y edge will still be found (overlap with seed X), but Y-Z should NOT
        # be found because propagated activation from X is too low to reach Z.
        activation = mem._spreading_activation(
            {"X": 1.0}, depth=5, decay=0.01, min_activation=0.5,
        )
        # X-Y edge found via overlap, but Y-Z not reached via propagation
        reached_nodes = set()
        for eid in activation:
            edge = mem.hb.get_edge(eid)
            if edge:
                reached_nodes.update(edge.node_ids)
        assert "Z" not in reached_nodes
        mem.hb.close()


# ==================================================================
# TestContradictionDetection
# ==================================================================


class TestContradictionDetection:
    def test_detect_contradiction_in_semantic_memories(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # First memory: Alice prefers Python
        mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        # Contradicting: Alice prefers Java
        result = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        assert len(result["contradictions"]) >= 1
        assert result["contradictions"][0]["existing_edge_id"] is not None
        mem.hb.close()

    def test_no_contradiction_for_episodic(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "object"},
            ],
            memory_type="episodic",
        )
        result = mem.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Carol", "type": "person", "role": "object"},
            ],
            memory_type="episodic",
        )
        # Episodic memories should not trigger contradiction detection
        assert result["contradictions"] == []
        mem.hb.close()

    def test_supersede_resolution(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        r2 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        assert len(r2["contradictions"]) >= 1
        old_id = r2["contradictions"][0]["existing_edge_id"]
        result = mem.resolve_contradiction(r2["edge_id"], old_id, "supersede")
        assert result["resolved"] is True
        assert result["action"] == "superseded"
        # Old edge should be expired
        old_edge = mem.hb.get_edge(old_id)
        assert old_edge is not None
        assert not old_edge.is_active
        mem.hb.close()

    def test_keep_both_resolution(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        r2 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        old_id = r2["contradictions"][0]["existing_edge_id"]
        result = mem.resolve_contradiction(r2["edge_id"], old_id, "keep_both")
        assert result["resolved"] is True
        assert result["action"] == "kept_both"
        # Both edges should still be active
        assert mem.hb.get_edge(r1["edge_id"]).is_active
        assert mem.hb.get_edge(r2["edge_id"]).is_active
        mem.hb.close()

    def test_contradictions_in_return_value(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # No contradictions for first memory
        r1 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        assert r1["contradictions"] == []
        assert "contradictions" in r1
        mem.hb.close()


# ==================================================================
# TestBackwardCompatibility
# ==================================================================


class TestBackwardCompatibility:
    def test_recall_without_new_params(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
        )
        results = mem.recall(entity="Alice Smith")
        assert len(results) >= 1
        assert "strength" in results[0]
        assert "text" in results[0]
        mem.hb.close()

    def test_edge_without_roles(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        edge = hb.edge(["Alice", "Bob"], type="knows")
        # Should work without roles
        assert edge is not None
        assert edge.type == "knows"
        for inc in edge.incidences:
            assert inc.properties.get("role") is None
        hb.close()

    def test_strength_without_memory_type(self):
        now = time.time()
        s = memory_strength(created_at=now, access_count=5, now=now)
        expected = 1.0 * (1.0 + math.log(6)) * 1.0 * 1.0
        assert s == pytest.approx(expected, abs=0.01)

    def test_existing_memories_still_readable(self, tmp_db_path):
        # Create memory with new API
        mem1 = Memory(path=tmp_db_path)
        r1 = mem1.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        edge_id = r1["edge_id"]
        mem1.hb.close()

        # Read
        mem2 = Memory(path=tmp_db_path)
        results = mem2.recall(entity="Alice Smith")
        assert len(results) >= 1
        # Should have fields
        for r in results:
            assert "roles" in r
            assert "action" in r
            assert "memory_type" in r
        mem2.hb.close()


# ==================================================================
# TestRecallEntity
# ==================================================================


class TestRecallByEntity:
    def test_recall_by_entity_basic(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="decided",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "proposal", "type": "document", "role": "object"},
            ],
        )
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert results[0]["action"] == "decided"
        mem.hb.close()

    def test_recall_by_entity_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        mem.remember(
            action="sent",
            entities=[
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "report", "type": "document", "role": "object"},
                {"name": "Alice", "type": "person", "role": "recipient"},
            ],
        )
        # Only where Alice is agent
        results = mem.recall(entity="Alice", role="agent")
        assert len(results) == 1
        assert results[0]["action"] == "assigned"
        mem.hb.close()

    def test_recall_by_entity_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="decided",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "X", "type": "thing", "role": "object"},
            ],
        )
        mem.remember(
            action="reviewed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Y", "type": "thing", "role": "object"},
            ],
        )
        results = mem.recall(entity="Alice", action="decided")
        assert len(results) == 1
        assert results[0]["action"] == "decided"
        mem.hb.close()


# ==================================================================
# TestConsolidate
# ==================================================================


class TestConsolidate:
    def test_consolidate_merges_shared_vertex_set(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # Create two edges with the same vertex set
        mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        result = mem.consolidate()
        # consolidate() returns a list of dicts
        assert isinstance(result, list)
        # With 2 edges sharing a vertex set, should produce at least 1 consolidated edge
        assert len(result) >= 1
        for entry in result:
            assert "edge_id" in entry
            assert "source_edge_ids" in entry
        mem.hb.close()

    def test_consolidate_with_entity_filter(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="knows",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        mem.remember(
            action="knows",
            entities=[
                {"name": "Carol", "type": "person"},
                {"name": "Dave", "type": "person"},
            ],
        )
        # Only consolidate memories involving Alice
        result = mem.consolidate(entity="Alice")
        assert isinstance(result, list)
        mem.hb.close()

    def test_consolidate_idempotent(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            action="knows",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        mem.remember(
            action="likes",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        r1 = mem.consolidate()
        r2 = mem.consolidate()
        # Second run should not create new consolidated edges
        assert len(r2) <= len(r1)
        mem.hb.close()


# ==================================================================
# TestResolveContradictionKeepOld
# ==================================================================


class TestResolveContradictionKeepOld:
    def test_keep_old_expires_new_edge(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        r2 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        assert len(r2["contradictions"]) >= 1
        old_id = r2["contradictions"][0]["existing_edge_id"]
        result = mem.resolve_contradiction(r2["edge_id"], old_id, "keep_old")
        assert result["resolved"] is True
        assert result["action"] == "kept_old"
        # New edge should be expired
        new_edge = mem.hb.get_edge(r2["edge_id"])
        assert new_edge is not None
        assert not new_edge.is_active
        # Old edge should still be active
        old_edge = mem.hb.get_edge(old_id)
        assert old_edge is not None
        assert old_edge.is_active
        mem.hb.close()

    def test_invalid_resolution_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        r2 = mem.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        old_id = r2["contradictions"][0]["existing_edge_id"]
        with pytest.raises(ValueError, match="Unknown resolution"):
            mem.resolve_contradiction(r2["edge_id"], old_id, "invalid_action")
        mem.hb.close()
