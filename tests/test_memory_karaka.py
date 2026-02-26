"""Tests for the karaka-based agent memory system."""

from __future__ import annotations

import math
import time

import pytest

from hypabase import Hypabase
from hypabase.memory.agent import Memory
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength

# ==================================================================
# TestKarakaRoles
# ==================================================================


class TestKarakaRoles:
    def test_remember_with_structured_entities_stores_roles(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(assigned :subject Alice :object task :recipient Bob)")
        # Verify roles on incidences via recall
        results = mem.recall(entity="Alice", action="assigned")
        assert len(results) >= 1
        edge = results[0]["edge"]
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences if inc.node_id}
        assert roles["Alice"] == "subject"
        assert roles["task"] == "object"
        assert roles["Bob"] == "recipient"
        mem.hb.close()

    def test_roles_survive_persist_reload(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(decided :subject Alice :object proposal)")
        mem.hb.close()

        mem2 = Memory(path=tmp_db_path)
        results = mem2.recall(entity="Alice", action="decided")
        assert len(results) >= 1
        edge = results[0]["edge"]
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences if inc.node_id}
        assert roles["Alice"] == "subject"
        assert roles["proposal"] == "object"
        mem2.hb.close()

    def test_client_edge_accepts_roles(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice", type="person")
        hb.node("task", type="task")
        edge = hb.edge(
            ["Alice", "task"],
            type="assigned",
            roles=["subject", "object"],
        )
        roles = {inc.node_id: inc.properties.get("role") for inc in edge.incidences}
        assert roles["Alice"] == "subject"
        assert roles["task"] == "object"
        hb.close()

    def test_roles_length_mismatch_raises(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        with pytest.raises(ValueError, match="roles length"):
            hb.edge(
                ["Alice", "Bob"],
                type="test",
                roles=["subject"],  # too few
            )
        hb.close()

    def test_recall_includes_roles(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(assigned :subject Alice :object task :recipient Bob)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert "roles" in results[0]
        assert results[0]["roles"]["Alice"] == "subject"
        mem.hb.close()

    def test_recall_filter_by_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        # Alice as agent
        mem.remember("(assigned :subject Alice :object task)")
        # Alice as recipient
        mem.remember("(sent :subject Bob :object report :recipient Alice)")
        # Recall only where Alice is agent
        results = mem.recall(entity="Alice", role="subject")
        assert len(results) >= 1
        for r in results:
            assert r["roles"].get("Alice") == "subject"
        mem.hb.close()


# ==================================================================
# TestMemoryTypes
# ==================================================================


class TestMemoryTypes:
    def test_episodic_decays_faster_than_semantic(self):
        now = time.time()
        created = now - 86400 * 7
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        assert s_sem > s_epi

    def test_semantic_persists_longer_than_episodic(self):
        now = time.time()
        created = now - 86400 * 30
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        assert s_sem > s_epi

    def test_procedural_most_durable(self):
        now = time.time()
        created = now - 86400 * 30
        s_pro = memory_strength(created_at=created, memory_type="procedural", now=now)
        s_sem = memory_strength(created_at=created, memory_type="semantic", now=now)
        s_epi = memory_strength(created_at=created, memory_type="episodic", now=now)
        assert s_pro > s_sem > s_epi

    def test_default_decay_backward_compat(self):
        now = time.time()
        s1 = memory_strength(created_at=now - 86400, now=now)
        s2 = memory_strength(created_at=now - 86400, decay=0.1, now=now)
        assert s1 == s2

    def test_memory_type_stored_in_edge_properties(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(uses :subject API :object REST :memory_type semantic)")
        results = mem.recall(entity="API", memory_type="semantic")
        assert len(results) >= 1
        assert results[0]["edge"].properties["memory_type"] == "semantic"
        mem.hb.close()

    def test_forget_by_entity(self, tmp_db_path):
        """forget(entity=...) expires memories involving that entity."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(happened :subject Meeting :object Office :memory_type episodic)")
        mem.remember("(is :subject Python :object language :memory_type semantic)")
        count = mem.forget(entity="Meeting")["expired_count"]
        assert count == 1
        active = mem.hb.edges(active=True)
        types = [e.properties.get("memory_type") for e in active]
        assert "semantic" in types
        mem.hb.close()


# ==================================================================
# TestActionAsEdgeType
# ==================================================================


class TestActionAsEdgeType:
    def test_action_becomes_edge_type(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(decided :subject Alice :object proposal)")
        results = mem.recall(entity="Alice", action="decided")
        assert len(results) >= 1
        assert results[0]["edge"].type == "decided"
        mem.hb.close()

    def test_recall_filter_by_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(decided :subject Alice :object X)")
        mem.remember("(reviewed :subject Alice :object Y)")
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
        """Normalized cache resolves case-insensitive matches."""
        hb = Hypabase(tmp_db_path)
        hb.node("Alice Smith", type="person")
        resolver = EntityResolver(hb)
        resolver.warm_cache()
        resolved = resolver.resolve("alice smith")
        assert resolved == "Alice Smith"
        hb.close()

    def test_no_false_merges(self, tmp_db_path):
        """Different names should not be merged by cache alone."""
        hb = Hypabase(tmp_db_path)
        hb.node("Alice Smith", type="person")
        hb.node("Carol Davis", type="person")
        resolver = EntityResolver(hb)
        resolver.warm_cache()
        id1 = resolver.resolve("Alice Smith")
        id2 = resolver.resolve("Carol Davis")
        assert id1 != id2
        hb.close()

    def test_warm_cache(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("Alice", type="person")
        hb.node("Bob", type="person")
        resolver = EntityResolver(hb)
        resolver.warm_cache()
        assert resolver.resolve("Alice") == "Alice"
        assert resolver.resolve("Bob") == "Bob"
        hb.close()

    def test_new_name_returns_unchanged(self, tmp_db_path):
        """Unknown names are returned as-is (no embedding search at write time)."""
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        resolver.warm_cache()
        result = resolver.resolve("Brand New Entity")
        assert result == "Brand New Entity"
        hb.close()

    def test_register_updates_cache(self, tmp_db_path):
        """register() makes aliases resolvable."""
        hb = Hypabase(tmp_db_path)
        resolver = EntityResolver(hb)
        resolver.register("machine learning", ["ML", "ml"])
        assert resolver.resolve("ML") == "machine learning"
        assert resolver.resolve("ml") == "machine learning"
        hb.close()


# ==================================================================
# TestBackwardCompatibility
# ==================================================================


class TestBackwardCompatibility:
    def test_recall_without_new_params(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones" :locus park)')
        results = mem.recall(entity="Alice Smith")
        assert len(results) >= 1
        assert "strength" in results[0]
        assert "text" in results[0]
        mem.hb.close()

    def test_edge_without_roles(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        edge = hb.edge(["Alice", "Bob"], type="knows")
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
        mem1 = Memory(path=tmp_db_path)
        mem1.remember('(met :subject "Alice Smith" :object "Bob Jones")')
        mem1.hb.close()

        mem2 = Memory(path=tmp_db_path)
        results = mem2.recall(entity="Alice Smith")
        assert len(results) >= 1
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
        mem.remember("(decided :subject Alice :object proposal)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        assert results[0]["action"] == "decided"
        mem.hb.close()

    def test_recall_by_entity_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(assigned :subject Alice :object task)")
        mem.remember("(sent :subject Bob :object report :recipient Alice)")
        results = mem.recall(entity="Alice", role="subject")
        assert len(results) == 1
        assert results[0]["action"] == "assigned"
        mem.hb.close()

    def test_recall_by_entity_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(decided :subject Alice :object X)")
        mem.remember("(reviewed :subject Alice :object Y)")
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
        mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
        mem.remember("(likes :subject Alice :object Python :memory_type semantic)")
        result = mem.consolidate()
        assert isinstance(result, list)
        assert len(result) >= 1
        # Edge consolidation results have edge_id and source_edge_ids
        edge_results = [r for r in result if "edge_id" in r]
        for entry in edge_results:
            assert "edge_id" in entry
            assert "source_edge_ids" in entry
        mem.hb.close()

    def test_consolidate_with_entity_filter(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(knows :subject Alice :object Bob)")
        mem.remember("(knows :subject Carol :object Dave)")
        result = mem.consolidate(entity="Alice")
        assert isinstance(result, list)
        mem.hb.close()

    def test_consolidate_idempotent(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(knows :subject Alice :object Bob)")
        mem.remember("(likes :subject Alice :object Bob)")
        r1 = mem.consolidate()
        r2 = mem.consolidate()
        assert len(r2) <= len(r1)
        mem.hb.close()
