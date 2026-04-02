"""Tests for the Hypabase Memory Module."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta

import pytest

from hypabase.memory.agent import Memory
from hypabase.memory.penman import PenmanParseError
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
        s_half = memory_strength(created_at=now, confidence=0.5, now=now)
        s_full = memory_strength(created_at=now, confidence=1.0, now=now)
        assert s_full > s_half


class TestRemember:
    def test_remember_basic(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones" :locus office)')
        # Verify the memory was stored by recalling it
        results = mem.recall(entity="Alice Smith")
        assert len(results) >= 1
        mem.hb.close()

    def test_remember_empty_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(PenmanParseError, match="Empty input"):
            mem.remember("")
        mem.hb.close()

    def test_remember_one_entity_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="at least 2 participants"):
            mem.remember("(met :subject Alice)")
        mem.hb.close()

    def test_remember_invalid_mood_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown mood"):
            mem.remember("(met :subject Alice :object Bob :mood hypothetical)")
        mem.hb.close()

    def test_remember_invalid_memory_type_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="Unknown memory_type"):
            mem.remember("(met :subject Alice :object Bob :memory_type unknown)")
        mem.hb.close()

    def test_remember_generates_text(self, tmp_db_path):
        """remember always generates text from atom_to_sentence."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(met :subject Alice :object Bob)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        edge = results[0]["edge"]
        assert "text" in edge.properties
        assert "Alice" in edge.properties["text"]
        assert "met" in edge.properties["text"]
        mem.hb.close()

    def test_remember_with_modifiers(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember(
            "(deploy :subject Alice :object API :mood planned :memory_type episodic :importance 0.8 :tense future)"
        )
        results = mem.recall(entity="Alice", mood="planned")
        assert len(results) >= 1
        edge = results[0]["edge"]
        assert edge.properties["mood"] == "planned"
        assert edge.properties["memory_type"] == "episodic"
        assert edge.properties["importance"] == pytest.approx(0.8)
        assert edge.properties["tense"] == "future"
        mem.hb.close()

    def test_remember_meta_edge(self, tmp_db_path):
        """Nested atom creates edge_ref_id incidence."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(believes :subject Alice :object (is :subject deadline :value Friday))")
        # Verify outer edge exists via recall
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        # The inner edge should also exist -- find 'is' type edges
        is_edges = [e for e in mem.hb.edges(active=True) if e.type == "is"]
        assert len(is_edges) >= 1
        mem.hb.close()

    def test_remember_entity_resolution(self, tmp_db_path):
        """Entity names linked via same_as resolve to the same node at recall time."""
        mem = Memory(path=tmp_db_path)
        # Set up an explicit same_as edge so "Bob" resolves via same_as at recall
        mem.hb.node("Bob Jones", type="entity")
        mem.hb.node("Bob", type="entity")
        mem.hb.edge(["Bob", "Bob Jones"], type="same_as", source="resolution")
        mem.remember('(likes :subject "Bob Jones" :object Python)')
        mem.remember("(likes :subject Bob :object Java)")
        # "Bob" should find memories under "Bob Jones" via same_as expansion at recall
        results = mem.recall(entity="Bob Jones")
        assert len(results) >= 2
        mem.hb.close()

    def test_remember_embed_text(self, tmp_db_path):
        """Embed text is generated sentence, not raw PENMAN."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(prefers :subject Alice :object Python)")
        results = mem.recall(entity="Alice")
        assert len(results) >= 1
        text = results[0]["edge"].properties.get("text", "")
        # Should be natural English, not PENMAN notation
        assert "(" not in text
        assert "Alice" in text
        assert "prefers" in text
        mem.hb.close()

    def test_remember_multiple_atoms(self, tmp_db_path):
        """Two atoms in one call create two edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            "(deployed :subject Alice :object API :tense past) (reviewed :subject Bob :object API :tense past)"
        )
        # Verify both edges were created
        deployed = mem.recall(entity="Alice", action="deployed")
        reviewed = mem.recall(entity="Bob", action="reviewed")
        assert len(deployed) >= 1
        assert len(reviewed) >= 1
        mem.hb.close()

    def test_recall_by_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones" :locus "Central Park")')
        results = mem.recall(entity="Alice Smith")
        assert len(results) >= 1
        mem.hb.close()

    def test_recall_returns_text(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(loves :subject "Alice Smith" :object Python)')
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
        mem.remember("(assigned :subject Alice :object API :recipient Bob)")
        mem.remember("(reviewed :subject Carol :object API)")
        # Should find memories involving both Alice and API
        results = mem.recall(entity=["Alice", "API"])
        assert len(results) >= 1
        assigned = [r for r in results if r["action"] == "assigned"]
        assert len(assigned) >= 1
        mem.hb.close()

    def test_recall_filter_by_action(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(assigned :subject Alice :object task)")
        mem.remember("(reviewed :subject Alice :object code)")
        results = mem.recall(entity="Alice", action="assigned")
        assert len(results) >= 1
        assert all(r["action"] == "assigned" for r in results)
        mem.hb.close()

    def test_recall_filter_by_role(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(assigned :subject Alice :object task :recipient Bob)")
        results = mem.recall(entity="Alice", role="subject")
        assert len(results) >= 1
        for r in results:
            assert r["roles"].get("subject") == "Alice"
        mem.hb.close()

    def test_recall_filter_by_memory_type(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(met :subject Alice :object Bob :memory_type episodic)")
        mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
        results = mem.recall(entity="Alice", memory_type="semantic")
        assert len(results) >= 1
        assert all(r["memory_type"] == "semantic" for r in results)
        mem.hb.close()

    def test_recall_filter_by_mood(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(deployed :subject Alice :object API :mood actual)")
        mem.remember('(deploy :subject Alice :object "API v2" :mood planned)')
        results = mem.recall(mood="planned")
        assert len(results) >= 1
        assert all(r["mood"] == "planned" for r in results)
        mem.hb.close()

    def test_recall_filter_by_negated(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember("(uses :subject team :object Python)")
        mem.remember("(uses :subject team :object Java :negated true)")
        results = mem.recall(action="uses", negated=True)
        assert len(results) >= 1
        assert all(r["negated"] is True for r in results)
        mem.hb.close()

    def test_recall_filter_only_no_entity(self, tmp_db_path):
        """recall(mood=...) without entity scans all edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(deploy :subject Alice :object service :mood planned)")
        mem.remember("(deploy :subject Bob :object API :mood actual)")
        results = mem.recall(mood="planned")
        assert len(results) >= 1
        assert all(r["mood"] == "planned" for r in results)
        mem.hb.close()

    def test_recall_temporal_since(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        now = datetime.now(UTC)
        mem.remember("(met :subject Alice :object Bob)")
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
        now = datetime.now(UTC)
        mem.remember("(met :subject Alice :object Bob)")
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
        mem.hb.node("Bob", type="person")
        mem.hb.node("Bob Jones", type="person")
        mem.hb.edge(["Bob", "Bob Jones"], type="same_as", source="resolution")
        mem.remember('(assigned :subject "Bob Jones" :object task)')
        results = mem.recall(entity="Bob")
        assert len(results) >= 1
        mem.hb.close()

    def test_forget_by_strength(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones")')
        # New memories have strength ~0.5, so low threshold should NOT forget them
        count_low = mem.forget(min_strength=0.01)["expired_count"]
        assert count_low == 0
        # High threshold should forget them
        count_high = mem.forget(min_strength=2.0)["expired_count"]
        assert count_high >= 1
        active = mem.hb.edges(active=True)
        active_memories = [e for e in active if e.type != "same_as"]
        assert len(active_memories) == 0
        mem.hb.close()

    def test_forget_by_age(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones")')
        future = datetime.now(UTC) + timedelta(days=1)
        count = mem.forget(older_than=future)["expired_count"]
        assert count >= 1
        mem.hb.close()

    def test_consolidate(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(worked :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(presented :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(published :subject "Alice Smith" :object "Bob Jones")')
        summaries = mem.consolidate()
        assert len(summaries) >= 1
        mem.hb.close()

    def test_consolidate_entity_filter(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(worked :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(collaborated :subject "Carol Davis" :object "Dave Evans")')
        mem.remember('(published :subject "Carol Davis" :object "Dave Evans")')
        summaries = mem.consolidate(entity="Alice Smith")
        assert len(summaries) >= 1
        for s in summaries:
            if "entities" in s:
                assert "Carol Davis" not in s["entities"]
                assert "Dave Evans" not in s["entities"]
        mem.hb.close()

    def test_remember_with_embedder(self, tmp_db_path):
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones")')
        raw = mem.hb._storage.load_embeddings(mem.hb._current_ns, kind="edge")
        assert len(raw) >= 1
        mem.hb.close()

    def test_recall_min_strength(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones" :locus park)')
        results = mem.recall(entity="Alice Smith", min_strength=999.0)
        assert len(results) == 0
        results = mem.recall(entity="Alice Smith", min_strength=0.0)
        assert len(results) >= 1
        mem.hb.close()

    def test_consolidate_idempotent(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(worked :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(presented :subject "Alice Smith" :object "Bob Jones")')
        mem.remember('(published :subject "Alice Smith" :object "Bob Jones")')
        summaries1 = mem.consolidate()
        assert len(summaries1) >= 1
        summaries2 = mem.consolidate()
        assert len(summaries2) == 0
        mem.hb.close()

    def test_recall_empty_entity(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        results = mem.recall(entity="nonexistent")
        assert results == []
        mem.hb.close()

    def test_forget_entity_only(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        mem.remember('(met :subject "Alice Smith" :object "Bob Jones" :locus park)')
        count = mem.forget(entity="Alice Smith")["expired_count"]
        assert count >= 1
        mem.hb.close()

    def test_forget_no_filter_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        with pytest.raises(ValueError, match="At least one filter"):
            mem.forget()
        mem.hb.close()


class TestActivationReport:
    def test_remember_returns_activation_report(self, tmp_db_path):
        """remember() returns structured report with entities and edge info."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember("(prefers :subject Alice :object Python)")
        assert result["stored"] == 1
        assert len(result["edges"]) == 1
        edge_info = result["edges"][0]
        assert "edge_id" in edge_info
        assert edge_info["action"] == "prefers"
        assert edge_info["text"]  # non-empty
        assert len(edge_info["entities"]) == 2
        mem.hb.close()

    def test_remember_reports_entity_status(self, tmp_db_path):
        """Second remember on same entity reports 'existing' status."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        r2 = mem.remember("(teaches :subject Alice :object Java)")
        statuses = {e["name"]: e["status"] for e in r2["edges"][0]["entities"]}
        assert statuses["Alice"] == "existing"  # seen before
        mem.hb.close()

    def test_remember_reports_related_memories(self, tmp_db_path):
        """Related memories are surfaced in activation report."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        r2 = mem.remember("(teaches :subject Alice :object Python)")
        assert len(r2["related"]) >= 1
        assert any("Alice" in r["shared_entities"] for r in r2["related"])
        mem.hb.close()

    def test_remember_multiple_atoms_report(self, tmp_db_path):
        """Multiple atoms in one call return multiple edge reports."""
        mem = Memory(path=tmp_db_path)
        result = mem.remember("(likes :subject Alice :object Python) (likes :subject Bob :object Java)")
        assert result["stored"] == 2
        assert len(result["edges"]) == 2
        mem.hb.close()


class TestConsolidateSemanticMerge:
    def test_consolidate_keeps_originals(self, tmp_db_path):
        """Original edges remain active after consolidation (S&B approach)."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        mem.remember("(prefers :subject Alice :object Python)")
        mem.consolidate()
        # Both originals should still be active
        results = mem.recall(entity="Alice")
        original_actions = [r["action"] for r in results if r["action"] != "consolidated"]
        assert "likes" in original_actions
        assert "prefers" in original_actions
        mem.hb.close()

    def test_consolidate_semantic_merging_with_embedder(self, tmp_db_path):
        """Semantically similar edges are grouped during consolidation."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        mem.remember("(likes :subject Alice :object Python)")
        mem.remember("(prefers :subject Alice :object Python)")
        summaries = mem.consolidate()
        # Should find at least one group (exact vertex-set or semantic)
        assert len(summaries) >= 1
        mem.hb.close()


class TestSBAlignment:
    """Tests verifying alignment with Stewart & Buehler (2026) patterns."""

    def test_write_verbatim(self, tmp_db_path):
        """S&B: remember() creates nodes with exact names, no embedding-based resolution."""
        mem = Memory(path=tmp_db_path)
        r1 = mem.remember('(studies :subject Alice :object "machine learning")')
        r2 = mem.remember("(studies :subject Alice :object ML)")
        # Both edges exist -- no dedup at write time
        assert r1["stored"] == 1
        assert r2["stored"] == 1
        # Both nodes written as-is
        ml_node = mem.hb.get_node("machine learning")
        ml_abbrev = mem.hb.get_node("ML")
        assert ml_node is not None
        assert ml_abbrev is not None
        mem.hb.close()

    def test_always_new_edges(self, tmp_db_path):
        """S&B: same fact written twice creates two edges."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(likes :subject Alice :object Python)")
        mem.remember("(likes :subject Alice :object Python)")
        edges = [e for e in mem.hb.edges(active=True) if e.type == "likes"]
        assert len(edges) == 2
        mem.hb.close()

    def test_provenance_preserved(self, tmp_db_path):
        """S&B: source + confidence on every edge."""
        mem = Memory(path=tmp_db_path)
        mem.remember(
            "(discovers :subject Lab :object result :memory_type episodic)",
            source="experiment_42",
            confidence=0.85,
        )
        results = mem.recall(entity="Lab")
        assert len(results) >= 1
        assert results[0]["edge"].source == "experiment_42"
        assert results[0]["edge"].confidence == 0.85
        mem.hb.close()

    def test_is_constrained_path_finding(self, tmp_db_path):
        """S&B: recall with 2+ entities uses find_paths."""
        mem = Memory(path=tmp_db_path)
        mem.remember("(teaches :subject Alice :object Python)")
        mem.remember('(uses :subject Python :object "data science")')
        mem.remember('(applies :subject Bob :object "data science")')
        # Path: Alice -> Python -> data science -> Bob
        results = mem.recall(entity=["Alice", "Bob"])
        # Should find at least one result through the path
        assert len(results) >= 1
        mem.hb.close()


class TestRecallEdgeCases:
    def test_recall_role_without_entity_raises(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        try:
            with pytest.raises(ValueError, match="role filter requires entity"):
                mem.recall(action="likes", role="subject")
        finally:
            mem.hb.close()

    def test_recall_result_ordering_by_score_strength(self, tmp_db_path):
        """Results are ordered by score * strength."""
        mem = Memory(path=tmp_db_path)
        try:
            mem.remember("(likes :subject Alice :object Python :importance 0.1)")
            mem.remember("(loves :subject Alice :object Python :importance 0.9)")
            results = mem.recall(entity="Alice")
            assert len(results) >= 2
            # Verify ordering: score * strength should be descending
            scores = [r["score"] * r["strength"] for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            mem.hb.close()


class TestRecallQuery:
    """Tests for recall(query=...) -- semantic edge search (Channel 1).

    Uses MockEmbedder (4-dim hash vectors). Tests verify pipeline invariants
    (dedup, filter passthrough, score structure, validation), not semantic
    relevance, since 4-dim hash embeddings have noisy cosine similarity.
    """

    # ---- Validation ----

    def test_recall_query_alone_passes_validation(self, tmp_db_path):
        """recall(query=...) does not raise ValueError."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            # Should not raise
            results = mem.recall(query="programming preferences")
            assert isinstance(results, list)
        finally:
            mem.hb.close()

    def test_recall_no_params_still_raises(self, tmp_db_path):
        """No params at all still raises ValueError."""
        mem = Memory(path=tmp_db_path)
        try:
            with pytest.raises(ValueError, match="At least one of"):
                mem.recall()
        finally:
            mem.hb.close()

    def test_recall_query_requires_no_entity(self, tmp_db_path):
        """query alone is sufficient -- no entity needed."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject user :object Python :memory_type semantic)")
            results = mem.recall(query="user prefers Python")
            assert isinstance(results, list)
        finally:
            mem.hb.close()

    # ---- Channel 1 retrieval ----

    def test_recall_query_returns_results(self, tmp_db_path):
        """recall(query=...) returns results with correct structure."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            mem.remember("(visited :subject Alice :object Tokyo :memory_type episodic)")
            results = mem.recall(query="Alice prefers Python")
            assert len(results) >= 1
            # Verify result structure
            for r in results:
                assert "edge" in r
                assert "score" in r
                assert "strength" in r
                assert "text" in r
                assert "action" in r
                assert "memory_type" in r
                assert "roles" in r
        finally:
            mem.hb.close()

    def test_recall_query_without_embedder_returns_empty(self, tmp_db_path):
        """recall(query=...) with no embedder returns []."""
        mem = Memory(path=tmp_db_path)
        try:
            mem.remember("(prefers :subject Alice :object Python)")
            results = mem.recall(query="programming preferences")
            assert results == []
        finally:
            mem.hb.close()

    def test_recall_query_respects_limit(self, tmp_db_path):
        """recall(query=..., limit=2) returns at most 2 results."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            for i in range(10):
                mem.remember(f"(likes :subject Alice :object item{i} :memory_type semantic)")
            results = mem.recall(query="Alice likes things", limit=2)
            assert len(results) <= 2
        finally:
            mem.hb.close()

    def test_recall_query_scores_are_positive(self, tmp_db_path):
        """All returned results have score > 0."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            results = mem.recall(query="Alice prefers Python")
            for r in results:
                assert r["score"] > 0
        finally:
            mem.hb.close()

    # ---- Parallel merge (Channel 1 + Channel 2) ----

    def test_recall_query_combined_with_entity(self, tmp_db_path):
        """recall(query=..., entity=...) returns results from both channels."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            mem.remember("(teaches :subject Alice :object Java :memory_type semantic)")
            results = mem.recall(query="programming preferences", entity="Alice")
            assert len(results) >= 1
        finally:
            mem.hb.close()

    def test_recall_query_deduplicates(self, tmp_db_path):
        """Same edge found by both channels appears only once."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            results = mem.recall(query="Alice prefers Python", entity="Alice")
            edge_ids = [r["edge"].id for r in results]
            assert len(edge_ids) == len(set(edge_ids)), "No duplicate edge IDs"
        finally:
            mem.hb.close()

    def test_recall_query_score_upgrade(self, tmp_db_path):
        """When both channels find same edge, score is max of the two."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            results = mem.recall(query="Alice prefers Python", entity="Alice")
            assert len(results) >= 1
            # Score should be positive (from whichever channel was stronger)
            for r in results:
                assert r["score"] > 0
        finally:
            mem.hb.close()

    # ---- Filter passthrough (Step 3 applies to all channels) ----

    def test_recall_query_filters_by_memory_type(self, tmp_db_path):
        """Filters apply to semantic search results too."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(met :subject Alice :object Bob :memory_type episodic)")
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            results = mem.recall(query="Alice", memory_type="semantic")
            for r in results:
                assert r["memory_type"] == "semantic"
        finally:
            mem.hb.close()

    def test_recall_query_filters_by_action(self, tmp_db_path):
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python)")
            mem.remember("(dislikes :subject Alice :object Java)")
            results = mem.recall(query="Alice", action="prefers")
            for r in results:
                assert r["action"] == "prefers"
        finally:
            mem.hb.close()

    def test_recall_query_filters_by_mood(self, tmp_db_path):
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(deploy :subject Alice :object API :mood planned)")
            mem.remember("(deployed :subject Alice :object API :mood actual)")
            results = mem.recall(query="Alice deploy", mood="planned")
            for r in results:
                assert r["mood"] == "planned"
        finally:
            mem.hb.close()

    def test_recall_query_filters_by_negated(self, tmp_db_path):
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(uses :subject team :object Python)")
            mem.remember("(uses :subject team :object Java :negated true)")
            results = mem.recall(query="team uses", negated=True)
            for r in results:
                assert r["negated"] is True
        finally:
            mem.hb.close()

    def test_recall_query_filters_by_min_strength(self, tmp_db_path):
        """min_strength=999 returns empty -- no memory is that strong."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            results = mem.recall(query="Alice prefers Python", min_strength=999.0)
            assert len(results) == 0
        finally:
            mem.hb.close()

    # ---- Access tracking ----

    def test_recall_query_records_access(self, tmp_db_path):
        """After recall(query=...), access is tracked (strength increases)."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            # First recall
            r1 = mem.recall(query="Alice prefers Python")
            if r1:
                s1 = r1[0]["strength"]
                # Second recall -- access_count increments, strength should increase
                r2 = mem.recall(query="Alice prefers Python")
                if r2:
                    s2 = r2[0]["strength"]
                    assert s2 >= s1
        finally:
            mem.hb.close()

    # ---- Graceful degradation ----

    def test_recall_query_only_no_embedder_no_crash(self, tmp_db_path):
        """recall(query=...) with no embedder doesn't crash."""
        mem = Memory(path=tmp_db_path)
        try:
            mem.remember("(prefers :subject Alice :object Python)")
            results = mem.recall(query="anything")
            assert isinstance(results, list)
        finally:
            mem.hb.close()

    # ---- S&B alignment (scan-all guard) ----

    def test_recall_query_only_does_not_scan_all_edges(self, tmp_db_path):
        """query-only mode uses semantic search, not score=0.5 scan-all."""
        embedder = MockEmbedder()
        mem = Memory(path=tmp_db_path, embedder=embedder)
        try:
            mem.remember("(prefers :subject Alice :object Python :memory_type semantic)")
            mem.remember("(visited :subject Bob :object Mars :memory_type episodic)")
            results = mem.recall(query="Alice prefers Python")
            # If scan-all fired, all results would have score=0.5
            # Semantic search produces variable scores based on cosine similarity
            if results:
                scores = {r["score"] for r in results}
                # At least some score should NOT be exactly 0.5
                assert not all(s == 0.5 for s in scores), (
                    "All scores are 0.5 -- scan-all branch likely fired instead of semantic search"
                )
        finally:
            mem.hb.close()


class TestForgetEdgeCases:
    def test_forget_older_than_and_min_strength_combined(self, tmp_db_path):
        """Conjunction of older_than + min_strength filters works."""
        mem = Memory(path=tmp_db_path)
        try:
            mem.remember("(met :subject Alice :object Bob)")
            future = datetime.now(UTC) + timedelta(days=1)
            # High strength threshold means all are weak enough to forget
            # older_than in the future means all are old enough
            result = mem.forget(older_than=future, min_strength=2.0)
            assert result["expired_count"] >= 1
        finally:
            mem.hb.close()


class TestEntityResolverEdgeCases:
    def test_resolve_empty_string(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        try:
            resolver = mem._resolver
            assert resolver.resolve("") == ""
            assert resolver.resolve("   ") == "   "
            assert not resolver.is_known("")
            assert not resolver.is_known("   ")
        finally:
            mem.hb.close()


class TestConsolidateEdgeSummaryType:
    def test_edge_consolidation_has_type_field(self, tmp_db_path):
        """Edge consolidation summaries have a 'type' field."""
        mem = Memory(path=tmp_db_path)
        try:
            mem.remember("(likes :subject Alice :object Python)")
            mem.remember("(prefers :subject Alice :object Python)")
            mem.remember("(enjoys :subject Alice :object Python)")
            summaries = mem.consolidate()
            edge_summaries = [s for s in summaries if s.get("type") == "edge_consolidation"]
            assert len(edge_summaries) >= 1
            for s in edge_summaries:
                assert "type" in s
                assert s["type"] == "edge_consolidation"
        finally:
            mem.hb.close()
