"""Tests for vector search in Hypabase.

Uses a mock embedder so sentence-transformers is not required in CI.
"""

from __future__ import annotations

import pytest

from hypabase import Hypabase
from hypabase.engine.vector import cosine_similarity, pack_embedding, unpack_embedding
from tests.conftest import MockEmbedder


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0], [1.0, 2.0])


class TestPackUnpack:
    def test_roundtrip_normalized(self):
        """pack_embedding normalizes, so unpacked values are unit-length."""
        import math

        vec = [1.0, 2.5, -3.14, 0.0]
        blob = pack_embedding(vec)
        unpacked = unpack_embedding(blob, 4)
        assert len(unpacked) == 4
        # Should be unit vector
        mag = math.sqrt(sum(x * x for x in unpacked))
        assert mag == pytest.approx(1.0, abs=1e-5)
        # Direction should be preserved (cosine similarity ~1.0)
        assert cosine_similarity(vec, unpacked) == pytest.approx(1.0, abs=1e-5)

    def test_already_normalized_idempotent(self):
        """Packing an already-normalized vector should be idempotent."""

        vec = [0.6, 0.8, 0.0]
        blob = pack_embedding(vec)
        unpacked = unpack_embedding(blob, 3)
        for a, b in zip(vec, unpacked):
            assert a == pytest.approx(b, abs=1e-5)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero or near-zero vector"):
            pack_embedding([0.0, 0.0, 0.0])

    def test_denormal_vector_raises(self):
        """Vectors with near-denormal magnitude should raise ValueError."""
        with pytest.raises(ValueError, match="zero or near-zero vector"):
            pack_embedding([1e-40] * 100)

    def test_empty_vector(self):
        blob = pack_embedding([])
        result = unpack_embedding(blob, 0)
        assert result == []

    def test_blob_size(self):
        vec = [1.0] * 384
        blob = pack_embedding(vec)
        assert len(blob) == 384 * 4  # 4 bytes per float32


class TestEmbeddingStorage:
    def test_save_and_load(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        blob = pack_embedding([1.0, 2.0, 3.0, 4.0])
        storage.save_embedding(
            id="emb1",
            namespace="default",
            kind="node",
            ref_id="alice",
            text="Alice is a person",
            embedding=blob,
            dimension=4,
            model="mock",
        )
        results = storage.load_embeddings("default")
        assert len(results) == 1
        assert results[0]["ref_id"] == "alice"
        assert results[0]["dimension"] == 4
        storage.close()

    def test_load_filtered_by_kind(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        blob = pack_embedding([1.0, 0.0, 0.0, 0.0])
        storage.save_embedding("e1", "default", "node", "a", "A", blob, 4, "mock")
        storage.save_embedding("e2", "default", "edge", "e1", "E", blob, 4, "mock")
        nodes = storage.load_embeddings("default", kind="node")
        assert len(nodes) == 1
        edges = storage.load_embeddings("default", kind="edge")
        assert len(edges) == 1
        storage.close()

    def test_delete_embeddings(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        blob = pack_embedding([1.0, 0.0, 0.0, 0.0])
        storage.save_embedding("e1", "default", "node", "a", "A", blob, 4, "mock")
        storage.save_embedding("e2", "default", "node", "b", "B", blob, 4, "mock")
        deleted = storage.delete_embeddings("default", kind="node", ref_id="a")
        assert deleted == 1
        remaining = storage.load_embeddings("default")
        assert len(remaining) == 1
        storage.close()

    def test_save_embedding_rowid_integrity(self, tmp_db_path):
        """Verify save_embedding succeeds and vec entry exists."""
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        blob = pack_embedding([1.0, 2.0, 3.0, 4.0])
        storage.save_embedding(
            id="emb_test",
            namespace="default",
            kind="node",
            ref_id="alice",
            text="Alice",
            embedding=blob,
            dimension=4,
            model="mock",
        )
        # Verify the embedding exists in both tables
        row = storage._conn.execute(
            "SELECT rowid FROM embeddings WHERE id = ?", ("emb_test",)
        ).fetchone()
        assert row is not None
        rowid = row[0]
        vec_row = storage._conn.execute(
            "SELECT rowid FROM vec_embeddings WHERE rowid = ?", (rowid,)
        ).fetchone()
        assert vec_row is not None
        storage.close()

    def test_upsert_embedding(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        blob1 = pack_embedding([1.0, 0.0, 0.0, 0.0])
        blob2 = pack_embedding([0.0, 1.0, 0.0, 0.0])
        storage.save_embedding("e1", "default", "node", "a", "A", blob1, 4, "mock")
        storage.save_embedding("e1", "default", "node", "a", "A updated", blob2, 4, "mock")
        results = storage.load_embeddings("default")
        assert len(results) == 1
        assert results[0]["text"] == "A updated"
        storage.close()


class TestClientSearch:
    def test_embed_and_search_node(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.embed_node("alice", "Alice is a software engineer")
        hb.embed_node("bob", "Bob is a doctor")
        results = hb.search("software engineer")
        assert len(results) >= 1
        hb.close()

    def test_embed_and_search_edge(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        e = hb.edge(["alice", "bob"], type="knows")
        hb.embed_edge(e.id, "Alice knows Bob")
        results = hb.search("Alice knows Bob", kind="edge")
        assert len(results) >= 1
        hb.close()

    def test_search_no_embedder(self):
        hb = Hypabase()
        assert hb.search("anything") == []

    def test_search_no_storage(self):
        hb = Hypabase(embedder=MockEmbedder())
        assert hb.search("anything") == []

    def test_embed_node_no_embedder(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        assert hb.embed_node("alice") is False

    def test_embed_node_nonexistent(self, tmp_db_path):
        hb = Hypabase(tmp_db_path, embedder=MockEmbedder())
        assert hb.embed_node("nonexistent") is False
        hb.close()

    def test_search_with_type_filter(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.node("aspirin", type="medication")
        hb.embed_node("alice", "Alice person")
        hb.embed_node("aspirin", "aspirin medication")
        results = hb.search("medication", kind="node", type="medication")
        assert all(r["node"].type == "medication" for r in results if "node" in r)
        hb.close()

    def test_embed_edge_without_text(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        e = hb.edge(["alice", "bob"], type="knows")
        result = hb.embed_edge(e.id)
        assert result is True
        # Verify the fallback text is the node IDs joined
        raw = hb._storage.load_embeddings(hb._current_ns, kind="edge")
        assert len(raw) == 1
        assert raw[0]["text"] == "alice bob"
        hb.close()

    def test_embed_edge_empty_text_returns_false(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        e = hb.edge(["alice", "bob"], type="knows")
        result = hb.embed_edge(e.id, text="")
        assert result is False
        hb.close()

    def test_search_with_min_score(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("a", type="t")
        hb.embed_node("a", "test")
        results = hb.search("test", min_score=0.999)
        # Same text should have score ~1.0
        assert len(results) >= 1
        assert all(r["score"] >= 0.999 for r in results)
        hb.close()


class TestSqliteVec:
    """Tests for sqlite-vec integration."""

    def test_search_with_vec(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.embed_node("alice", "Alice is a software engineer")
        hb.embed_node("bob", "Bob is a doctor")
        results = hb.search("software engineer")
        assert len(results) >= 1
        # sqlite-vec should have been used (vec_dimension set)
        assert hb._storage._vec_dimension is not None
        hb.close()

    def test_vec_upsert(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.embed_node("alice", "Alice version 1")
        hb.embed_node("alice", "Alice version 2")
        # Should still have exactly 1 embedding
        raw = hb._storage.load_embeddings(hb._current_ns, kind="node")
        assert len(raw) == 1
        assert raw[0]["text"] == "Alice version 2"
        # Vec search should still work
        results = hb.search("Alice version 2")
        assert len(results) >= 1
        hb.close()

    def test_vec_delete(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.embed_node("alice", "Alice")
        hb.embed_node("bob", "Bob")
        deleted = hb._storage.delete_embeddings(hb._current_ns, kind="node", ref_id="alice")
        assert deleted == 1
        remaining = hb._storage.load_embeddings(hb._current_ns)
        assert len(remaining) == 1
        assert remaining[0]["ref_id"] == "bob"
        hb.close()

    def test_search_empty_database(self, tmp_db_path):
        """Search on a fresh db with no embeddings returns empty list."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        results = hb.search("anything")
        assert results == []
        hb.close()

    def test_search_limit_larger_than_total(self, tmp_db_path):
        """limit=1000 with only 3 embeddings returns 3."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        for nid in ("a", "b", "c"):
            hb.node(nid, type="t")
            hb.embed_node(nid, f"text for {nid}")
        results = hb.search("text", limit=1000)
        assert len(results) == 3
        hb.close()


class TestNamespaceIsolation:
    """Tests for cross-namespace KNN isolation via partition key."""

    def test_search_only_returns_own_namespace(self, tmp_db_path):
        embedder = MockEmbedder()
        hb_alpha = Hypabase(tmp_db_path, embedder=embedder, database="alpha")
        hb_beta = Hypabase(tmp_db_path, embedder=embedder, database="beta")

        hb_alpha.node("a1", type="t")
        hb_alpha.embed_node("a1", "alpha specific text")
        hb_beta.node("b1", type="t")
        hb_beta.embed_node("b1", "beta specific text")

        alpha_results = hb_alpha.search("alpha specific text")
        beta_results = hb_beta.search("beta specific text")

        alpha_refs = {r["ref_id"] for r in alpha_results}
        beta_refs = {r["ref_id"] for r in beta_results}

        assert "a1" in alpha_refs
        assert "b1" not in alpha_refs
        assert "b1" in beta_refs
        assert "a1" not in beta_refs
        hb_alpha.close()

    def test_delete_in_one_namespace_preserves_other(self, tmp_db_path):
        """Delete embeddings in ns1 does not affect ns2."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        ns1 = hb.database("ns1")
        ns2 = hb.database("ns2")

        ns1.node("a", type="t")
        ns1.embed_node("a", "alpha text")
        ns2.node("b", type="t")
        ns2.embed_node("b", "beta text")

        ns1._storage.delete_embeddings("ns1")
        # ns2 embeddings should be unaffected
        remaining = ns2._storage.load_embeddings("ns2")
        assert len(remaining) == 1
        assert remaining[0]["ref_id"] == "b"
        hb.close()

    def test_small_namespace_returns_correct_results(self, tmp_db_path):
        """Many embeddings in one namespace, few in another — small still works."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        big = hb.database("big")
        small = hb.database("small")

        # Store many in "big"
        for i in range(50):
            big.node(f"big_{i}", type="t")
            big.embed_node(f"big_{i}", f"big namespace item number {i}")

        # Store few in "small"
        small.node("s1", type="t")
        small.embed_node("s1", "small namespace item")

        results = small.search("small namespace item", limit=5)
        assert len(results) >= 1
        assert all(r["ref_id"].startswith("s") for r in results)
        hb.close()


class TestDimensionMismatch:
    def test_dimension_mismatch_raises(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage
        from hypabase.engine.vector import pack_embedding

        storage = SQLiteStorage(tmp_db_path)
        blob4 = pack_embedding([1.0, 2.0, 3.0, 4.0])
        storage.save_embedding(
            id="e1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob4, dimension=4, model="mock",
        )
        blob8 = pack_embedding([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            storage.save_embedding(
                id="e2", namespace="default", kind="node", ref_id="b",
                text="B", embedding=blob8, dimension=8, model="mock",
            )
        storage.close()

    def test_dimension_mismatch_detected_on_reopen(self, tmp_db_path):
        """Reopening a database detects the prior dimension and rejects mismatches."""
        from hypabase.engine.storage import SQLiteStorage
        from hypabase.engine.vector import pack_embedding

        storage = SQLiteStorage(tmp_db_path)
        blob4 = pack_embedding([1.0, 2.0, 3.0, 4.0])
        storage.save_embedding(
            id="e1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob4, dimension=4, model="mock",
        )
        storage.close()

        # Reopen — _detect_vec_dimension should pick up dimension=4
        storage2 = SQLiteStorage(tmp_db_path)
        assert storage2.vec_dimension == 4
        blob8 = pack_embedding([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            storage2.save_embedding(
                id="e2", namespace="default", kind="node", ref_id="b",
                text="B", embedding=blob8, dimension=8, model="mock",
            )
        storage2.close()

    def test_first_embedding_sets_dimension(self, tmp_db_path):
        """Fresh db — first embed determines dimension."""
        from hypabase.engine.storage import SQLiteStorage
        from hypabase.engine.vector import pack_embedding

        storage = SQLiteStorage(tmp_db_path)
        assert storage.vec_dimension is None
        blob = pack_embedding([1.0, 2.0, 3.0, 4.0])
        storage.save_embedding(
            id="e1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob, dimension=4, model="mock",
        )
        assert storage.vec_dimension == 4
        storage.close()

    def test_same_dimension_succeeds(self, tmp_db_path):
        from hypabase.engine.storage import SQLiteStorage
        from hypabase.engine.vector import pack_embedding

        storage = SQLiteStorage(tmp_db_path)
        blob1 = pack_embedding([1.0, 0.0, 0.0, 0.0])
        blob2 = pack_embedding([0.0, 1.0, 0.0, 0.0])
        storage.save_embedding(
            id="e1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob1, dimension=4, model="mock",
        )
        storage.save_embedding(
            id="e2", namespace="default", kind="node", ref_id="b",
            text="B", embedding=blob2, dimension=4, model="mock",
        )
        results = storage.load_embeddings("default")
        assert len(results) == 2
        storage.close()


class TestSearchVecOverFetch:
    def test_search_all_filtered_returns_empty(self, tmp_db_path):
        """All KNN results filtered by kind returns empty list."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        # Store only node embeddings
        for i in range(5):
            hb.node(f"n{i}", type="item")
            hb.embed_node(f"n{i}", f"node text {i}")
        # Search for edges — all node embeddings should be filtered out
        results = hb.search("node text", kind="edge", limit=10)
        assert results == []
        hb.close()

    def test_search_vec_returns_enough_with_mixed_kinds(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)

        # Store 20 node embeddings
        for i in range(20):
            hb.node(f"n{i}", type="item")
            hb.embed_node(f"n{i}", f"node item number {i}")

        # Store 20 edge embeddings
        for i in range(20):
            e = hb.edge([f"a{i}", f"b{i}"], type="link")
            hb.embed_edge(e.id, f"edge link number {i}")

        # Search with kind="edge", limit=10 — should get 10 even though
        # half the KNN results may be node embeddings
        results = hb.search("edge link", kind="edge", limit=10)
        assert len(results) == 10
        hb.close()


class TestRebuildVecIndex:
    def test_rebuild_restores_search(self, tmp_db_path):
        """Drop vec table, rebuild, search works again."""
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.embed_node("alice", "Alice is a software engineer")
        hb.node("bob", type="person")
        hb.embed_node("bob", "Bob is a doctor")

        # Verify search works before
        results_before = hb.search("software engineer")
        assert len(results_before) >= 1

        # Drop vec table manually to simulate corruption
        hb._storage._conn.execute("DROP TABLE vec_embeddings")
        hb._storage._vec_dimension = None

        # Rebuild and verify
        count = hb.rebuild_search_index()
        assert count == 2

        # Search works again
        results_after = hb.search("software engineer")
        assert len(results_after) >= 1
        hb.close()

    def test_rebuild_no_embeddings(self, tmp_db_path):
        """Fresh db returns 0."""
        from hypabase.engine.storage import SQLiteStorage

        storage = SQLiteStorage(tmp_db_path)
        count = storage.rebuild_vec_index()
        assert count == 0
        storage.close()

    def test_rebuild_in_memory_returns_zero(self):
        """In-memory instance returns 0."""
        hb = Hypabase()
        assert hb.rebuild_search_index() == 0


class TestSearchTypeFilter:
    def test_search_with_edge_type_filter(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        e1 = hb.edge(["alice", "bob"], type="memory")
        e2 = hb.edge(["alice", "carol"], type="fact")
        hb.embed_edge(e1.id, "Alice remembers meeting Bob")
        hb.embed_edge(e2.id, "Alice knows Carol is a doctor")

        # Unfiltered search should return both types
        all_results = hb.search("Alice", kind="edge")
        all_types = {r["edge"].type for r in all_results if "edge" in r and r["edge"]}
        assert "memory" in all_types or "fact" in all_types, "Should have edge results"

        # Filtered search should only return memory edges
        results = hb.search("Alice", kind="edge", type="memory")
        for r in results:
            assert "edge" in r and r["edge"] is not None, "All results should have edge objects"
            assert r["edge"].type == "memory", f"Expected type 'memory', got '{r['edge'].type}'"
        hb.close()

    def test_search_with_node_type_filter(self, tmp_db_path):
        embedder = MockEmbedder()
        hb = Hypabase(tmp_db_path, embedder=embedder)
        hb.node("alice", type="person")
        hb.node("aspirin", type="medication")
        hb.embed_node("alice", "Alice is a person")
        hb.embed_node("aspirin", "aspirin is medication")

        # Filtered search should only return medication nodes
        results = hb.search("medication", kind="node", type="medication")
        for r in results:
            assert "node" in r and r["node"] is not None, "All results should have node objects"
            assert r["node"].type == "medication", f"Expected type 'medication', got '{r['node'].type}'"
        hb.close()


