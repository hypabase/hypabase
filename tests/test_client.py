"""Tests for the Hypabase client API."""

import sqlite3

import pytest

from hypabase import Hypabase
from hypabase.engine.storage import SQLiteStorage


class TestNodes:
    def test_create_node(self):
        hb = Hypabase()
        node = hb.node("alice", type="person")
        assert node.id == "alice"
        assert node.type == "person"

    def test_get_node(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        node = hb.get_node("alice")
        assert node is not None
        assert node.id == "alice"

    def test_get_nonexistent_node(self):
        hb = Hypabase()
        assert hb.get_node("nope") is None

    def test_query_nodes_by_type(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.node("aspirin", type="medication")
        persons = hb.nodes(type="person")
        assert len(persons) == 2

    def test_delete_node(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        assert hb.delete_node("alice") is True
        assert hb.get_node("alice") is None

    def test_node_with_properties(self):
        hb = Hypabase()
        node = hb.node("alice", type="person", age=30, role="engineer")
        assert node.properties["age"] == 30
        assert node.properties["role"] == "engineer"


class TestEdges:
    def test_create_edge(self):
        hb = Hypabase()
        edge = hb.edge(["alice", "bob"], type="knows")
        assert edge.type == "knows"
        assert edge.node_ids == ["alice", "bob"]
        assert edge.source == "unknown"
        assert edge.confidence == 1.0

    def test_hyperedge_multiple_nodes(self):
        hb = Hypabase()
        edge = hb.edge(
            ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
            type="treatment",
        )
        assert len(edge.node_ids) == 5
        assert edge.type == "treatment"

    def test_edge_auto_creates_nodes(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        assert hb.get_node("alice") is not None
        assert hb.get_node("bob") is not None

    def test_edge_with_provenance(self):
        hb = Hypabase()
        edge = hb.edge(
            ["a", "b", "c"],
            type="rel",
            source="llm_extraction",
            confidence=0.85,
        )
        assert edge.source == "llm_extraction"
        assert edge.confidence == 0.85

    def test_edge_requires_two_nodes(self):
        hb = Hypabase()
        with pytest.raises(ValueError):
            hb.edge(["only_one"], type="invalid")

    def test_query_edges_containing(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="knows")
        hb.edge(["bob", "carol"], type="knows")
        edges = hb.edges(containing=["alice"])
        assert len(edges) == 2

    def test_query_edges_match_all(self):
        hb = Hypabase()
        hb.edge(["alice", "bob", "carol"], type="group")
        hb.edge(["alice", "bob"], type="pair")
        hb.edge(["alice", "dave"], type="pair")
        edges = hb.edges(containing=["alice", "bob"], match_all=True)
        assert len(edges) == 2  # group and pair both contain alice+bob

    def test_query_edges_by_type(self):
        hb = Hypabase()
        hb.edge(["a", "b"], type="fk")
        hb.edge(["c", "d"], type="fk")
        hb.edge(["e", "f"], type="relation")
        edges = hb.edges(type="fk")
        assert len(edges) == 2

    def test_vertex_set_lookup(self):
        hb = Hypabase()
        hb.edge(["alice", "bob", "carol"], type="group")
        hb.edge(["alice", "bob"], type="pair")
        # Exact vertex set match (order-independent)
        results = hb.edges_by_vertex_set(["carol", "alice", "bob"])
        assert len(results) == 1
        assert results[0].type == "group"

    def test_delete_edge(self):
        hb = Hypabase()
        edge = hb.edge(["a", "b"], type="rel")
        assert hb.delete_edge(edge.id) is True
        assert hb.get_edge(edge.id) is None


class TestProvenance:
    def test_context_manager(self):
        hb = Hypabase()
        with hb.context(source="schema_analysis", confidence=0.9):
            e1 = hb.edge(["a", "b"], type="fk")
            e2 = hb.edge(["b", "c"], type="fk")
        assert e1.source == "schema_analysis"
        assert e1.confidence == 0.9
        assert e2.source == "schema_analysis"

    def test_context_override(self):
        hb = Hypabase()
        with hb.context(source="default_source", confidence=0.8):
            e = hb.edge(["a", "b"], type="fk", source="explicit", confidence=0.99)
        assert e.source == "explicit"
        assert e.confidence == 0.99

    def test_context_restores(self):
        hb = Hypabase()
        with hb.context(source="inner", confidence=0.5):
            pass
        e = hb.edge(["a", "b"], type="rel")
        assert e.source == "unknown"
        assert e.confidence == 1.0


class TestTraversal:
    def test_neighbors(self):
        hb = Hypabase()
        hb.edge(["alice", "bob", "carol"], type="group")
        hb.edge(["alice", "dave"], type="pair")
        neighbors = hb.neighbors("alice")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"bob", "carol", "dave"}

    def test_neighbors_filtered(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="works_with")
        neighbors = hb.neighbors("alice", edge_types=["knows"])
        assert len(neighbors) == 1
        assert neighbors[0].id == "bob"

    def test_find_paths(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["bob", "carol"], type="knows")
        paths = hb.paths("alice", "carol")
        assert len(paths) >= 1
        assert paths[0] == ["alice", "bob", "carol"]

    def test_find_paths_max_hops(self):
        hb = Hypabase()
        hb.edge(["a", "b"], type="link")
        hb.edge(["b", "c"], type="link")
        hb.edge(["c", "d"], type="link")
        # max_hops=1 should not find a->d
        paths = hb.paths("a", "d", max_hops=1)
        assert len(paths) == 0


class TestStats:
    def test_stats(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.edge(["alice", "bob"], type="knows")
        s = hb.stats()
        assert s.node_count == 2
        assert s.edge_count == 1
        assert s.nodes_by_type["person"] == 2
        assert s.edges_by_type["knows"] == 1


class TestProvenanceQueries:
    def test_edges_filter_by_source_clinical(self, populated_hb):
        edges = populated_hb.edges(source="clinical_records")
        assert len(edges) == 2
        assert all(e.source == "clinical_records" for e in edges)

    def test_edges_filter_by_source_lab(self, populated_hb):
        edges = populated_hb.edges(source="lab_results")
        assert len(edges) == 1
        assert edges[0].source == "lab_results"

    def test_edges_filter_by_source_hospital(self, populated_hb):
        edges = populated_hb.edges(source="hospital_system")
        assert len(edges) == 1
        assert edges[0].source == "hospital_system"

    def test_edges_filter_by_source_no_match(self, populated_hb):
        edges = populated_hb.edges(source="nonexistent_source")
        assert edges == []

    def test_edges_filter_by_min_confidence(self, populated_hb):
        edges = populated_hb.edges(min_confidence=0.9)
        assert len(edges) == 2  # treatment (0.95) and prescribes (0.92)
        assert all(e.confidence >= 0.9 for e in edges)

    def test_edges_filter_by_min_confidence_boundary(self, populated_hb):
        edges = populated_hb.edges(min_confidence=0.88)
        assert len(edges) == 3  # treatment (0.95), diagnosis (0.88), prescribes (0.92)
        assert all(e.confidence >= 0.88 for e in edges)

    def test_edges_filter_composed(self, populated_hb):
        edges = populated_hb.edges(
            containing=["patient_123"],
            source="clinical_records",
            min_confidence=0.9,
        )
        assert len(edges) == 1
        assert edges[0].type == "treatment"

    def test_edges_filter_type_and_source(self, populated_hb):
        edges = populated_hb.edges(type="treatment", source="clinical_records")
        assert len(edges) == 1
        assert edges[0].type == "treatment"
        assert edges[0].source == "clinical_records"

    def test_sources_aggregation(self, populated_hb):
        sources = populated_hb.sources()
        assert len(sources) == 3
        src_map = {s["source"]: s for s in sources}
        assert src_map["clinical_records"]["edge_count"] == 2
        assert src_map["lab_results"]["edge_count"] == 1
        assert src_map["hospital_system"]["edge_count"] == 1
        # avg of 0.95 and 0.92
        assert src_map["clinical_records"]["avg_confidence"] == pytest.approx(0.935, abs=0.001)

    def test_sources_empty_graph(self, hb):
        assert hb.sources() == []


class TestPersistence:
    def test_persistence_roundtrip(self, tmp_db_path):
        # Write
        hb = Hypabase(tmp_db_path)
        hb.node("alice", type="person", age=30)
        hb.edge(
            ["alice", "bob"],
            type="knows",
            source="manual",
            confidence=0.9,
        )
        hb.close()

        # Read back
        hb2 = Hypabase(tmp_db_path)
        alice = hb2.get_node("alice")
        assert alice is not None
        assert alice.type == "person"
        assert alice.properties["age"] == 30
        edges = hb2.edges(containing=["alice"])
        assert len(edges) == 1
        assert edges[0].source == "manual"
        assert edges[0].confidence == 0.9
        hb2.close()

    def test_context_manager_persistence(self, tmp_db_path):
        with Hypabase(tmp_db_path) as hb:
            hb.node("alice", type="person")
            hb.edge(["alice", "bob"], type="knows")

        with Hypabase(tmp_db_path) as hb2:
            assert hb2.get_node("alice") is not None
            assert len(hb2.edges()) == 1

    def test_persistence_empty_db(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.close()
        hb2 = Hypabase(tmp_db_path)
        assert hb2.nodes() == []
        assert hb2.edges() == []
        hb2.close()

    def test_persistence_multiple_sessions(self, tmp_db_path):
        # Session 1
        with Hypabase(tmp_db_path) as hb:
            hb.node("alice", type="person")
            hb.edge(["alice", "bob"], type="knows")

        # Session 2
        with Hypabase(tmp_db_path) as hb:
            hb.node("carol", type="person")
            hb.edge(["bob", "carol"], type="knows")

        # Session 3
        with Hypabase(tmp_db_path) as hb:
            hb.edge(["carol", "dave"], type="knows")

        # Verify all data present
        with Hypabase(tmp_db_path) as hb:
            assert len(hb.nodes()) == 4  # alice, bob, carol, dave
            assert len(hb.edges()) == 3

    def test_edge_ref_persistence_roundtrip(self, tmp_db_path):
        """Save/load through SQLiteStorage preserves edge refs."""
        from hypabase.engine.core import Hyperedge as CoreEdge
        from hypabase.engine.core import HypergraphCore
        from hypabase.engine.core import Incidence as CoreIncidence
        from hypabase.engine.core import Node as CoreNode

        storage = SQLiteStorage(tmp_db_path)
        store = HypergraphCore()
        store.add_node(CoreNode("A", "t"))
        store.add_node(CoreNode("B", "t"))
        store.add_edge(
            CoreEdge(
                id="e1", type="link",
                incidences=[CoreIncidence(node_id="A"), CoreIncidence(node_id="B")],
            )
        )
        store.add_edge(
            CoreEdge(
                id="e2", type="meta",
                incidences=[CoreIncidence(node_id="A"), CoreIncidence(edge_ref_id="e1")],
            )
        )
        storage.save_namespace("default", store)
        storage.close()

        # Reload
        storage2 = SQLiteStorage(tmp_db_path)
        loaded = storage2.load_namespace("default")
        storage2.close()

        e2 = loaded.get_edge("e2")
        assert e2 is not None
        assert len(e2.incidences) == 2
        assert e2.incidences[0].node_id == "A"
        assert e2.incidences[0].edge_ref_id is None
        assert e2.incidences[1].node_id is None
        assert e2.incidences[1].edge_ref_id == "e1"
        assert e2.edge_refs == ["e1"]


class TestDirectedEdges:
    def test_directed_edge_head_tail(self):
        hb = Hypabase()
        edge = hb.edge(["alice", "bob"], type="follows", directed=True)
        assert edge.directed is True
        assert edge.incidences[0].node_id == "alice"
        assert edge.incidences[0].direction == "tail"
        assert edge.incidences[1].node_id == "bob"
        assert edge.incidences[1].direction == "head"

    def test_directed_edge_three_nodes(self):
        hb = Hypabase()
        edge = hb.edge(["a", "b", "c"], type="flow", directed=True)
        assert edge.incidences[0].direction == "tail"
        assert edge.incidences[1].direction is None
        assert edge.incidences[2].direction == "head"


class TestAdvancedOperations:
    def test_delete_node_cascade(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="knows")
        hb.edge(["bob", "carol"], type="knows")
        deleted, edge_count = hb.delete_node_cascade("alice")
        assert deleted is True
        assert edge_count == 2
        assert hb.get_node("alice") is None
        # bob-carol edge still exists
        assert len(hb.edges()) == 1

    def test_delete_node_cascade_nonexistent(self):
        hb = Hypabase()
        deleted, edge_count = hb.delete_node_cascade("nobody")
        assert deleted is False
        assert edge_count == 0

    def test_validate_clean_graph(self, populated_hb):
        result = populated_hb.validate()
        assert result.valid is True
        assert result.errors == []

    def test_batch_operations(self):
        hb = Hypabase()
        with hb.batch():
            hb.node("alice", type="person")
            hb.node("bob", type="person")
            hb.edge(["alice", "bob"], type="knows")
        assert len(hb.nodes()) == 2
        assert len(hb.edges()) == 1

    def test_upsert_edge_by_vertex_set(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        e1 = hb.upsert_edge_by_vertex_set(
            {"alice", "bob"},
            "knows",
            source="manual",
            confidence=0.8,
        )
        e2 = hb.upsert_edge_by_vertex_set(
            {"alice", "bob"},
            "knows",
            {"note": "updated"},
            source="manual",
            confidence=0.9,
        )
        assert e1.id == e2.id  # same edge updated

    def test_edges_of_node(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="works_with")
        hb.edge(["bob", "carol"], type="knows")
        edges = hb.edges_of_node("alice")
        assert len(edges) == 2

    def test_edges_of_node_filtered(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="works_with")
        edges = hb.edges_of_node("alice", edge_types=["knows"])
        assert len(edges) == 1
        assert edges[0].type == "knows"

    def test_hif_roundtrip(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.edge(["alice", "bob"], type="knows", source="test", confidence=0.9)
        hif = hb.to_hif()
        hb2 = Hypabase.from_hif(hif)
        assert len(hb2.nodes()) == len(hb.nodes())
        assert len(hb2.edges()) == len(hb.edges())
        edge = hb2.edges()[0]
        assert edge.type == "knows"
        assert edge.source == "test"

    def test_context_nesting(self):
        hb = Hypabase()
        with hb.context(source="outer", confidence=0.8):
            e_outer = hb.edge(["a", "b"], type="rel")
            with hb.context(source="inner", confidence=0.5):
                e_inner = hb.edge(["c", "d"], type="rel")
            e_restored = hb.edge(["e", "f"], type="rel")
        e_outside = hb.edge(["g", "h"], type="rel")

        assert e_outer.source == "outer"
        assert e_outer.confidence == 0.8
        assert e_inner.source == "inner"
        assert e_inner.confidence == 0.5
        assert e_restored.source == "outer"
        assert e_restored.confidence == 0.8
        assert e_outside.source == "unknown"
        assert e_outside.confidence == 1.0


class TestEdgeCases:
    def test_empty_string_node_id_rejected(self):
        hb = Hypabase()
        with pytest.raises(ValueError, match="non-empty"):
            hb.node("", type="person")

    def test_empty_string_in_edge_nodes_rejected(self):
        hb = Hypabase()
        with pytest.raises(ValueError, match="non-empty"):
            hb.edge(["alice", ""], type="knows")

    def test_duplicate_nodes_in_edge(self):
        hb = Hypabase()
        edge = hb.edge(["alice", "alice"], type="self_loop")
        assert len(edge.node_ids) == 2
        assert edge.node_ids == ["alice", "alice"]


class TestAutoPersist:
    def test_node_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("alice", type="person", age=30)
        # No close()!
        del hb

        hb2 = Hypabase(tmp_db_path)
        alice = hb2.get_node("alice")
        assert alice is not None
        assert alice.type == "person"
        assert alice.properties["age"] == 30
        hb2.close()

    def test_edge_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.edge(
            ["alice", "bob", "carol"],
            type="group",
            source="test_src",
            confidence=0.85,
        )
        del hb

        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_node("alice") is not None
        assert hb2.get_node("bob") is not None
        assert hb2.get_node("carol") is not None
        edges = hb2.edges()
        assert len(edges) == 1
        assert edges[0].type == "group"
        assert edges[0].source == "test_src"
        assert edges[0].confidence == 0.85
        assert len(edges[0].node_ids) == 3
        hb2.close()

    def test_delete_node_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        hb.delete_node("alice")
        del hb

        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_node("alice") is None
        assert hb2.get_node("bob") is not None
        hb2.close()

    def test_delete_edge_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        edge = hb.edge(["alice", "bob"], type="knows")
        hb.delete_edge(edge.id)
        del hb

        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_edge(edge.id) is None
        # Nodes remain
        assert hb2.get_node("alice") is not None
        assert hb2.get_node("bob") is not None
        hb2.close()

    def test_delete_node_cascade_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="knows")
        hb.delete_node_cascade("alice")
        del hb

        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_node("alice") is None
        assert len(hb2.edges()) == 0
        # bob and carol remain
        assert hb2.get_node("bob") is not None
        assert hb2.get_node("carol") is not None
        hb2.close()

    def test_upsert_edge_by_vertex_set_auto_persists(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("alice", type="person")
        hb.node("bob", type="person")
        edge = hb.upsert_edge_by_vertex_set(
            {"alice", "bob"},
            "knows",
            source="manual",
            confidence=0.9,
        )
        del hb

        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_edge(edge.id) is not None
        hb2.close()

    def test_batch_defers_persist(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        with hb.batch():
            hb.node("alice", type="person")
            hb.edge(["alice", "bob"], type="knows")
            # Mid-batch: open a second reader to verify NOT on disk
            reader = SQLiteStorage(tmp_db_path)
            store = reader.load_namespace("default")
            assert len(store.get_all_nodes()) == 0
            reader.close()
        # After batch exits: data IS on disk
        reader = SQLiteStorage(tmp_db_path)
        store = reader.load_namespace("default")
        assert len(store.get_all_nodes()) == 2
        assert len(store.get_all_edges()) == 1
        reader.close()
        hb.close()

    def test_batch_nesting(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        with hb.batch():
            hb.node("alice", type="person")
            with hb.batch():
                hb.edge(["alice", "bob"], type="knows")
                # Still inside outermost batch — not on disk
                reader = SQLiteStorage(tmp_db_path)
                store = reader.load_namespace("default")
                assert len(store.get_all_nodes()) == 0
                reader.close()
            # Inner batch exited but outer still active — still not on disk
            reader = SQLiteStorage(tmp_db_path)
            store = reader.load_namespace("default")
            assert len(store.get_all_nodes()) == 0
            reader.close()
        # Outermost batch exited — now on disk
        reader = SQLiteStorage(tmp_db_path)
        store = reader.load_namespace("default")
        assert len(store.get_all_nodes()) == 2
        assert len(store.get_all_edges()) == 1
        reader.close()
        hb.close()

    def test_batch_exception_resets_depth(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("pre_existing", type="person")

        with pytest.raises(RuntimeError, match="boom"):
            with hb.batch():
                hb.node("alice", type="person")
                raise RuntimeError("boom")

        # Partial changes from the failed batch are persisted (no rollback)
        reader = SQLiteStorage(tmp_db_path)
        store = reader.load_namespace("default")
        assert store.get_node("pre_existing") is not None
        assert store.get_node("alice") is not None
        reader.close()

        # Subsequent operations still auto-persist (depth is back to 0)
        hb.node("bob", type="person")
        reader = SQLiteStorage(tmp_db_path)
        store = reader.load_namespace("default")
        assert store.get_node("bob") is not None
        reader.close()
        hb.close()

    def test_in_memory_no_auto_persist(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        hb.edge(["alice", "bob"], type="knows")
        hb.delete_node("alice")
        hb.node("carol", type="person")
        hb.edge(["carol", "dave"], type="knows")
        hb.delete_edge(hb.edges()[0].id)
        hb.node("x", type="t")
        hb.node("y", type="t")
        hb.upsert_edge_by_vertex_set({"x", "y"}, "link")
        hb.delete_node_cascade("carol")
        # All operations succeed — _auto_save is a no-op for in-memory


class TestConstructor:
    def test_in_memory_no_args(self):
        hb = Hypabase()
        assert hb._storage is None
        hb.node("a", type="t")
        assert hb.get_node("a") is not None

    def test_local_sqlite(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("a", type="t")
        hb.close()
        hb2 = Hypabase(tmp_db_path)
        assert hb2.get_node("a") is not None
        hb2.close()

    def test_url_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Cloud backends"):
            Hypabase("https://project.hypabase.app", key="hb_pk_test")

    def test_http_url_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Cloud backends"):
            Hypabase("http://localhost:8080")

    def test_default_database(self):
        hb = Hypabase()
        assert hb.current_database == "default"


class TestDatabases:
    def test_database_creates_scoped_view(self):
        hb = Hypabase()
        drugs = hb.database("drugs")
        assert drugs.current_database == "drugs"

    def test_namespace_isolation(self):
        hb = Hypabase()
        drugs = hb.database("drugs")
        sessions = hb.database("sessions")
        drugs.node("aspirin", type="drug")
        sessions.node("s1", type="session")
        assert len(drugs.nodes()) == 1
        assert drugs.nodes()[0].id == "aspirin"
        assert len(sessions.nodes()) == 1
        assert sessions.nodes()[0].id == "s1"

    def test_databases_lists_all(self):
        hb = Hypabase()
        hb.database("drugs")
        hb.database("sessions")
        dbs = hb.databases()
        assert "default" in dbs
        assert "drugs" in dbs
        assert "sessions" in dbs

    def test_delete_database(self):
        hb = Hypabase()
        drugs = hb.database("drugs")
        drugs.node("aspirin", type="drug")
        assert hb.delete_database("drugs") is True
        assert "drugs" not in hb.databases()

    def test_delete_nonexistent_database(self):
        hb = Hypabase()
        assert hb.delete_database("nope") is False

    def test_cross_namespace_independence(self):
        hb = Hypabase()
        drugs = hb.database("drugs")
        sessions = hb.database("sessions")
        drugs.node("aspirin", type="drug")
        drugs.edge(["aspirin", "ibuprofen"], type="interaction")
        sessions.node("s1", type="session")
        sessions.edge(["s1", "user_alice"], type="attended")
        # Each namespace sees only its own data
        assert len(drugs.nodes()) == 2
        assert len(drugs.edges()) == 1
        assert len(sessions.nodes()) == 2
        assert len(sessions.edges()) == 1
        # Default is empty
        assert len(hb.nodes()) == 0

    def test_namespace_persistence(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        drugs = hb.database("drugs")
        drugs.node("aspirin", type="drug")
        sessions = hb.database("sessions")
        sessions.node("s1", type="session")
        hb.close()
        # Reopen
        hb2 = Hypabase(tmp_db_path)
        drugs2 = hb2.database("drugs")
        assert drugs2.get_node("aspirin") is not None
        sessions2 = hb2.database("sessions")
        assert sessions2.get_node("s1") is not None
        hb2.close()

    def test_current_database_property(self):
        hb = Hypabase()
        assert hb.current_database == "default"
        drugs = hb.database("drugs")
        assert drugs.current_database == "drugs"
        # Original is unchanged
        assert hb.current_database == "default"


class TestNewQueryMethods:
    def test_find_nodes(self):
        hb = Hypabase()
        hb.node("a", type="person", role="admin")
        hb.node("b", type="person", role="user")
        hb.node("c", type="person", role="admin")
        admins = hb.find_nodes(role="admin")
        assert len(admins) == 2
        admin_ids = {n.id for n in admins}
        assert admin_ids == {"a", "c"}

    def test_find_nodes_no_match(self):
        hb = Hypabase()
        hb.node("a", type="person", role="admin")
        assert hb.find_nodes(role="superuser") == []

    def test_has_node(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        assert hb.has_node("alice") is True
        assert hb.has_node("bob") is False

    def test_find_edges(self):
        hb = Hypabase()
        hb.edge(["a", "b"], type="link", properties={"weight": 5})
        hb.edge(["c", "d"], type="link", properties={"weight": 10})
        hb.edge(["e", "f"], type="link", properties={"weight": 5})
        results = hb.find_edges(weight=5)
        assert len(results) == 2

    def test_has_edge_with_nodes(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        assert hb.has_edge_with_nodes({"alice", "bob"}) is True
        assert hb.has_edge_with_nodes({"alice", "carol"}) is False

    def test_has_edge_with_nodes_typed(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        assert hb.has_edge_with_nodes({"alice", "bob"}, "knows") is True
        assert hb.has_edge_with_nodes({"alice", "bob"}, "works_with") is False


class TestDeleteCascade:
    def test_delete_node_cascade_flag(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="knows")
        hb.edge(["bob", "carol"], type="knows")
        assert hb.delete_node("alice", cascade=True) is True
        assert hb.get_node("alice") is None
        assert len(hb.edges()) == 1  # bob-carol remains

    def test_delete_node_no_cascade(self):
        hb = Hypabase()
        hb.node("alice", type="person")
        assert hb.delete_node("alice") is True
        assert hb.get_node("alice") is None

    def test_delete_node_cascade_nonexistent(self):
        hb = Hypabase()
        assert hb.delete_node("nobody", cascade=True) is False

    def test_legacy_delete_node_cascade(self):
        hb = Hypabase()
        hb.edge(["alice", "bob"], type="knows")
        hb.edge(["alice", "carol"], type="knows")
        deleted, edge_count = hb.delete_node_cascade("alice")
        assert deleted is True
        assert edge_count == 2


class TestSchemaMigration:
    """Tests for SQLite schema migration edge cases."""

    def _create_v3_database(self, path: str) -> None:
        """Create a v3 schema database with test data."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE nodes (
                id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                type TEXT NOT NULL DEFAULT 'unknown',
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (id, namespace)
            );
            CREATE TABLE edges (
                id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                type TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 1.0,
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (id, namespace)
            );
            CREATE TABLE incidences (
                edge_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                node_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                direction TEXT,
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (edge_id, namespace, position),
                FOREIGN KEY (edge_id, namespace)
                    REFERENCES edges(id, namespace) ON DELETE CASCADE
            );
            CREATE TABLE vertex_set_index (
                vertex_set_hash TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (vertex_set_hash, edge_id, namespace),
                FOREIGN KEY (edge_id, namespace)
                    REFERENCES edges(id, namespace) ON DELETE CASCADE
            );
        """)
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '3')")
        # Add test data
        conn.execute(
            "INSERT INTO nodes (id, namespace, type, properties) VALUES (?, ?, ?, ?)",
            ("alice", "default", "person", "{}"),
        )
        conn.execute(
            "INSERT INTO nodes (id, namespace, type, properties) VALUES (?, ?, ?, ?)",
            ("bob", "default", "person", "{}"),
        )
        conn.execute(
            "INSERT INTO edges (id, namespace, type, source, confidence, properties)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("e1", "default", "knows", "manual", 0.9, "{}"),
        )
        conn.execute(
            "INSERT INTO incidences (edge_id, namespace, node_id, position, direction, properties)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("e1", "default", "alice", 0, None, "{}"),
        )
        conn.execute(
            "INSERT INTO incidences (edge_id, namespace, node_id, position, direction, properties)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("e1", "default", "bob", 1, None, "{}"),
        )
        conn.commit()
        conn.close()

    def test_schema_v3_to_v4_migration(self, tmp_db_path):
        """v3 database is migrated to v4, preserving all data."""
        self._create_v3_database(tmp_db_path)

        storage = SQLiteStorage(tmp_db_path)
        loaded = storage.load_namespace("default")
        storage.close()

        # Verify data preserved
        assert loaded.get_node("alice") is not None
        assert loaded.get_node("bob") is not None
        e1 = loaded.get_edge("e1")
        assert e1 is not None
        assert e1.type == "knows"
        assert e1.source == "manual"
        assert e1.confidence == 0.9
        assert len(e1.incidences) == 2

        # Verify version is now 4
        conn = sqlite3.connect(tmp_db_path)
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        conn.close()
        assert version == "4"

    def test_schema_v3_migration_rejects_null_node_id(self, tmp_db_path):
        """v3 database with NULL node_id raises ValueError during migration."""
        # Create a v3 database with a permissive incidences schema (nullable node_id)
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=OFF")  # Allow corrupt data
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (
                id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                type TEXT NOT NULL DEFAULT 'unknown',
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (id, namespace)
            );
            CREATE TABLE edges (
                id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                type TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 1.0,
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (id, namespace)
            );
            CREATE TABLE incidences (
                edge_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                node_id TEXT,
                position INTEGER NOT NULL,
                direction TEXT,
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (edge_id, namespace, position)
            );
            CREATE TABLE vertex_set_index (
                vertex_set_hash TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (vertex_set_hash, edge_id, namespace)
            );
        """)
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '3')")
        conn.execute(
            "INSERT INTO edges (id, namespace, type) VALUES (?, ?, ?)",
            ("e_bad", "default", "bad"),
        )
        conn.execute(
            "INSERT INTO incidences (edge_id, namespace, node_id, position)"
            " VALUES (?, ?, NULL, ?)",
            ("e_bad", "default", 0),
        )
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="NULL node_id"):
            SQLiteStorage(tmp_db_path)

    def test_schema_unknown_version_raises(self, tmp_db_path):
        """Database with unsupported schema version raises ValueError."""
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '99')")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="Unsupported schema version '99'"):
            SQLiteStorage(tmp_db_path)

    def test_schema_missing_version_key_raises(self, tmp_db_path):
        """Database with meta table but no schema_version key raises ValueError."""
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta (key, value) VALUES ('other_key', 'whatever')")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="no schema_version key"):
            SQLiteStorage(tmp_db_path)
