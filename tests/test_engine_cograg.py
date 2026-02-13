"""Tests for Cog-RAG support methods in HypergraphStore."""

import pytest

from hypabase.engine.core import (
    Hyperedge,
    HypergraphStore,
    Incidence,
    Node,
)


class TestHasNode:
    """Tests for has_node() method."""

    def test_existing_node_returns_true(self):
        """has_node returns True for existing node."""
        store = HypergraphStore()
        store.add_node(Node("A", "entity"))
        assert store.has_node("A") is True

    def test_missing_node_returns_false(self):
        """has_node returns False for non-existent node."""
        store = HypergraphStore()
        assert store.has_node("nonexistent") is False

    def test_empty_store(self):
        """has_node returns False on empty store."""
        store = HypergraphStore()
        assert store.has_node("any") is False

    def test_deleted_node_returns_false(self):
        """has_node returns False after node deletion."""
        store = HypergraphStore()
        store.add_node(Node("A", "entity"))
        store.delete_node("A")
        assert store.has_node("A") is False

    def test_empty_string_id(self):
        """has_node works with empty string ID."""
        store = HypergraphStore()
        store.add_node(Node("", "entity"))
        assert store.has_node("") is True

    def test_special_characters_in_id(self):
        """has_node works with special characters."""
        store = HypergraphStore()
        special_ids = ["node.with.dots", "node::colons", "中文", "emoji_😀"]
        for nid in special_ids:
            store.add_node(Node(nid, "entity"))
            assert store.has_node(nid) is True


class TestGetEdgesOfNode:
    """Tests for get_edges_of_node() method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Store with nodes and multiple edge types."""
        s = HypergraphStore()
        for nid in ["A", "B", "C", "D"]:
            s.add_node(Node(nid, "entity"))
        s.add_edge(Hyperedge("e1", "relation", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e2", "relation", [Incidence("A"), Incidence("C")]))
        s.add_edge(Hyperedge("e3", "theme", [Incidence("A"), Incidence("B"), Incidence("C")]))
        s.add_edge(Hyperedge("e4", "relation", [Incidence("B"), Incidence("D")]))
        return s

    def test_returns_all_incident_edges(self, store):
        """Returns all edges containing the node."""
        edges = store.get_edges_of_node("A")
        assert len(edges) == 3
        edge_ids = {e.id for e in edges}
        assert edge_ids == {"e1", "e2", "e3"}

    def test_node_in_single_edge(self, store):
        """Node in single edge returns that edge."""
        edges = store.get_edges_of_node("D")
        assert len(edges) == 1
        assert edges[0].id == "e4"

    def test_filter_by_edge_type(self, store):
        """Filter edges by type."""
        edges = store.get_edges_of_node("A", edge_types=["relation"])
        assert len(edges) == 2
        edge_ids = {e.id for e in edges}
        assert edge_ids == {"e1", "e2"}

    def test_filter_by_multiple_types(self, store):
        """Filter by multiple edge types."""
        edges = store.get_edges_of_node("A", edge_types=["relation", "theme"])
        assert len(edges) == 3

    def test_filter_by_nonexistent_type(self, store):
        """Filter by non-existent type returns empty."""
        edges = store.get_edges_of_node("A", edge_types=["nonexistent"])
        assert edges == []

    def test_isolated_node_returns_empty(self):
        """Node with no edges returns empty list."""
        store = HypergraphStore()
        store.add_node(Node("isolated", "entity"))
        edges = store.get_edges_of_node("isolated")
        assert edges == []

    def test_nonexistent_node_returns_empty(self):
        """Non-existent node returns empty list."""
        store = HypergraphStore()
        edges = store.get_edges_of_node("nonexistent")
        assert edges == []

    def test_returns_hyperedge_objects(self, store):
        """Returned items are Hyperedge instances."""
        edges = store.get_edges_of_node("A")
        assert all(isinstance(e, Hyperedge) for e in edges)

    def test_edge_properties_preserved(self, store):
        """Edge properties accessible on returned edges."""
        store.add_edge(
            Hyperedge(
                "e5",
                "relation",
                [Incidence("A"), Incidence("D")],
                properties={"weight": 0.8},
                source="test",
                confidence=0.9,
            )
        )
        edges = store.get_edges_of_node("A", edge_types=["relation"])
        e5 = next(e for e in edges if e.id == "e5")
        assert e5.properties["weight"] == 0.8
        assert e5.source == "test"
        assert e5.confidence == 0.9


class TestGetEdgeNodeTuplesOfNode:
    """Tests for get_edge_node_tuples_of_node() method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Store with various edge configurations."""
        s = HypergraphStore()
        for nid in ["A", "B", "C", "D", "E"]:
            s.add_node(Node(nid, "entity"))
        s.add_edge(Hyperedge("e1", "low", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e2", "low", [Incidence("A"), Incidence("C")]))
        s.add_edge(Hyperedge("e3", "high", [Incidence("A"), Incidence("B"), Incidence("C")]))
        s.add_edge(Hyperedge("e4", "low", [Incidence("D"), Incidence("E")]))
        return s

    def test_returns_frozensets(self, store):
        """Returns set of frozensets."""
        tuples = store.get_edge_node_tuples_of_node("A")
        assert isinstance(tuples, set)
        assert all(isinstance(t, frozenset) for t in tuples)

    def test_correct_vertex_sets(self, store):
        """Returns correct vertex sets for incident edges."""
        tuples = store.get_edge_node_tuples_of_node("A")
        assert len(tuples) == 3
        assert frozenset({"A", "B"}) in tuples
        assert frozenset({"A", "C"}) in tuples
        assert frozenset({"A", "B", "C"}) in tuples

    def test_filter_by_edge_type(self, store):
        """Filter by edge type."""
        tuples = store.get_edge_node_tuples_of_node("A", edge_types=["low"])
        assert len(tuples) == 2
        assert frozenset({"A", "B"}) in tuples
        assert frozenset({"A", "C"}) in tuples

    def test_isolated_node_returns_empty_set(self):
        """Isolated node returns empty set."""
        store = HypergraphStore()
        store.add_node(Node("isolated", "entity"))
        tuples = store.get_edge_node_tuples_of_node("isolated")
        assert tuples == set()

    def test_nonexistent_node_returns_empty_set(self):
        """Non-existent node returns empty set."""
        store = HypergraphStore()
        tuples = store.get_edge_node_tuples_of_node("nonexistent")
        assert tuples == set()

    def test_node_in_single_edge(self, store):
        """Node in single edge returns single frozenset."""
        tuples = store.get_edge_node_tuples_of_node("D")
        assert tuples == {frozenset({"D", "E"})}

    def test_duplicate_vertex_sets_deduplicated(self):
        """Multiple edges with same vertex set appear once."""
        store = HypergraphStore()
        store.add_node(Node("A", "entity"))
        store.add_node(Node("B", "entity"))
        # Two edges with same vertex set but different types
        store.add_edge(Hyperedge("e1", "type1", [Incidence("A"), Incidence("B")]))
        store.add_edge(Hyperedge("e2", "type2", [Incidence("A"), Incidence("B")]))
        tuples = store.get_edge_node_tuples_of_node("A")
        # frozenset deduplicates automatically
        assert tuples == {frozenset({"A", "B"})}


class TestHyperedgeDegree:
    """Tests for hyperedge_degree() method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Store with nodes of varying degrees."""
        s = HypergraphStore()
        # Create nodes
        for nid in ["A", "B", "C", "D"]:
            s.add_node(Node(nid, "entity"))
        # A is in 3 edges, B in 2, C in 1, D in 1
        s.add_edge(Hyperedge("e1", "rel", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e2", "rel", [Incidence("A"), Incidence("C")]))
        s.add_edge(Hyperedge("e3", "rel", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e4", "other", [Incidence("C"), Incidence("D")]))
        return s

    def test_sum_of_vertex_degrees(self, store):
        """Returns sum of degrees of participating vertices."""
        # e1 has {A, B}: degree(A)=3, degree(B)=2 → 5
        degree = store.hyperedge_degree({"A", "B"}, edge_type="rel")
        assert degree == 5

    def test_nonexistent_edge_returns_zero(self, store):
        """Non-existent edge returns 0."""
        degree = store.hyperedge_degree({"X", "Y"})
        assert degree == 0

    def test_filter_by_type(self, store):
        """Filter by edge type."""
        # {C, D} exists only as "other" type
        # C is in e2 (rel) and e4 (other), so degree(C)=2
        # D is in e4 only, so degree(D)=1
        degree = store.hyperedge_degree({"C", "D"}, edge_type="other")
        assert degree == 3  # degree(C)=2, degree(D)=1

        degree = store.hyperedge_degree({"C", "D"}, edge_type="rel")
        assert degree == 0  # No "rel" edge with {C, D}

    def test_without_type_filter(self, store):
        """Without type filter, finds any matching edge."""
        degree = store.hyperedge_degree({"C", "D"})
        assert degree == 3  # degree(C)=2, degree(D)=1

    def test_single_node_edge(self):
        """Single-node edge degree equals that node's degree."""
        store = HypergraphStore()
        store.add_node(Node("A", "entity"))
        store.add_edge(Hyperedge("e1", "singleton", [Incidence("A")]))
        store.add_edge(Hyperedge("e2", "pair", [Incidence("A"), Incidence("B")]))
        store.add_node(Node("B", "entity"))
        # A is in 2 edges
        degree = store.hyperedge_degree({"A"}, edge_type="singleton")
        assert degree == 2

    def test_highly_connected_edge(self):
        """Edge connecting hub nodes has high degree."""
        store = HypergraphStore()
        # Create hub nodes with many connections
        for i in range(10):
            store.add_node(Node(f"N{i}", "entity"))
        # Hub edges
        store.add_edge(Hyperedge("hub1", "rel", [Incidence("N0"), Incidence("N1")]))
        store.add_edge(Hyperedge("hub2", "rel", [Incidence("N0"), Incidence("N2")]))
        store.add_edge(Hyperedge("hub3", "rel", [Incidence("N0"), Incidence("N3")]))
        store.add_edge(Hyperedge("hub4", "rel", [Incidence("N1"), Incidence("N4")]))
        store.add_edge(Hyperedge("hub5", "rel", [Incidence("N1"), Incidence("N5")]))
        # Target edge with {N0, N1}: degree(N0)=3, degree(N1)=3 → 6
        degree = store.hyperedge_degree({"N0", "N1"})
        assert degree == 6


class TestCogRAGIntegration:
    """Integration tests simulating Cog-RAG usage patterns."""

    @pytest.fixture
    def cograg_store(self) -> HypergraphStore:
        """Store mimicking Cog-RAG entity hypergraph."""
        s = HypergraphStore()
        # Entities from document extraction
        entities = [
            ("JOHN", "person", {"description": "Main character"}),
            ("ACME_CORP", "organization", {"description": "Company"}),
            ("NEW_YORK", "location", {"description": "City"}),
            ("PROJECT_X", "event", {"description": "Secret project"}),
        ]
        for eid, etype, props in entities:
            s.add_node(Node(eid, etype, props))

        # Low-order relations (pairwise)
        s.add_edge(
            Hyperedge(
                "rel1",
                "low_order",
                [Incidence("JOHN"), Incidence("ACME_CORP")],
                properties={"description": "works at", "keywords": "employment"},
            )
        )
        s.add_edge(
            Hyperedge(
                "rel2",
                "low_order",
                [Incidence("JOHN"), Incidence("NEW_YORK")],
                properties={"description": "lives in", "keywords": "residence"},
            )
        )
        # High-order relation (3+ entities)
        s.add_edge(
            Hyperedge(
                "rel3",
                "high_order",
                [Incidence("JOHN"), Incidence("ACME_CORP"), Incidence("PROJECT_X")],
                properties={"description": "leads project at company", "keywords": "leadership"},
            )
        )
        return s

    def test_diffusion_from_entity(self, cograg_store):
        """Simulate Cog-RAG diffusion: get neighboring edges then their vertices."""
        # Step 1: Get edges containing JOHN
        edges = cograg_store.get_edges_of_node("JOHN")
        assert len(edges) == 3

        # Step 2: Get all vertices from those edges (1-hop neighbors)
        all_neighbors = set()
        for edge in edges:
            all_neighbors.update(edge.node_set)
        all_neighbors.discard("JOHN")  # Exclude self

        assert all_neighbors == {"ACME_CORP", "NEW_YORK", "PROJECT_X"}

    def test_edge_ranking_by_degree(self, cograg_store):
        """Rank edges by hyperedge_degree for retrieval prioritization."""
        # Get tuples of edges containing JOHN
        tuples = cograg_store.get_edge_node_tuples_of_node("JOHN")

        # Rank by hyperedge_degree
        ranked = sorted(
            tuples,
            key=lambda t: cograg_store.hyperedge_degree(set(t)),
            reverse=True,
        )

        # High-order edge {JOHN, ACME_CORP, PROJECT_X} should rank highest
        # because JOHN(3) + ACME_CORP(2) + PROJECT_X(1) = 6
        assert ranked[0] == frozenset({"JOHN", "ACME_CORP", "PROJECT_X"})

    def test_existence_check_before_merge(self, cograg_store):
        """Simulate Cog-RAG merge: check existence before upsert."""
        # During extraction, check if entity exists before merging
        assert cograg_store.has_node("JOHN") is True
        assert cograg_store.has_node("NEW_ENTITY") is False

        # If new, add it
        if not cograg_store.has_node("NEW_ENTITY"):
            cograg_store.add_node(Node("NEW_ENTITY", "concept"))

        assert cograg_store.has_node("NEW_ENTITY") is True

    def test_filter_edges_by_order(self, cograg_store):
        """Filter low-order vs high-order relations."""
        low_edges = cograg_store.get_edges_of_node("JOHN", edge_types=["low_order"])
        high_edges = cograg_store.get_edges_of_node("JOHN", edge_types=["high_order"])

        assert len(low_edges) == 2
        assert len(high_edges) == 1
        assert high_edges[0].id == "rel3"
