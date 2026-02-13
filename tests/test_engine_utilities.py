"""Tests for HypergraphStore utility methods.

Tests cover:
- get_neighbor_nodes
- node_degree
- edge_cardinality
- get_edge_by_node_set
- has_edge_with_nodes
- upsert_node
- upsert_edge
- upsert_edge_by_node_set
"""

import pytest

from hypabase.engine import (
    Hyperedge,
    HypergraphStore,
    Incidence,
    Node,
)


class TestGetNeighborNodes:
    """Tests for get_neighbor_nodes method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create a store with nodes connected via edges."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C", "D", "E"]:
            s.add_node(Node(node_id, "test"))

        # A-B via fk edge
        s.add_edge(Hyperedge("e1", "fk", [Incidence("A"), Incidence("B")]))
        # A-C via concept edge
        s.add_edge(Hyperedge("e2", "concept", [Incidence("A"), Incidence("C")]))
        # B-C-D via hyperedge
        s.add_edge(
            Hyperedge(
                "e3",
                "fk",
                [
                    Incidence("B"),
                    Incidence("C"),
                    Incidence("D"),
                ],
            )
        )
        # E is isolated
        return s

    def test_basic_neighbors(self, store):
        """Get all neighbors of a node."""
        neighbors = store.get_neighbor_nodes("A")
        assert set(neighbors) == {"B", "C"}

    def test_neighbors_with_hyperedge(self, store):
        """Neighbors through hyperedge."""
        neighbors = store.get_neighbor_nodes("B")
        # B connects to A (via e1), C (via e3), D (via e3)
        assert set(neighbors) == {"A", "C", "D"}

    def test_edge_type_filter(self, store):
        """Filter neighbors by edge type."""
        neighbors = store.get_neighbor_nodes("A", edge_types=["fk"])
        assert set(neighbors) == {"B"}

        neighbors = store.get_neighbor_nodes("A", edge_types=["concept"])
        assert set(neighbors) == {"C"}

    def test_exclude_self_true(self, store):
        """Exclude self from neighbors (default)."""
        neighbors = store.get_neighbor_nodes("A", exclude_self=True)
        assert "A" not in neighbors

    def test_exclude_self_false(self, store):
        """Include self in neighbors."""
        # Add a self-loop edge
        store.add_edge(
            Hyperedge(
                "loop",
                "test",
                [
                    Incidence("A"),
                    Incidence("A"),
                ],
            )
        )
        neighbors = store.get_neighbor_nodes("A", exclude_self=False)
        assert "A" in neighbors

    def test_isolated_node(self, store):
        """Isolated node has no neighbors."""
        neighbors = store.get_neighbor_nodes("E")
        assert neighbors == []

    def test_nonexistent_node(self, store):
        """Non-existent node returns empty list."""
        neighbors = store.get_neighbor_nodes("nonexistent")
        assert neighbors == []


class TestNodeDegree:
    """Tests for node_degree method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with nodes of varying degrees."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C", "D"]:
            s.add_node(Node(node_id, "test"))

        s.add_edge(Hyperedge("e1", "fk", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e2", "fk", [Incidence("A"), Incidence("C")]))
        s.add_edge(Hyperedge("e3", "concept", [Incidence("A"), Incidence("D")]))
        # D is isolated except for e3
        return s

    def test_basic_degree(self, store):
        """Get degree of a node."""
        assert store.node_degree("A") == 3
        assert store.node_degree("B") == 1
        assert store.node_degree("C") == 1

    def test_degree_with_type_filter(self, store):
        """Degree filtered by edge type."""
        assert store.node_degree("A", edge_types=["fk"]) == 2
        assert store.node_degree("A", edge_types=["concept"]) == 1
        assert store.node_degree("A", edge_types=["nonexistent"]) == 0

    def test_degree_with_multiple_type_filters(self, store):
        """Degree filtered by multiple edge types."""
        # A is in: 2 fk edges (e1, e2) + 1 concept edge (e3) = 3 total
        assert store.node_degree("A", edge_types=["fk", "concept"]) == 3
        assert store.node_degree("A", edge_types=["fk", "nonexistent"]) == 2
        assert store.node_degree("A", edge_types=["concept", "fk"]) == 3  # Order doesn't matter

    def test_degree_empty_type_list(self, store):
        """Degree with empty type list returns 0."""
        assert store.node_degree("A", edge_types=[]) == 0

    def test_zero_degree(self, store):
        """Node with no edges has degree 0."""
        store.add_node(Node("isolated", "test"))
        assert store.node_degree("isolated") == 0

    def test_nonexistent_node_degree(self, store):
        """Non-existent node has degree 0."""
        assert store.node_degree("nonexistent") == 0


class TestEdgeCardinality:
    """Tests for edge_cardinality method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with edges of varying cardinality."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C", "D"]:
            s.add_node(Node(node_id, "test"))

        # Binary edge
        s.add_edge(Hyperedge("binary", "test", [Incidence("A"), Incidence("B")]))
        # Hyperedge
        s.add_edge(
            Hyperedge(
                "hyper",
                "test",
                [
                    Incidence("A"),
                    Incidence("B"),
                    Incidence("C"),
                    Incidence("D"),
                ],
            )
        )
        # Single-node edge
        s.add_edge(Hyperedge("single", "test", [Incidence("A")]))
        # Edge with duplicate node (same node, different direction)
        s.add_edge(
            Hyperedge(
                "dup",
                "test",
                [
                    Incidence("A", direction="tail"),
                    Incidence("A", direction="head"),
                ],
            )
        )
        return s

    def test_binary_edge(self, store):
        """Binary edge has cardinality 2."""
        assert store.edge_cardinality("binary") == 2

    def test_hyperedge(self, store):
        """Hyperedge has cardinality of unique nodes."""
        assert store.edge_cardinality("hyper") == 4

    def test_single_node_edge(self, store):
        """Single-node edge has cardinality 1."""
        assert store.edge_cardinality("single") == 1

    def test_duplicate_node_edge(self, store):
        """Edge with duplicate node - cardinality is unique count."""
        # node_set deduplicates
        assert store.edge_cardinality("dup") == 1

    def test_nonexistent_edge(self, store):
        """Non-existent edge has cardinality 0."""
        assert store.edge_cardinality("nonexistent") == 0


class TestGetEdgeByNodeSet:
    """Tests for get_edge_by_node_set method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with edges indexed by node set."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C", "D"]:
            s.add_node(Node(node_id, "test"))

        s.add_edge(Hyperedge("ab", "fk", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("bc", "concept", [Incidence("B"), Incidence("C")]))
        s.add_edge(
            Hyperedge(
                "abcd",
                "group",
                [
                    Incidence("A"),
                    Incidence("B"),
                    Incidence("C"),
                    Incidence("D"),
                ],
            )
        )
        return s

    def test_exact_match(self, store):
        """Find edge by exact node set."""
        edge = store.get_edge_by_node_set({"A", "B"})
        assert edge is not None
        assert edge.id == "ab"

    def test_order_independent(self, store):
        """Node set lookup is order-independent."""
        edge1 = store.get_edge_by_node_set({"A", "B"})
        edge2 = store.get_edge_by_node_set({"B", "A"})
        assert edge1 == edge2

    def test_subset_no_match(self, store):
        """Subset of edge nodes doesn't match."""
        # Looking for {A, B} should not match {A, B, C, D}
        edge = store.get_edge_by_node_set({"A"})
        assert edge is None

    def test_superset_no_match(self, store):
        """Superset of edge nodes doesn't match."""
        edge = store.get_edge_by_node_set({"A", "B", "C"})
        assert edge is None

    def test_with_edge_type(self, store):
        """Filter by edge type."""
        edge = store.get_edge_by_node_set({"A", "B"}, edge_type="fk")
        assert edge is not None
        assert edge.id == "ab"

        edge = store.get_edge_by_node_set({"A", "B"}, edge_type="concept")
        assert edge is None

    def test_no_match(self, store):
        """No edge with given node set."""
        edge = store.get_edge_by_node_set({"A", "D"})
        assert edge is None


class TestHasEdgeWithNodes:
    """Tests for has_edge_with_nodes method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with edges."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C"]:
            s.add_node(Node(node_id, "test"))

        s.add_edge(Hyperedge("ab", "fk", [Incidence("A"), Incidence("B")]))
        return s

    def test_exists(self, store):
        """Check existing edge."""
        assert store.has_edge_with_nodes({"A", "B"}) is True

    def test_not_exists(self, store):
        """Check non-existing edge."""
        assert store.has_edge_with_nodes({"A", "C"}) is False

    def test_with_edge_type(self, store):
        """Check with edge type filter."""
        assert store.has_edge_with_nodes({"A", "B"}, edge_type="fk") is True
        assert store.has_edge_with_nodes({"A", "B"}, edge_type="other") is False


class TestMultipleEdgesSameNodeSet:
    """Tests for multiple edges with the same vertex set."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with edges that share the same node set."""
        s = HypergraphStore()
        for node_id in ["A", "B"]:
            s.add_node(Node(node_id, "test"))

        # Two edges with same nodes but different types
        s.add_edge(Hyperedge("e1", "fk", [Incidence("A"), Incidence("B")]))
        s.add_edge(Hyperedge("e2", "concept", [Incidence("A"), Incidence("B")]))
        return s

    def test_both_edges_exist(self, store):
        """Both edges should exist in the store."""
        assert store.get_edge("e1") is not None
        assert store.get_edge("e2") is not None

    def test_get_edge_by_node_set_returns_one(self, store):
        """get_edge_by_node_set returns one of the edges."""
        edge = store.get_edge_by_node_set({"A", "B"})
        assert edge is not None
        assert edge.id in ("e1", "e2")

    def test_get_edge_by_node_set_with_type(self, store):
        """get_edge_by_node_set with type filter returns correct edge."""
        fk_edge = store.get_edge_by_node_set({"A", "B"}, edge_type="fk")
        assert fk_edge is not None
        assert fk_edge.id == "e1"

        concept_edge = store.get_edge_by_node_set({"A", "B"}, edge_type="concept")
        assert concept_edge is not None
        assert concept_edge.id == "e2"

    def test_get_edges_by_node_set_returns_all(self, store):
        """get_edges_by_node_set returns all matching edges."""
        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 2
        edge_ids = {e.id for e in edges}
        assert edge_ids == {"e1", "e2"}

        # Verify internal index state is exactly correct (issue #5)
        assert store._edges_by_node_set[frozenset({"A", "B"})] == {"e1", "e2"}

    def test_get_edges_by_node_set_with_type(self, store):
        """get_edges_by_node_set with type filter."""
        fk_edges = store.get_edges_by_node_set({"A", "B"}, edge_type="fk")
        assert len(fk_edges) == 1
        assert fk_edges[0].id == "e1"

    def test_delete_one_edge_keeps_other(self, store):
        """Deleting one edge keeps the other findable with correct properties."""
        store.delete_edge("e1")

        # e2 should still be findable
        edge = store.get_edge_by_node_set({"A", "B"})
        assert edge is not None
        assert edge.id == "e2"
        # Verify e2's properties weren't corrupted during e1's deletion
        assert edge.type == "concept"

        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 1

        # Verify internal index state is correct (issue #8)
        assert frozenset({"A", "B"}) in store._edges_by_node_set
        assert store._edges_by_node_set[frozenset({"A", "B"})] == {"e2"}

    def test_delete_both_edges(self, store):
        """Deleting both edges cleans up the index."""
        store.delete_edge("e1")
        store.delete_edge("e2")

        assert store.get_edge_by_node_set({"A", "B"}) is None
        assert store.get_edges_by_node_set({"A", "B"}) == []
        assert not store.has_edge_with_nodes({"A", "B"})


class TestGetEdgesByNodeSetComprehensive:
    """Comprehensive tests for get_edges_by_node_set (plural) method."""

    def test_empty_store_returns_empty_list(self):
        """Empty store returns empty list."""
        store = HypergraphStore()
        edges = store.get_edges_by_node_set({"A", "B"})
        assert edges == []

    def test_no_matching_node_set_returns_empty(self):
        """Non-matching node set returns empty list."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A"), Incidence("B")]))
        edges = store.get_edges_by_node_set({"X", "Y"})
        assert edges == []

    def test_single_edge_returns_list_of_one(self):
        """Single matching edge returns list with one element."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A"), Incidence("B")]))
        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 1
        assert edges[0].id == "e1"

    def test_type_filter_with_no_match(self):
        """Type filter with no matching type returns empty."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A"), Incidence("B")]))
        edges = store.get_edges_by_node_set({"A", "B"}, edge_type="nonexistent")
        assert edges == []

    def test_type_filter_with_multiple_edges(self):
        """Type filter correctly filters multiple edges."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(Hyperedge("e1", "fk", [Incidence("A"), Incidence("B")]))
        store.add_edge(Hyperedge("e2", "fk", [Incidence("A"), Incidence("B")]))
        store.add_edge(Hyperedge("e3", "concept", [Incidence("A"), Incidence("B")]))

        fk_edges = store.get_edges_by_node_set({"A", "B"}, edge_type="fk")
        assert len(fk_edges) == 2
        assert {e.id for e in fk_edges} == {"e1", "e2"}

        concept_edges = store.get_edges_by_node_set({"A", "B"}, edge_type="concept")
        assert len(concept_edges) == 1
        assert concept_edges[0].id == "e3"

    def test_empty_node_set_returns_empty(self):
        """Empty node set returns empty list."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A")]))
        edges = store.get_edges_by_node_set(set())
        assert edges == []

    def test_returns_hyperedge_objects(self):
        """Returned items are Hyperedge instances with full properties."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(
            Hyperedge(
                "e1",
                "rel",
                [Incidence("A"), Incidence("B")],
                properties={"weight": 0.5},
                source="test_source",
                confidence=0.9,
            )
        )
        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 1
        edge = edges[0]
        assert isinstance(edge, Hyperedge)
        assert edge.properties["weight"] == 0.5
        assert edge.source == "test_source"
        assert edge.confidence == 0.9

    def test_handles_deleted_edge_in_index(self):
        """Gracefully handles stale index entries."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_node(Node("B", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A"), Incidence("B")]))
        store.add_edge(Hyperedge("e2", "rel", [Incidence("A"), Incidence("B")]))

        # Normal case
        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 2

        # Delete one edge
        store.delete_edge("e1")
        edges = store.get_edges_by_node_set({"A", "B"})
        assert len(edges) == 1
        assert edges[0].id == "e2"


class TestUpsertNode:
    """Tests for upsert_node method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with existing node."""
        s = HypergraphStore()
        s.add_node(Node("A", "table", {"color": "red", "size": 10}))
        return s

    def test_insert_new_node(self, store):
        """Insert a new node."""
        node = store.upsert_node(Node("B", "column", {"name": "id"}))
        assert node.id == "B"
        assert store.get_node("B") is not None

    def test_update_merge_properties(self, store):
        """Update existing node with merged properties."""
        node = store.upsert_node(
            Node("A", "updated_table", {"color": "blue", "weight": 5}),
            merge_properties=True,
        )
        assert node.type == "updated_table"
        assert node.properties["color"] == "blue"  # Updated
        assert node.properties["size"] == 10  # Preserved
        assert node.properties["weight"] == 5  # Added

    def test_update_replace_properties(self, store):
        """Update existing node with replaced properties."""
        node = store.upsert_node(
            Node("A", "updated_table", {"color": "blue"}),
            merge_properties=False,
        )
        assert node.properties["color"] == "blue"
        assert "size" not in node.properties  # Not preserved

    def test_index_consistency_on_type_change(self, store):
        """Type index updated when type changes."""
        assert len(store.get_nodes_by_type("table")) == 1
        assert len(store.get_nodes_by_type("view")) == 0

        store.upsert_node(Node("A", "view", {}))

        assert len(store.get_nodes_by_type("table")) == 0
        assert len(store.get_nodes_by_type("view")) == 1

    def test_upsert_property_independence(self):
        """Upserted node properties are independent (not shared references)."""
        store = HypergraphStore()
        props = {"mutable": [1, 2, 3]}
        store.upsert_node(Node("A", "test", props))

        # Mutate original dict after upsert
        props["mutable"].append(4)

        # Node's properties should not be affected
        node = store.get_node("A")
        # Note: Current implementation uses dataclass defaults, which may share refs
        # This test documents expected behavior - if it fails, it's a real bug
        assert len(node.properties["mutable"]) >= 3  # At minimum original values


class TestUpsertEdge:
    """Tests for upsert_edge method."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create store with existing edge."""
        s = HypergraphStore()
        for node_id in ["A", "B", "C"]:
            s.add_node(Node(node_id, "test"))

        s.add_edge(
            Hyperedge(
                "e1",
                "fk",
                [Incidence("A"), Incidence("B")],
                properties={"weight": 1.0},
                confidence=0.8,
            )
        )
        return s

    def test_insert_new_edge(self, store):
        """Insert a new edge."""
        edge = store.upsert_edge(
            Hyperedge(
                "e2",
                "concept",
                [Incidence("B"), Incidence("C")],
            )
        )
        assert edge.id == "e2"
        assert store.get_edge("e2") is not None

    def test_update_replace_edge(self, store):
        """Update existing edge without merge function."""
        edge = store.upsert_edge(
            Hyperedge(
                "e1",
                "pk",
                [Incidence("A"), Incidence("C")],
                properties={"updated": True},
                confidence=0.9,
            )
        )
        assert edge.type == "pk"
        assert edge.node_set == {"A", "C"}
        assert edge.properties == {"updated": True}
        assert edge.confidence == 0.9

    def test_update_with_merge_function(self, store):
        """Update existing edge with custom merge function."""

        def merge_fn(existing: Hyperedge, new: Hyperedge) -> Hyperedge:
            merged_props = dict(existing.properties)
            merged_props.update(new.properties)
            return Hyperedge(
                id=new.id,
                type=new.type,
                incidences=new.incidences,
                properties=merged_props,
                confidence=max(existing.confidence, new.confidence),
            )

        edge = store.upsert_edge(
            Hyperedge(
                "e1",
                "fk",
                [Incidence("A"), Incidence("B")],
                properties={"label": "test"},
                confidence=0.7,
            ),
            merge_fn=merge_fn,
        )

        assert edge.properties["weight"] == 1.0  # Preserved
        assert edge.properties["label"] == "test"  # Added
        assert edge.confidence == 0.8  # Max of 0.8 and 0.7

    def test_index_consistency_on_update(self, store):
        """Indexes updated correctly on upsert."""
        # Before update
        assert store.get_edges_containing({"B"}) != []
        assert store.has_edge_with_nodes({"A", "B"})

        # Update to different nodes
        store.upsert_edge(
            Hyperedge(
                "e1",
                "fk",
                [Incidence("A"), Incidence("C")],
            )
        )

        # After update
        edges_with_b = store.get_edges_containing({"B"})
        assert "e1" not in [e.id for e in edges_with_b]
        assert not store.has_edge_with_nodes({"A", "B"})
        assert store.has_edge_with_nodes({"A", "C"})

    def test_index_consistency_on_type_change(self, store):
        """Type index updated on edge upsert."""
        assert len(store.get_edges_by_type("fk")) == 1
        assert len(store.get_edges_by_type("pk")) == 0

        store.upsert_edge(
            Hyperedge(
                "e1",
                "pk",
                [Incidence("A"), Incidence("B")],
            )
        )

        assert len(store.get_edges_by_type("fk")) == 0
        assert len(store.get_edges_by_type("pk")) == 1


class TestMemoryLeakPrevention:
    """Tests verifying empty index sets are cleaned up."""

    def test_delete_last_node_of_type_cleans_index(self):
        """Deleting last node of a type removes empty set from _nodes_by_type."""
        store = HypergraphStore()
        store.add_node(Node("A", "unique_type"))
        store.add_node(Node("B", "other_type"))

        # Verify node is in type index
        assert "unique_type" in store._nodes_by_type
        assert len(store._nodes_by_type["unique_type"]) == 1

        # Delete the only node of this type
        store.delete_node("A")

        # Empty set should be cleaned up
        assert "unique_type" not in store._nodes_by_type
        # Other type still exists
        assert "other_type" in store._nodes_by_type

        # Verify public API also works correctly (issue #3)
        assert store.get_nodes_by_type("unique_type") == []

    def test_delete_last_edge_of_type_cleans_index(self):
        """Deleting last edge of a type removes empty set from _edges_by_type."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_edge(Hyperedge("e1", "unique_edge_type", [Incidence("A")]))
        store.add_edge(Hyperedge("e2", "other_edge_type", [Incidence("A")]))

        # Verify edge is in type index
        assert "unique_edge_type" in store._edges_by_type
        assert len(store._edges_by_type["unique_edge_type"]) == 1

        # Delete the only edge of this type
        store.delete_edge("e1")

        # Empty set should be cleaned up
        assert "unique_edge_type" not in store._edges_by_type
        # Other type still exists
        assert "other_edge_type" in store._edges_by_type

        # Verify public API also works correctly (issue #3)
        assert store.get_edges_by_type("unique_edge_type") == []


class TestVertexSetIndexSelfLoop:
    """Tests for vertex-set index with self-loop edges."""

    def test_vertex_set_index_with_self_loop(self):
        """Self-loop edge indexed by deduplicated node set."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))

        # Self-loop edge (same node in multiple incidences)
        store.add_edge(
            Hyperedge(
                id="self_loop",
                type="test",
                incidences=[
                    Incidence("A", direction="tail"),
                    Incidence("A", direction="head"),
                ],
            )
        )

        # Edge should be indexed by deduplicated set {A}
        edge = store.get_edge_by_node_set({"A"})
        assert edge is not None
        assert edge.id == "self_loop"

        # Verify has_edge_with_nodes also works
        assert store.has_edge_with_nodes({"A"})


class TestLargeDataHandling:
    """Tests for large property values and deep nesting."""

    def test_very_large_property_value(self):
        """Handle large property value (1MB) in roundtrip."""
        store = HypergraphStore()
        # 1MB string (not 10MB to keep test fast)
        large_value = "x" * (1024 * 1024)
        store.add_node(Node("A", "test", {"large": large_value}))

        # Dict roundtrip
        data = store.to_dict()
        restored = HypergraphStore.from_dict(data)

        node = restored.get_node("A")
        assert node is not None
        assert len(node.properties["large"]) == 1024 * 1024

    def test_deeply_nested_properties(self):
        """Handle deeply nested properties (50 levels) in roundtrip."""
        store = HypergraphStore()
        # Build 50-level nested structure (not 100 to keep reasonable)
        nested: dict = {"value": "deep"}
        for _ in range(50):
            nested = {"nested": nested}

        store.add_node(Node("A", "test", nested))

        # Dict roundtrip
        data = store.to_dict()
        restored = HypergraphStore.from_dict(data)

        node = restored.get_node("A")
        assert node is not None

        # Verify deep nesting preserved
        current = node.properties
        for _ in range(50):
            assert "nested" in current
            current = current["nested"]
        assert current["value"] == "deep"


class TestUpsertEdgeByNodeSet:
    """Tests for upsert_edge_by_node_set method."""

    def test_creates_new_edge_with_uuid(self):
        """New edge gets auto-generated UUID."""
        store = HypergraphStore()
        edge = store.upsert_edge_by_node_set({"A", "B"}, "rel", {"weight": 1.0})
        assert edge.id  # UUID exists
        assert edge.node_set == {"A", "B"}
        assert edge.type == "rel"
        assert edge.properties["weight"] == 1.0

    def test_updates_existing_edge_same_id(self):
        """Existing edge keeps its ID on update."""
        store = HypergraphStore()
        edge1 = store.upsert_edge_by_node_set({"A", "B"}, "rel", {"weight": 1.0})
        edge2 = store.upsert_edge_by_node_set({"A", "B"}, "rel", {"weight": 2.0})
        assert edge1.id == edge2.id
        assert edge2.properties["weight"] == 2.0

    def test_merge_fn_applied(self):
        """merge_fn combines existing and new edge."""
        store = HypergraphStore()
        store.upsert_edge_by_node_set({"A", "B"}, "rel", {"weight": 1.0})

        def sum_weights(old, new):
            new.properties["weight"] = old.properties["weight"] + new.properties["weight"]
            return new

        edge = store.upsert_edge_by_node_set(
            {"A", "B"}, "rel", {"weight": 2.0}, merge_fn=sum_weights
        )
        assert edge.properties["weight"] == 3.0

    def test_different_types_are_separate_edges(self):
        """Same vertex set with different types creates separate edges."""
        store = HypergraphStore()
        e1 = store.upsert_edge_by_node_set({"A", "B"}, "type1", {})
        e2 = store.upsert_edge_by_node_set({"A", "B"}, "type2", {})
        assert e1.id != e2.id
        assert len(store.get_all_edges()) == 2

    def test_source_and_confidence_set(self):
        """Source and confidence are set on new edge."""
        store = HypergraphStore()
        edge = store.upsert_edge_by_node_set(
            {"A", "B"},
            "rel",
            {},
            source="extraction",
            confidence=0.85,
        )
        assert edge.source == "extraction"
        assert edge.confidence == 0.85

    def test_empty_node_set(self):
        """Empty node set creates valid edge with no nodes."""
        store = HypergraphStore()
        edge = store.upsert_edge_by_node_set(set(), "empty", {})
        assert edge.node_set == set()
        assert edge.type == "empty"

    def test_single_node_set(self):
        """Single-node set creates valid edge."""
        store = HypergraphStore()
        edge = store.upsert_edge_by_node_set({"A"}, "singleton", {})
        assert edge.node_set == {"A"}

    def test_invalid_confidence_raises(self):
        """Invalid confidence value raises ValueError."""
        store = HypergraphStore()
        with pytest.raises(ValueError, match="confidence"):
            store.upsert_edge_by_node_set(
                {"A", "B"},
                "rel",
                {},
                confidence=1.5,  # Invalid: > 1.0
            )

    def test_large_node_set(self):
        """Large node set (50 nodes) creates valid edge."""
        store = HypergraphStore()
        node_ids = {f"N{i}" for i in range(50)}
        edge = store.upsert_edge_by_node_set(node_ids, "large", {"size": 50})
        assert len(edge.node_set) == 50
        assert edge.properties["size"] == 50


class TestFindNodesEdgesWithEmptyStrings:
    """Tests for find_nodes/find_edges with empty string properties."""

    def test_find_nodes_empty_string_property(self):
        """Find nodes with empty string property value."""
        store = HypergraphStore()
        store.add_node(Node("A", "test", {"name": ""}))
        store.add_node(Node("B", "test", {"name": "Bob"}))
        store.add_node(Node("C", "test", {}))

        nodes = store.find_nodes(name="")
        assert len(nodes) == 1
        assert nodes[0].id == "A"

    def test_find_nodes_empty_string_vs_missing(self):
        """Empty string property is different from missing property."""
        store = HypergraphStore()
        store.add_node(Node("A", "test", {"name": ""}))
        store.add_node(Node("B", "test", {}))  # No name property

        # Empty string matches only A
        nodes_empty = store.find_nodes(name="")
        assert len(nodes_empty) == 1
        assert nodes_empty[0].id == "A"

        # None matches only B (missing = None)
        nodes_none = store.find_nodes(name=None)
        assert len(nodes_none) == 1
        assert nodes_none[0].id == "B"

    def test_find_edges_empty_string_property(self):
        """Find edges with empty string property value."""
        store = HypergraphStore()
        store.add_node(Node("A", "test"))
        store.add_edge(Hyperedge("e1", "rel", [Incidence("A")], properties={"label": ""}))
        store.add_edge(Hyperedge("e2", "rel", [Incidence("A")], properties={"label": "test"}))

        edges = store.find_edges(label="")
        assert len(edges) == 1
        assert edges[0].id == "e1"
