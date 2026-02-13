"""Tests for core hypergraph data structures and operations."""

import copy
import json

import pytest

from hypabase.engine.core import (
    Hyperedge,
    HypergraphStore,
    Incidence,
    Node,
)


class TestNode:
    """Tests for Node dataclass."""

    def test_basic_node(self):
        node = Node(id="customers", type="table")
        assert node.id == "customers"
        assert node.type == "table"
        assert node.properties == {}

    def test_node_with_properties(self):
        node = Node(
            id="customers.id",
            type="column",
            properties={"data_type": "INTEGER", "primary_key": True},
        )
        assert node.properties["data_type"] == "INTEGER"
        assert node.properties["primary_key"] is True


class TestIncidence:
    """Tests for Incidence dataclass."""

    def test_undirected_incidence(self):
        inc = Incidence(node_id="customers.id")
        assert inc.node_id == "customers.id"
        assert inc.direction is None
        assert inc.properties == {}

    def test_directed_incidence_head(self):
        inc = Incidence(node_id="customers.id", direction="head")
        assert inc.direction == "head"

    def test_directed_incidence_tail(self):
        inc = Incidence(node_id="orders.customer_id", direction="tail")
        assert inc.direction == "tail"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            Incidence(node_id="x", direction="invalid")

    def test_incidence_with_properties(self):
        inc = Incidence(
            node_id="orders.amount",
            properties={"role": "measure"},
        )
        assert inc.properties["role"] == "measure"


class TestHyperedge:
    """Tests for Hyperedge dataclass."""

    def test_undirected_edge(self):
        edge = Hyperedge(
            id="revenue_mapping",
            type="concept_mapping",
            incidences=[
                Incidence("revenue"),
                Incidence("orders.amount"),
            ],
        )
        assert edge.id == "revenue_mapping"
        assert edge.type == "concept_mapping"
        assert edge.nodes == ["revenue", "orders.amount"]
        assert edge.node_set == {"revenue", "orders.amount"}
        assert edge.is_directed is False
        assert edge.head_nodes == []
        assert edge.tail_nodes == []

    def test_directed_edge(self):
        edge = Hyperedge(
            id="fk_orders_customers",
            type="foreign_key",
            incidences=[
                Incidence("orders.customer_id", direction="tail"),
                Incidence("customers.id", direction="head"),
            ],
        )
        assert edge.is_directed is True
        assert edge.tail_nodes == ["orders.customer_id"]
        assert edge.head_nodes == ["customers.id"]
        assert edge.nodes == ["orders.customer_id", "customers.id"]

    def test_edge_with_provenance(self):
        edge = Hyperedge(
            id="e1",
            type="test",
            incidences=[Incidence("a")],
            source="schema",
            confidence=0.95,
        )
        assert edge.source == "schema"
        assert edge.confidence == 0.95

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence must be"):
            Hyperedge(id="e1", type="test", incidences=[], confidence=1.5)

    def test_edge_with_properties(self):
        edge = Hyperedge(
            id="metric_revenue",
            type="concept_mapping",
            incidences=[Incidence("revenue"), Incidence("orders.amount")],
            properties={"aggregation": "sum", "sql_template": "SUM(orders.amount)"},
        )
        assert edge.properties["aggregation"] == "sum"
        assert edge.properties["sql_template"] == "SUM(orders.amount)"


class TestHypergraphStore:
    """Tests for HypergraphStore operations."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create a test store with sample data."""
        s = HypergraphStore()

        # Add nodes
        s.add_node(Node("customers", "table", {"description": "Customer accounts"}))
        s.add_node(Node("orders", "table", {"description": "Customer orders"}))
        s.add_node(Node("products", "table"))
        s.add_node(Node("customers.id", "column", {"table": "customers", "data_type": "INTEGER"}))
        s.add_node(Node("customers.name", "column", {"table": "customers", "data_type": "VARCHAR"}))
        s.add_node(Node("orders.id", "column", {"table": "orders", "data_type": "INTEGER"}))
        s.add_node(
            Node("orders.customer_id", "column", {"table": "orders", "data_type": "INTEGER"})
        )
        s.add_node(Node("orders.amount", "column", {"table": "orders", "data_type": "DECIMAL"}))
        s.add_node(Node("revenue", "concept", {"synonyms": ["sales", "income"]}))

        # Add edges
        s.add_edge(
            Hyperedge(
                id="fk_orders_customers",
                type="foreign_key",
                incidences=[
                    Incidence("orders.customer_id", direction="tail"),
                    Incidence("customers.id", direction="head"),
                ],
                source="schema",
                confidence=1.0,
            )
        )
        s.add_edge(
            Hyperedge(
                id="revenue_mapping",
                type="concept_mapping",
                incidences=[
                    Incidence("revenue"),
                    Incidence("orders.amount"),
                ],
                properties={"aggregation": "sum"},
                source="docs",
                confidence=0.95,
            )
        )

        return s

    # === Node Operations ===

    def test_add_and_get_node(self, store: HypergraphStore):
        node = store.get_node("customers")
        assert node is not None
        assert node.type == "table"
        assert node.properties["description"] == "Customer accounts"

    def test_get_nonexistent_node(self, store: HypergraphStore):
        assert store.get_node("nonexistent") is None

    def test_get_nodes_by_type(self, store: HypergraphStore):
        tables = store.get_nodes_by_type("table")
        assert len(tables) == 3
        table_ids = {t.id for t in tables}
        assert table_ids == {"customers", "orders", "products"}

    def test_find_nodes_by_properties(self, store: HypergraphStore):
        columns = store.find_nodes(table="customers")
        assert len(columns) == 2
        column_ids = {c.id for c in columns}
        assert column_ids == {"customers.id", "customers.name"}

    def test_delete_node(self, store: HypergraphStore):
        assert store.delete_node("products") is True
        assert store.get_node("products") is None
        assert len(store.get_nodes_by_type("table")) == 2

    def test_delete_nonexistent_node(self, store: HypergraphStore):
        assert store.delete_node("nonexistent") is False

    def test_get_all_nodes(self, store: HypergraphStore):
        nodes = store.get_all_nodes()
        assert len(nodes) == 9

    # === Edge Operations ===

    def test_add_and_get_edge(self, store: HypergraphStore):
        edge = store.get_edge("fk_orders_customers")
        assert edge is not None
        assert edge.type == "foreign_key"
        assert edge.source == "schema"
        assert edge.confidence == 1.0

    def test_get_nonexistent_edge(self, store: HypergraphStore):
        assert store.get_edge("nonexistent") is None

    def test_get_edges_by_type(self, store: HypergraphStore):
        fks = store.get_edges_by_type("foreign_key")
        assert len(fks) == 1
        assert fks[0].id == "fk_orders_customers"

    def test_get_edges_containing_any(self, store: HypergraphStore):
        edges = store.get_edges_containing({"orders.amount"}, match_all=False)
        assert len(edges) == 1
        assert edges[0].id == "revenue_mapping"

    def test_get_edges_containing_all(self, store: HypergraphStore):
        # Edge must contain BOTH nodes
        edges = store.get_edges_containing(
            {"orders.customer_id", "customers.id"},
            match_all=True,
        )
        assert len(edges) == 1
        assert edges[0].id == "fk_orders_customers"

    def test_get_edges_containing_all_no_match(self, store: HypergraphStore):
        # No edge contains both these nodes
        edges = store.get_edges_containing(
            {"orders.amount", "customers.name"},
            match_all=True,
        )
        assert len(edges) == 0

    def test_find_edges_by_properties(self, store: HypergraphStore):
        edges = store.find_edges(aggregation="sum")
        assert len(edges) == 1
        assert edges[0].id == "revenue_mapping"

    def test_delete_edge(self, store: HypergraphStore):
        assert store.delete_edge("revenue_mapping") is True
        assert store.get_edge("revenue_mapping") is None
        # Node should still have no edges referencing it
        edges = store.get_edges_containing({"orders.amount"})
        assert len(edges) == 0

    def test_delete_nonexistent_edge(self, store: HypergraphStore):
        assert store.delete_edge("nonexistent") is False

    def test_get_all_edges(self, store: HypergraphStore):
        edges = store.get_all_edges()
        assert len(edges) == 2

    # === Statistics ===

    def test_stats(self, store: HypergraphStore):
        stats = store.stats()
        assert stats["num_nodes"] == 9
        assert stats["num_edges"] == 2
        assert stats["nodes_by_type"]["table"] == 3
        assert stats["nodes_by_type"]["column"] == 5
        assert stats["nodes_by_type"]["concept"] == 1
        assert stats["edges_by_type"]["foreign_key"] == 1
        assert stats["edges_by_type"]["concept_mapping"] == 1


class TestPathFinding:
    """Tests for path finding with intersection constraints."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create a store with a chain of connected edges for path finding tests.

        The edges include both table nodes and column nodes so they share
        the table node for path connectivity (IS=1).

        Chain: customers <-> orders <-> order_items <-> products
        """
        s = HypergraphStore()

        # Nodes: Tables and their columns
        for table in ["customers", "orders", "order_items", "products"]:
            s.add_node(Node(table, "table"))
            s.add_node(Node(f"{table}.id", "column", {"table": table}))

        s.add_node(Node("orders.customer_id", "column", {"table": "orders"}))
        s.add_node(Node("order_items.order_id", "column", {"table": "order_items"}))
        s.add_node(Node("order_items.product_id", "column", {"table": "order_items"}))

        # Foreign key edges include table nodes for path connectivity
        # Edge: orders.customer_id -> customers.id (connects orders and customers tables)
        s.add_edge(
            Hyperedge(
                id="fk_orders_customers",
                type="foreign_key",
                incidences=[
                    Incidence("orders", direction="tail"),  # Source table
                    Incidence("orders.customer_id", direction="tail"),
                    Incidence("customers", direction="head"),  # Target table
                    Incidence("customers.id", direction="head"),
                ],
            )
        )
        # Edge: order_items.order_id -> orders.id (connects order_items and orders tables)
        s.add_edge(
            Hyperedge(
                id="fk_order_items_orders",
                type="foreign_key",
                incidences=[
                    Incidence("order_items", direction="tail"),
                    Incidence("order_items.order_id", direction="tail"),
                    Incidence("orders", direction="head"),  # Shared with fk_orders_customers
                    Incidence("orders.id", direction="head"),
                ],
            )
        )
        # Edge: order_items.product_id -> products.id
        s.add_edge(
            Hyperedge(
                id="fk_order_items_products",
                type="foreign_key",
                incidences=[
                    Incidence("order_items", direction="tail"),
                    Incidence("order_items.product_id", direction="tail"),
                    Incidence("products", direction="head"),
                    Incidence("products.id", direction="head"),
                ],
            )
        )

        return s

    def test_find_direct_path(self, store: HypergraphStore):
        """Find a path between directly connected tables."""
        paths = store.find_paths(
            start_nodes={"orders"},
            end_nodes={"customers"},
        )
        assert len(paths) >= 1
        # Should find the FK edge
        assert any(p[0].id == "fk_orders_customers" for p in paths)

    def test_find_multi_hop_path(self, store: HypergraphStore):
        """Find a path requiring multiple hops."""
        # From order_items to customers requires 2 hops
        # order_items -> orders (via fk_order_items_orders) -> customers (via fk_orders_customers)
        paths = store.find_paths(
            start_nodes={"order_items"},
            end_nodes={"customers"},
            max_hops=3,
        )
        assert len(paths) >= 1
        # Path should be: fk_order_items_orders -> fk_orders_customers
        path = paths[0]
        assert len(path) == 2
        edge_ids = {e.id for e in path}
        assert edge_ids == {"fk_order_items_orders", "fk_orders_customers"}

    def test_no_path_found(self, store: HypergraphStore):
        """No path exists between disconnected nodes."""
        # Add an isolated node
        store.add_node(Node("isolated", "table"))

        paths = store.find_paths(
            start_nodes={"customers"},
            end_nodes={"isolated"},
        )
        assert len(paths) == 0

    def test_max_hops_limits_search(self, store: HypergraphStore):
        """max_hops parameter limits path length."""
        # From order_items to customers requires 2 hops
        # With max_hops=1, should not find path
        paths = store.find_paths(
            start_nodes={"order_items"},
            end_nodes={"customers"},
            max_hops=1,
        )
        assert len(paths) == 0

    def test_edge_type_filter(self, store: HypergraphStore):
        """edge_types parameter filters which edges to traverse."""
        # Add a non-FK edge connecting orders and customers
        store.add_edge(
            Hyperedge(
                id="other_edge",
                type="other",
                incidences=[
                    Incidence("customers"),
                    Incidence("orders"),
                ],
            )
        )

        # Should find path using only FK edges
        paths = store.find_paths(
            start_nodes={"orders"},
            end_nodes={"customers"},
            edge_types=["foreign_key"],
        )
        assert len(paths) >= 1
        for path in paths:
            for edge in path:
                assert edge.type == "foreign_key"

    def test_min_intersection_constraint(self, store: HypergraphStore):
        """min_intersection parameter requires more shared nodes."""
        # With IS=2, edges need to share 2 nodes to be adjacent
        # fk_order_items_orders and fk_orders_customers share only 1 node (orders)
        # So no multi-hop path should be found
        paths = store.find_paths(
            start_nodes={"order_items"},
            end_nodes={"customers"},
            min_intersection=2,
        )
        # Should not find a path since edges only share 1 node (orders)
        assert len(paths) == 0


class TestSerialization:
    """Tests for hypergraph serialization."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create a simple test store."""
        s = HypergraphStore()
        s.add_node(Node("customers", "table", {"description": "Customer data"}))
        s.add_node(Node("orders", "table"))
        s.add_node(Node("revenue", "concept", {"synonyms": ["sales"]}))
        s.add_edge(
            Hyperedge(
                id="fk_test",
                type="foreign_key",
                incidences=[
                    Incidence("customers", direction="head"),
                    Incidence("orders", direction="tail"),
                ],
                properties={"join_type": "inner"},
                source="schema",
                confidence=0.9,
            )
        )
        return s

    def test_to_dict_and_back(self, store: HypergraphStore):
        """Round-trip through dict serialization."""
        data = store.to_dict()
        restored = HypergraphStore.from_dict(data)

        assert restored.stats() == store.stats()
        assert restored.get_node("customers").properties["description"] == "Customer data"
        edge = restored.get_edge("fk_test")
        assert edge.type == "foreign_key"
        assert edge.source == "schema"
        assert edge.confidence == 0.9
        assert edge.properties["join_type"] == "inner"

    def test_to_hif_and_back(self, store: HypergraphStore):
        """Round-trip through HIF serialization."""
        hif_data = store.to_hif()

        # Check HIF-compliant structure (no "hypergraph" wrapper)
        assert "network-type" in hif_data
        assert "metadata" in hif_data
        assert "incidences" in hif_data  # Flat at root level
        assert "nodes" in hif_data
        assert "edges" in hif_data
        assert len(hif_data["nodes"]) == 3
        assert len(hif_data["edges"]) == 1
        # Incidences should be flat (2 incidences for the one edge)
        assert len(hif_data["incidences"]) == 2

        # Check HIF field naming
        assert all("node" in n for n in hif_data["nodes"])  # Not "id"
        assert all("edge" in e for e in hif_data["edges"])  # Not "id"
        assert all("node" in i and "edge" in i for i in hif_data["incidences"])

        # Round-trip
        restored = HypergraphStore.from_hif(hif_data)
        assert restored.stats() == store.stats()

        # Check edge details preserved
        edge = restored.get_edge("fk_test")
        assert edge.type == "foreign_key"
        assert edge.source == "schema"
        assert edge.confidence == 0.9
        # Check incidences preserved (order may differ due to dict iteration)
        directions = {inc.node_id: inc.direction for inc in edge.incidences}
        assert directions["customers"] == "head"
        assert directions["orders"] == "tail"

    def test_hif_json_serializable(self, store: HypergraphStore):
        """HIF output is JSON-serializable."""
        hif_data = store.to_hif()
        json_str = json.dumps(hif_data)
        parsed = json.loads(json_str)
        assert parsed == hif_data

    def test_dict_json_serializable(self, store: HypergraphStore):
        """Dict output is JSON-serializable."""
        data = store.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed == data


class TestMetagraphPrep:
    """Tests for metagraph foundation: edges referencing edges."""

    def test_incidence_requires_node_or_edge_ref(self):
        """Incidence() with no node_id or edge_ref_id raises ValueError."""
        with pytest.raises(ValueError, match="must have either node_id or edge_ref_id"):
            Incidence()

    def test_incidence_rejects_both_node_and_edge_ref(self):
        """Incidence with both node_id and edge_ref_id raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both"):
            Incidence(node_id="a", edge_ref_id="e1")

    def test_incidence_with_edge_ref(self):
        """Incidence with edge_ref_id works."""
        inc = Incidence(edge_ref_id="e1")
        assert inc.edge_ref_id == "e1"
        assert inc.node_id is None
        assert inc.direction is None

    def test_incidence_with_edge_ref_and_direction(self):
        """Incidence with edge_ref_id and direction works."""
        inc = Incidence(edge_ref_id="e1", direction="head")
        assert inc.edge_ref_id == "e1"
        assert inc.direction == "head"

    def test_incidence_edge_ref_type_validation(self):
        """Incidence edge_ref_id must be a string."""
        with pytest.raises(TypeError, match="edge_ref_id must be a string"):
            Incidence(edge_ref_id=42)

    def test_edge_refs_property(self):
        """Hyperedge with mixed members returns correct edge_refs list."""
        edge = Hyperedge(
            id="meta1",
            type="derivation",
            incidences=[
                Incidence(node_id="A"),
                Incidence(edge_ref_id="e1"),
                Incidence(edge_ref_id="e2"),
            ],
        )
        assert edge.edge_refs == ["e1", "e2"]

    def test_node_properties_exclude_edge_refs(self):
        """.nodes and .node_set only return node members."""
        edge = Hyperedge(
            id="meta2",
            type="derivation",
            incidences=[
                Incidence(node_id="A"),
                Incidence(node_id="B"),
                Incidence(edge_ref_id="e1"),
            ],
        )
        assert edge.nodes == ["A", "B"]
        assert edge.node_set == {"A", "B"}

    def test_head_tail_exclude_edge_refs(self):
        """head_nodes and tail_nodes only return node members."""
        edge = Hyperedge(
            id="meta3",
            type="derivation",
            incidences=[
                Incidence(node_id="A", direction="tail"),
                Incidence(edge_ref_id="e1", direction="head"),
                Incidence(node_id="B", direction="head"),
            ],
        )
        assert edge.tail_nodes == ["A"]
        assert edge.head_nodes == ["B"]

    def test_edge_to_edges_index(self):
        """Adding edge with edge_ref_id populates _edge_to_edges."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        # e1: normal edge
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        # e2: references e1
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        assert "e2" in store._edge_to_edges["e1"]

    def test_delete_edge_cleans_edge_to_edges_index(self):
        """Deleting edge cleans the _edge_to_edges index."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        assert "e2" in store._edge_to_edges["e1"]
        store.delete_edge("e2")
        assert "e1" not in store._edge_to_edges  # cleaned up entirely

    def test_validate_catches_missing_edge_ref(self):
        """validate() reports edges referencing non-existent edges."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="nonexistent")],
            )
        )
        result = store.validate()
        assert result["valid"] is False
        assert any("non-existent edges" in err for err in result["errors"])
        assert "e1" in result["orphaned_edges"]

    def test_dict_roundtrip_with_edge_refs(self):
        """to_dict()/from_dict() preserves edge refs."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        data = store.to_dict()
        restored = HypergraphStore.from_dict(data)

        e2 = restored.get_edge("e2")
        assert e2 is not None
        assert e2.incidences[0].node_id == "A"
        assert e2.incidences[0].edge_ref_id is None
        assert e2.incidences[1].node_id is None
        assert e2.incidences[1].edge_ref_id == "e1"
        assert e2.edge_refs == ["e1"]

    def test_delete_referenced_edge_creates_orphan(self):
        """Deleting an edge that is referenced by another leaves a dangling ref."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        store.delete_edge("e1")
        # e2 now has a dangling reference — validate catches it
        result = store.validate()
        assert result["valid"] is False
        assert "e2" in result["orphaned_edges"]
        assert any("non-existent edges" in err for err in result["errors"])

    def test_hif_export_skips_edge_refs(self):
        """to_hif() excludes edge-ref incidences."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        hif = store.to_hif()
        # e1 has 2 node-based incidences, e2 has only 1 (the node_id one)
        assert len(hif["incidences"]) == 3  # 2 from e1 + 1 from e2
        # All HIF incidences should have "node" field
        for inc in hif["incidences"]:
            assert "node" in inc

    def test_add_edge_overwrite_updates_edge_ref_index(self):
        """Overwriting edge e2 to reference e3 instead of e1 updates index."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e3", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        # e2 references e1
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        assert "e2" in store._edge_to_edges["e1"]
        # Overwrite e2 to reference e3 instead
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e3")],
            )
        )
        assert "e1" not in store._edge_to_edges
        assert "e2" in store._edge_to_edges["e3"]

    def test_upsert_edge_updates_edge_ref_index(self):
        """Upserting edge to change edge_ref from e1 to e3 updates index."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e3", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        assert "e2" in store._edge_to_edges["e1"]
        # Upsert e2 to reference e3
        store.upsert_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e3")],
            )
        )
        assert "e1" not in store._edge_to_edges
        assert "e2" in store._edge_to_edges["e3"]

    def test_deepcopy_with_edge_refs(self):
        """Deep copy preserves edge refs and mutating original doesn't affect copy."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        clone = copy.deepcopy(store)
        # Mutate original
        store.delete_edge("e2")
        # Clone should still have e2
        assert clone.get_edge("e2") is not None
        assert clone.get_edge("e2").edge_refs == ["e1"]
        assert "e2" in clone._edge_to_edges["e1"]

    def test_edge_ref_only_edge_vertex_set(self):
        """Edge with only edge-ref incidences has empty nodes/node_set
        and no vertex-set index entry."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e3", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        # Edge with only edge-ref incidences
        store.add_edge(
            Hyperedge(
                id="e_meta", type="meta",
                incidences=[Incidence(edge_ref_id="e1"), Incidence(edge_ref_id="e3")],
            )
        )
        e_meta = store.get_edge("e_meta")
        assert e_meta.nodes == []
        assert e_meta.node_set == set()
        # Empty frozenset should NOT be in the vertex-set index
        assert frozenset() not in store._edges_by_node_set

    def test_delete_referenced_edge_cleans_index_key(self):
        """Deleting e1 that is referenced by e2 removes the 'e1' key from _edge_to_edges."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        assert "e1" in store._edge_to_edges
        store.delete_edge("e1")
        # The key itself should be removed (not just empty)
        assert "e1" not in store._edge_to_edges

    def test_hif_export_warning_metadata(self):
        """to_hif() sets _hypabase_edge_refs_omitted metadata when edge refs are skipped."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        store.add_edge(
            Hyperedge(
                id="e2", type="meta",
                incidences=[Incidence(node_id="A"), Incidence(edge_ref_id="e1")],
            )
        )
        hif = store.to_hif()
        assert hif["metadata"]["_hypabase_edge_refs_omitted"] == 1

    def test_hif_export_no_warning_without_edge_refs(self):
        """to_hif() does not set _hypabase_edge_refs_omitted when no edge refs present."""
        store = HypergraphStore()
        store.add_node(Node("A", "t"))
        store.add_node(Node("B", "t"))
        store.add_edge(
            Hyperedge(
                id="e1", type="link",
                incidences=[Incidence(node_id="A"), Incidence(node_id="B")],
            )
        )
        hif = store.to_hif()
        assert "_hypabase_edge_refs_omitted" not in hif["metadata"]
