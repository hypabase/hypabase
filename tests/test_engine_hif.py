"""Tests for HIF (Hypergraph Interchange Format) standard compliance.

Reference: https://github.com/HIF-org/HIF-standard

These tests ensure our implementation can:
1. Export HIF-compliant JSON that other tools can read
2. Import HIF JSON from other tools (XGI, HyperNetX, etc.)
"""

import json

import pytest

from hypabase.engine import Hyperedge, HypergraphStore, Incidence, Node


class TestHIFStructure:
    """Tests for HIF structural compliance."""

    @pytest.fixture
    def store(self) -> HypergraphStore:
        """Create a test store with mixed directed/undirected edges."""
        s = HypergraphStore()
        s.add_node(Node("A", "table"))
        s.add_node(Node("B", "table"))
        s.add_node(Node("C", "column"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="foreign_key",
                incidences=[
                    Incidence("A", direction="tail"),
                    Incidence("B", direction="head"),
                ],
                source="schema",
                confidence=0.95,
            )
        )
        s.add_edge(
            Hyperedge(
                id="e2",
                type="concept",
                incidences=[Incidence("B"), Incidence("C")],
            )
        )
        return s

    def test_hif_has_required_incidences_array(self, store):
        """HIF output must have root-level 'incidences' array."""
        hif = store.to_hif()
        assert "incidences" in hif
        assert isinstance(hif["incidences"], list)

    def test_hif_no_wrapper_object(self, store):
        """HIF should not wrap content in 'hypergraph' object."""
        hif = store.to_hif()
        assert "hypergraph" not in hif

    def test_hif_incidences_are_flat(self, store):
        """Incidences must be flat array, not nested in edges."""
        hif = store.to_hif()
        # Should have 4 incidences total (2 for e1, 2 for e2)
        assert len(hif["incidences"]) == 4
        # Each incidence must have node and edge fields
        for inc in hif["incidences"]:
            assert "node" in inc
            assert "edge" in inc

    def test_hif_node_field_naming(self, store):
        """Nodes must use 'node' field, not 'id'."""
        hif = store.to_hif()
        for node in hif.get("nodes", []):
            assert "node" in node
            assert "id" not in node

    def test_hif_edge_field_naming(self, store):
        """Edges must use 'edge' field, not 'id'."""
        hif = store.to_hif()
        for edge in hif.get("edges", []):
            assert "edge" in edge
            assert "id" not in edge

    def test_hif_network_type_present(self, store):
        """HIF should include network-type field."""
        hif = store.to_hif()
        assert "network-type" in hif
        assert hif["network-type"] in ("undirected", "directed", "asc")

    def test_hif_network_type_directed_when_has_direction(self, store):
        """Network type should be 'directed' when edges have direction."""
        hif = store.to_hif()
        # Our fixture has directed edge e1
        assert hif["network-type"] == "directed"

    def test_hif_network_type_undirected_when_no_direction(self):
        """Network type should be 'undirected' when no edges have direction."""
        s = HypergraphStore()
        s.add_node(Node("A", "table"))
        s.add_node(Node("B", "table"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="test",
                incidences=[Incidence("A"), Incidence("B")],  # No direction
            )
        )
        hif = s.to_hif()
        assert hif["network-type"] == "undirected"


class TestHIFIncidenceFields:
    """Tests for HIF incidence record compliance."""

    def test_incidence_direction_values(self):
        """Direction must be 'head' or 'tail' only."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_node(Node("B", "test"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="test",
                incidences=[
                    Incidence("A", direction="tail"),
                    Incidence("B", direction="head"),
                ],
            )
        )
        hif = s.to_hif()
        for inc in hif["incidences"]:
            if "direction" in inc:
                assert inc["direction"] in ("head", "tail")

    def test_incidence_attrs_preserved(self):
        """Incidence attrs should be preserved in export."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="test",
                incidences=[
                    Incidence("A", properties={"role": "source", "weight": 0.5}),
                ],
            )
        )
        hif = s.to_hif()
        inc = hif["incidences"][0]
        assert "attrs" in inc
        assert inc["attrs"]["role"] == "source"


class TestHIFRoundtrip:
    """Tests for HIF import/export roundtrip."""

    def test_roundtrip_preserves_nodes(self):
        """Export then import should preserve all nodes."""
        s = HypergraphStore()
        s.add_node(Node("A", "table", {"description": "test"}))
        s.add_node(Node("B", "column", {"data_type": "INTEGER"}))

        hif = s.to_hif()
        restored = HypergraphStore.from_hif(hif)

        assert restored.get_node("A") is not None
        assert restored.get_node("B") is not None
        assert restored.get_node("A").type == "table"

    def test_roundtrip_preserves_edges(self):
        """Export then import should preserve all edges."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_node(Node("B", "test"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="foreign_key",
                incidences=[Incidence("A"), Incidence("B")],
                source="schema",
                confidence=0.9,
            )
        )

        hif = s.to_hif()
        restored = HypergraphStore.from_hif(hif)

        edge = restored.get_edge("e1")
        assert edge is not None
        assert edge.type == "foreign_key"
        assert edge.source == "schema"
        assert edge.confidence == 0.9

    def test_roundtrip_preserves_incidences(self):
        """Export then import should preserve incidence details."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_node(Node("B", "test"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="test",
                incidences=[
                    Incidence("A", direction="tail"),
                    Incidence("B", direction="head"),
                ],
            )
        )

        hif = s.to_hif()
        restored = HypergraphStore.from_hif(hif)

        edge = restored.get_edge("e1")
        assert len(edge.incidences) == 2
        # Build direction map since order may vary
        directions = {inc.node_id: inc.direction for inc in edge.incidences}
        assert directions["A"] == "tail"
        assert directions["B"] == "head"

    def test_roundtrip_json_serializable(self):
        """HIF output must be JSON serializable."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_edge(
            Hyperedge(
                id="e1",
                type="test",
                incidences=[Incidence("A")],
            )
        )

        hif = s.to_hif()
        json_str = json.dumps(hif)
        parsed = json.loads(json_str)

        restored = HypergraphStore.from_hif(parsed)
        assert restored.get_node("A") is not None


class TestHIFImportFromExternal:
    """Tests for importing HIF from external sources."""

    def test_import_minimal_hif(self):
        """Should import HIF with only required 'incidences' field."""
        hif = {
            "incidences": [
                {"node": "A", "edge": "e1"},
                {"node": "B", "edge": "e1"},
            ]
        }
        store = HypergraphStore.from_hif(hif)

        # Nodes should be auto-created
        assert store.get_node("A") is not None
        assert store.get_node("B") is not None
        # Edge should be created
        assert store.get_edge("e1") is not None
        assert len(store.get_edge("e1").incidences) == 2

    def test_import_hif_with_all_fields(self):
        """Should import HIF with all optional fields."""
        hif = {
            "network-type": "directed",
            "metadata": {"source": "test"},
            "incidences": [
                {"node": "A", "edge": "e1", "direction": "tail"},
                {"node": "B", "edge": "e1", "direction": "head"},
            ],
            "nodes": [
                {"node": "A", "attrs": {"_type": "table"}},
                {"node": "B", "attrs": {"_type": "column"}},
            ],
            "edges": [
                {"edge": "e1", "attrs": {"_type": "foreign_key", "_source": "schema"}},
            ],
        }
        store = HypergraphStore.from_hif(hif)

        assert store.get_node("A").type == "table"
        assert store.get_node("B").type == "column"
        assert store.get_edge("e1").type == "foreign_key"
        assert store.get_edge("e1").source == "schema"

    def test_import_xgi_style_hif(self):
        """Should import HIF as exported by XGI library (integer IDs)."""
        # XGI-style HIF export format uses integer IDs
        hif = {
            "network-type": "undirected",
            "incidences": [
                {"node": 0, "edge": 0},
                {"node": 1, "edge": 0},
                {"node": 2, "edge": 1},
                {"node": 3, "edge": 1},
            ],
            "nodes": [
                {"node": 0},
                {"node": 1},
                {"node": 2},
                {"node": 3},
            ],
            "edges": [
                {"edge": 0},
                {"edge": 1},
            ],
        }
        store = HypergraphStore.from_hif(hif)

        # Should handle integer IDs (converted to strings internally)
        assert store.get_node("0") is not None
        assert store.get_node("1") is not None
        assert len(store.get_all_edges()) == 2
        assert store.get_edge("0") is not None
        assert store.get_edge("1") is not None


class TestHIFDirectionModes:
    """Tests for direction handling in HIF."""

    def test_directed_edge_export(self):
        """Directed edges should export with head/tail directions."""
        s = HypergraphStore()
        s.add_node(Node("source", "test"))
        s.add_node(Node("target", "test"))
        s.add_edge(
            Hyperedge(
                id="directed_edge",
                type="test",
                incidences=[
                    Incidence("source", direction="tail"),
                    Incidence("target", direction="head"),
                ],
            )
        )

        hif = s.to_hif()

        # Find incidences for this edge
        edge_incs = [i for i in hif["incidences"] if i["edge"] == "directed_edge"]
        directions = {i["node"]: i.get("direction") for i in edge_incs}

        assert directions["source"] == "tail"
        assert directions["target"] == "head"

    def test_undirected_edge_export(self):
        """Undirected edges should export without direction field."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_node(Node("B", "test"))
        s.add_edge(
            Hyperedge(
                id="undirected_edge",
                type="test",
                incidences=[
                    Incidence("A"),  # No direction
                    Incidence("B"),  # No direction
                ],
            )
        )

        hif = s.to_hif()

        edge_incs = [i for i in hif["incidences"] if i["edge"] == "undirected_edge"]
        for inc in edge_incs:
            assert "direction" not in inc or inc["direction"] is None

    def test_mixed_edges_preserve_direction(self):
        """Graph with both directed and undirected edges preserves both."""
        s = HypergraphStore()
        s.add_node(Node("A", "test"))
        s.add_node(Node("B", "test"))
        s.add_node(Node("C", "test"))

        # Directed edge
        s.add_edge(
            Hyperedge(
                id="directed",
                type="test",
                incidences=[
                    Incidence("A", direction="tail"),
                    Incidence("B", direction="head"),
                ],
            )
        )
        # Undirected edge
        s.add_edge(
            Hyperedge(
                id="undirected",
                type="test",
                incidences=[Incidence("B"), Incidence("C")],
            )
        )

        hif = s.to_hif()
        restored = HypergraphStore.from_hif(hif)

        # Directed edge should preserve direction
        directed = restored.get_edge("directed")
        assert directed.is_directed
        dir_map = {inc.node_id: inc.direction for inc in directed.incidences}
        assert dir_map["A"] == "tail"
        assert dir_map["B"] == "head"

        # Undirected edge should have no directions
        undirected = restored.get_edge("undirected")
        assert not undirected.is_directed


class TestHIFEdgeCases:
    """Edge cases for HIF handling."""

    def test_empty_hypergraph_export(self):
        """Empty hypergraph should produce valid HIF."""
        s = HypergraphStore()
        hif = s.to_hif()

        assert "incidences" in hif
        assert "nodes" in hif
        assert "edges" in hif
        assert hif["incidences"] == []
        assert hif["nodes"] == []
        assert hif["edges"] == []

    def test_nodes_only_no_edges(self):
        """Hypergraph with nodes but no edges."""
        s = HypergraphStore()
        s.add_node(Node("orphan1", "test"))
        s.add_node(Node("orphan2", "test"))

        hif = s.to_hif()
        assert len(hif["nodes"]) == 2
        assert len(hif["edges"]) == 0
        assert len(hif["incidences"]) == 0

        # Roundtrip preserves orphan nodes
        restored = HypergraphStore.from_hif(hif)
        assert restored.get_node("orphan1") is not None
        assert restored.get_node("orphan2") is not None

    def test_incidences_without_nodes_array(self):
        """Should auto-create nodes from incidences if nodes array missing."""
        hif = {
            "incidences": [
                {"node": "auto1", "edge": "e1"},
                {"node": "auto2", "edge": "e1"},
            ],
            "edges": [
                {"edge": "e1", "attrs": {"_type": "test"}},
            ],
            # No "nodes" array
        }
        store = HypergraphStore.from_hif(hif)

        assert store.get_node("auto1") is not None
        assert store.get_node("auto2") is not None
        # Auto-created nodes have unknown type
        assert store.get_node("auto1").type == "unknown"

    def test_incidences_without_edges_array(self):
        """Should auto-create edges from incidences if edges array missing."""
        hif = {
            "incidences": [
                {"node": "A", "edge": "auto_edge"},
                {"node": "B", "edge": "auto_edge"},
            ],
            "nodes": [
                {"node": "A", "attrs": {"_type": "test"}},
                {"node": "B", "attrs": {"_type": "test"}},
            ],
            # No "edges" array
        }
        store = HypergraphStore.from_hif(hif)

        edge = store.get_edge("auto_edge")
        assert edge is not None
        assert len(edge.incidences) == 2
        # Auto-created edge has unknown type
        assert edge.type == "unknown"

    def test_special_characters_in_ids(self):
        """HIF should handle special characters in IDs."""
        s = HypergraphStore()
        s.add_node(Node("table.column", "test"))
        s.add_node(Node("schema::table", "test"))
        s.add_edge(
            Hyperedge(
                id="edge/with/slashes",
                type="test",
                incidences=[
                    Incidence("table.column"),
                    Incidence("schema::table"),
                ],
            )
        )

        hif = s.to_hif()
        json_str = json.dumps(hif)
        parsed = json.loads(json_str)
        restored = HypergraphStore.from_hif(parsed)

        assert restored.get_node("table.column") is not None
        assert restored.get_node("schema::table") is not None
        assert restored.get_edge("edge/with/slashes") is not None


class TestHIFValidation:
    """Tests for HIF import validation."""

    def test_from_hif_invalid_confidence(self):
        """Invalid confidence value raises ValueError."""
        hif = {
            "incidences": [
                {"node": "A", "edge": "e1"},
            ],
            "nodes": [
                {"node": "A", "attrs": {"_type": "test"}},
            ],
            "edges": [
                {"edge": "e1", "attrs": {"_type": "test", "_confidence": 1.5}},
            ],
        }
        with pytest.raises(ValueError, match="confidence must be between"):
            HypergraphStore.from_hif(hif)

    def test_from_hif_invalid_direction(self):
        """Invalid direction value raises ValueError."""
        hif = {
            "incidences": [
                {"node": "A", "edge": "e1", "direction": "invalid"},
            ],
            "nodes": [
                {"node": "A", "attrs": {"_type": "test"}},
            ],
            "edges": [
                {"edge": "e1", "attrs": {"_type": "test"}},
            ],
        }
        with pytest.raises(ValueError, match="direction must be"):
            HypergraphStore.from_hif(hif)
