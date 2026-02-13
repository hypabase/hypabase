"""Tests for HypergraphDB namespacing."""

import tempfile

import pytest

from hypabase.engine import (
    Hyperedge,
    HypergraphDB,
    Incidence,
    Node,
)


class TestDefaultNamespace:
    """Tests for default namespace behavior."""

    def test_default_namespace_created(self):
        """Default namespace is created on init."""
        db = HypergraphDB()
        assert db.current_namespace == "default"
        assert db.store is not None

    def test_custom_default_namespace(self):
        """Custom default namespace name."""
        db = HypergraphDB(default_namespace="main")
        assert db.current_namespace == "main"
        assert "main" in db.list_namespaces()


class TestSelectNamespace:
    """Tests for select method."""

    def test_select_creates_namespace(self):
        """Selecting non-existent namespace creates it."""
        db = HypergraphDB()
        db.select("new_namespace")
        assert db.current_namespace == "new_namespace"
        assert "new_namespace" in db.list_namespaces()

    def test_select_existing_namespace(self):
        """Selecting existing namespace switches to it."""
        db = HypergraphDB()
        db.select("ns1")
        db.store.add_node(Node("A", "test"))
        db.select("default")
        db.select("ns1")
        assert db.store.get_node("A") is not None

    def test_select_returns_self(self):
        """Select returns self for chaining."""
        db = HypergraphDB()
        result = db.select("ns1")
        assert result is db

    def test_select_chaining(self):
        """Method chaining with select."""
        db = HypergraphDB()
        db.select("ns1").store.add_node(Node("A", "test"))
        assert db.store.get_node("A") is not None


class TestNamespaceIsolation:
    """Tests for namespace isolation."""

    def test_namespaces_are_isolated(self):
        """Data in one namespace not visible in another."""
        db = HypergraphDB()

        db.select("ns1")
        db.store.add_node(Node("A", "test"))

        db.select("ns2")
        assert db.store.get_node("A") is None

    def test_same_id_different_namespaces(self):
        """Same ID can exist in multiple namespaces."""
        db = HypergraphDB()

        db.select("ns1")
        db.store.add_node(Node("shared_id", "type1"))

        db.select("ns2")
        db.store.add_node(Node("shared_id", "type2"))

        db.select("ns1")
        assert db.store.get_node("shared_id").type == "type1"

        db.select("ns2")
        assert db.store.get_node("shared_id").type == "type2"


class TestNamespaceCRUD:
    """Tests for namespace CRUD operations."""

    def test_list_namespaces(self):
        """List all namespaces."""
        db = HypergraphDB()
        db.select("ns1")
        db.select("ns2")
        namespaces = db.list_namespaces()
        assert set(namespaces) == {"default", "ns1", "ns2"}

    def test_list_namespaces_sorted(self):
        """Namespace list is sorted."""
        db = HypergraphDB()
        db.select("zebra")
        db.select("alpha")
        namespaces = db.list_namespaces()
        assert namespaces == sorted(namespaces)

    def test_namespace_exists(self):
        """Check if namespace exists."""
        db = HypergraphDB()
        assert db.namespace_exists("default") is True
        assert db.namespace_exists("nonexistent") is False
        db.select("new")
        assert db.namespace_exists("new") is True

    def test_delete_namespace(self):
        """Delete a namespace."""
        db = HypergraphDB()
        db.select("to_delete")
        db.store.add_node(Node("A", "test"))
        db.select("default")

        assert db.delete_namespace("to_delete") is True
        assert "to_delete" not in db.list_namespaces()

    def test_delete_nonexistent_namespace(self):
        """Delete non-existent namespace returns False."""
        db = HypergraphDB()
        assert db.delete_namespace("nonexistent") is False

    def test_delete_current_namespace_raises(self):
        """Cannot delete current namespace."""
        db = HypergraphDB()
        with pytest.raises(ValueError, match="Cannot delete current"):
            db.delete_namespace("default")

    def test_rename_namespace(self):
        """Rename a namespace."""
        db = HypergraphDB()
        db.select("old_name")
        db.store.add_node(Node("A", "test"))
        db.select("default")

        assert db.rename_namespace("old_name", "new_name") is True
        assert "old_name" not in db.list_namespaces()
        assert "new_name" in db.list_namespaces()

        # Data preserved
        db.select("new_name")
        assert db.store.get_node("A") is not None

    def test_rename_current_namespace(self):
        """Renaming current namespace updates current_namespace."""
        db = HypergraphDB()
        db.select("old_name")
        db.rename_namespace("old_name", "new_name")
        assert db.current_namespace == "new_name"

    def test_rename_nonexistent_namespace(self):
        """Rename non-existent namespace returns False."""
        db = HypergraphDB()
        assert db.rename_namespace("nonexistent", "new") is False

    def test_rename_to_existing_namespace_raises(self):
        """Cannot rename to existing namespace name."""
        db = HypergraphDB()
        db.select("ns1")
        db.select("ns2")
        with pytest.raises(ValueError, match="already exists"):
            db.rename_namespace("ns1", "ns2")

    def test_copy_namespace(self):
        """Copy a namespace."""
        db = HypergraphDB()
        db.select("source")
        db.store.add_node(Node("A", "test", {"key": "value"}))
        db.store.add_edge(Hyperedge("e1", "fk", [Incidence("A")]))

        db.copy_namespace("source", "target")

        # Target has copy of data
        db.select("target")
        assert db.store.get_node("A") is not None
        assert db.store.get_node("A").properties["key"] == "value"
        assert db.store.get_edge("e1") is not None

        # Source unchanged
        db.select("source")
        assert db.store.get_node("A") is not None

    def test_copy_creates_independent_copy(self):
        """Copied namespace is independent of source."""
        db = HypergraphDB()
        db.select("source")
        db.store.add_node(Node("A", "test"))

        db.copy_namespace("source", "target")

        # Modify source
        db.select("source")
        db.store.delete_node("A")
        db.store.add_node(Node("B", "test"))

        # Target unaffected
        db.select("target")
        assert db.store.get_node("A") is not None
        assert db.store.get_node("B") is None

    def test_copy_creates_deep_copy(self):
        """Copied namespace has independent property objects (deep copy)."""
        db = HypergraphDB()
        db.select("source")
        # Add node with mutable property
        db.store.add_node(Node("A", "test", {"mutable_list": [1, 2, 3]}))

        db.copy_namespace("source", "target")

        # Mutate source's property object
        db.select("source")
        source_node = db.store.get_node("A")
        source_node.properties["mutable_list"].append(4)

        # Target should NOT be affected by mutation (deep copy)
        db.select("target")
        target_node = db.store.get_node("A")
        assert target_node.properties["mutable_list"] == [1, 2, 3]

    def test_copy_nonexistent_namespace_raises(self):
        """Copy non-existent namespace raises."""
        db = HypergraphDB()
        with pytest.raises(ValueError, match="not found"):
            db.copy_namespace("nonexistent", "target")

    def test_copy_to_existing_namespace_raises(self):
        """Copy to existing namespace raises."""
        db = HypergraphDB()
        db.select("source")
        db.select("existing")
        with pytest.raises(ValueError, match="already exists"):
            db.copy_namespace("source", "existing")

    def test_copy_returns_self(self):
        """Copy returns self for chaining."""
        db = HypergraphDB()
        db.select("source")
        result = db.copy_namespace("source", "target")
        assert result is db


class TestHierarchicalNamespaces:
    """Tests for hierarchical namespace naming."""

    def test_hierarchical_names(self):
        """Hierarchical namespace names with /."""
        db = HypergraphDB()
        db.select("financial/entities")
        db.store.add_node(Node("customers", "table"))

        db.select("financial/themes")
        db.store.add_node(Node("revenue", "concept"))

        assert "financial/entities" in db.list_namespaces()
        assert "financial/themes" in db.list_namespaces()

    def test_hierarchical_isolation(self):
        """Hierarchical namespaces are fully isolated."""
        db = HypergraphDB()
        db.select("a/b")
        db.store.add_node(Node("X", "test"))

        db.select("a/c")
        assert db.store.get_node("X") is None

        db.select("a")
        assert db.store.get_node("X") is None

    def test_deep_hierarchy(self):
        """Deep hierarchical names."""
        db = HypergraphDB()
        db.select("a/b/c/d/e")
        db.store.add_node(Node("deep", "test"))

        db.select("a/b/c/d/e")
        assert db.store.get_node("deep") is not None


class TestGetNamespace:
    """Tests for get_namespace method."""

    def test_get_existing_namespace(self):
        """Get existing namespace without switching."""
        db = HypergraphDB()
        db.select("ns1")
        db.store.add_node(Node("A", "test"))
        db.select("default")

        store = db.get_namespace("ns1")
        assert store is not None
        assert store.get_node("A") is not None
        # Current namespace unchanged
        assert db.current_namespace == "default"

    def test_get_nonexistent_namespace(self):
        """Get non-existent namespace returns None."""
        db = HypergraphDB()
        assert db.get_namespace("nonexistent") is None


class TestStats:
    """Tests for stats method."""

    def test_empty_db_stats(self):
        """Stats on empty database."""
        db = HypergraphDB()
        stats = db.stats()
        assert stats["num_namespaces"] == 1
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_aggregation(self):
        """Stats aggregated across namespaces."""
        db = HypergraphDB()

        db.select("ns1")
        db.store.add_node(Node("A", "test"))
        db.store.add_node(Node("B", "test"))
        db.store.add_edge(Hyperedge("e1", "fk", [Incidence("A"), Incidence("B")]))

        db.select("ns2")
        db.store.add_node(Node("C", "test"))

        stats = db.stats()
        assert stats["num_namespaces"] == 3  # default, ns1, ns2
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 1
        assert stats["namespaces"]["ns1"]["num_nodes"] == 2
        assert stats["namespaces"]["ns1"]["num_edges"] == 1
        assert stats["namespaces"]["ns2"]["num_nodes"] == 1
        assert stats["namespaces"]["ns2"]["num_edges"] == 0

    def test_stats_current_namespace(self):
        """Stats includes current namespace."""
        db = HypergraphDB()
        db.select("active")
        stats = db.stats()
        assert stats["current_namespace"] == "active"


class TestClearNamespace:
    """Tests for clear_namespace method."""

    def test_clear_current_namespace(self):
        """Clear current namespace."""
        db = HypergraphDB()
        db.store.add_node(Node("A", "test"))
        assert db.store.get_node("A") is not None

        db.clear_namespace()
        assert db.store.get_node("A") is None
        assert db.stats()["namespaces"]["default"]["num_nodes"] == 0

    def test_clear_specific_namespace(self):
        """Clear specific namespace."""
        db = HypergraphDB()
        db.select("ns1")
        db.store.add_node(Node("A", "test"))
        db.select("default")

        db.clear_namespace("ns1")

        db.select("ns1")
        assert db.store.get_node("A") is None

    def test_clear_nonexistent_namespace(self):
        """Clear non-existent namespace returns False."""
        db = HypergraphDB()
        assert db.clear_namespace("nonexistent") is False


class TestSaveLoad:
    """Tests for save/load methods on HypergraphDB."""

    def test_save_load_roundtrip(self):
        """Save and load a database with multiple namespaces."""
        db = HypergraphDB()

        db.select("ns1")
        db.store.add_node(Node("A", "test", {"key": "value"}))

        db.select("ns2")
        db.store.add_edge(Hyperedge("e1", "fk", [Incidence("B")]))

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            loaded = HypergraphDB.load(tmpdir)

        assert set(loaded.list_namespaces()) == {"default", "ns1", "ns2"}

        loaded.select("ns1")
        assert loaded.store.get_node("A") is not None
        assert loaded.store.get_node("A").properties["key"] == "value"

        loaded.select("ns2")
        assert loaded.store.get_edge("e1") is not None

    def test_load_sets_current_namespace(self):
        """Load sets current namespace to default if available."""
        db = HypergraphDB()
        db.select("ns1")
        db.store.add_node(Node("A", "test"))

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            loaded = HypergraphDB.load(tmpdir)

        assert loaded.current_namespace == "default"

    def test_save_load_hif_format(self):
        """Save and load in HIF format."""
        db = HypergraphDB()
        db.store.add_node(Node("A", "table"))
        db.store.add_edge(
            Hyperedge(
                "e1",
                "fk",
                [Incidence("A", direction="tail")],
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir, format="hif")
            loaded = HypergraphDB.load(tmpdir)

        edge = loaded.store.get_edge("e1")
        assert edge is not None
        assert edge.incidences[0].direction == "tail"

    def test_load_db_no_default_namespace(self):
        """Load DB without default namespace sets current to first available."""
        db = HypergraphDB()
        # Create namespaces without "default"
        db.select("alpha")
        db.store.add_node(Node("A", "test"))
        db.select("beta")
        db.store.add_node(Node("B", "test"))
        # Delete default namespace
        db.delete_namespace("default")

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            loaded = HypergraphDB.load(tmpdir)

        # Should set current namespace to first sorted: "alpha"
        assert loaded.current_namespace == "alpha"
        assert "default" not in loaded.list_namespaces()
        assert loaded.store.get_node("A") is not None
