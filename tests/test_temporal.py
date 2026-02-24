"""Tests for temporal support in Hypabase."""

import sqlite3
import time
from datetime import datetime, timezone

from hypabase import Hypabase
from hypabase.engine.core import Hyperedge, HypergraphCore, Incidence, Node
from hypabase.engine.storage import SQLiteStorage


class TestNodeTimestamps:
    def test_node_created_at_set(self):
        hb = Hypabase()
        node = hb.node("alice", type="person")
        assert node.created_at is not None

    def test_node_updated_at_set(self):
        hb = Hypabase()
        node = hb.node("alice", type="person")
        assert node.updated_at is not None

    def test_node_updated_at_changes_on_update(self):
        hb = Hypabase()
        n1 = hb.node("alice", type="person")
        time.sleep(0.01)
        n2 = hb.node("alice", type="doctor")
        assert n2.updated_at is not None
        assert n1.updated_at is not None
        assert n2.updated_at >= n1.updated_at

    def test_node_created_at_stable_on_update(self):
        hb = Hypabase()
        n1 = hb.node("alice", type="person")
        created = n1.created_at
        time.sleep(0.01)
        n2 = hb.node("alice", type="doctor")
        assert n2.created_at == created


class TestEdgeTimestamps:
    def test_edge_created_at_set(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        assert e.created_at is not None

    def test_edge_valid_at(self):
        hb = Hypabase()
        valid = datetime(2026, 2, 18, tzinfo=timezone.utc)
        e = hb.edge(["alice", "bob"], type="meeting", valid_at=valid)
        assert e.valid_at is not None
        assert e.valid_at == valid

    def test_edge_no_valid_at_by_default(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        assert e.valid_at is None

    def test_edge_no_expired_at_by_default(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        assert e.expired_at is None

    def test_edge_is_active(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        assert e.is_active is True


class TestExpireEdge:
    def test_expire_existing_edge(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        result = hb.expire_edge(e.id)
        assert result is not None
        assert result.expired_at is not None
        assert result.is_active is False

    def test_expire_nonexistent_edge(self):
        hb = Hypabase()
        assert hb.expire_edge("nonexistent") is None

    def test_expired_edge_filtered_by_default(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        hb.expire_edge(e.id)
        assert hb.edges(active=True) == []

    def test_expired_edge_visible_with_include_expired(self):
        hb = Hypabase()
        e = hb.edge(["alice", "bob"], type="knows")
        hb.expire_edge(e.id)
        results = hb.edges(include_expired=True)
        assert len(results) == 1
        assert results[0].id == e.id

    def test_mix_active_and_expired(self):
        hb = Hypabase()
        e1 = hb.edge(["alice", "bob"], type="knows")
        e2 = hb.edge(["bob", "carol"], type="knows")
        hb.expire_edge(e1.id)
        active = hb.edges(active=True)
        assert len(active) == 1
        assert active[0].id == e2.id
        all_edges = hb.edges(include_expired=True)
        assert len(all_edges) == 2


class TestSupersedeEdge:
    def test_supersede_existing_edge(self):
        hb = Hypabase()
        old = hb.edge(["alice", "bob"], type="knows")
        result = hb.supersede_edge(
            old.id, ["alice", "bob"], type="friends",
        )
        assert result is not None
        expired, new = result
        assert expired.expired_at is not None
        assert new.type == "friends"
        assert new.expired_at is None

    def test_supersede_nonexistent(self):
        hb = Hypabase()
        assert hb.supersede_edge("nope", ["a", "b"], type="t") is None

    def test_supersede_preserves_old_edge(self):
        hb = Hypabase()
        old = hb.edge(["alice", "bob"], type="knows")
        hb.supersede_edge(old.id, ["alice", "bob"], type="friends")
        all_edges = hb.edges(include_expired=True)
        assert len(all_edges) == 2
        active = hb.edges(active=True)
        assert len(active) == 1
        assert active[0].type == "friends"


class TestTemporalQueries:
    def test_since_filter(self):
        hb = Hypabase()
        hb.edge(["a", "b"], type="link")
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.02)
        e2 = hb.edge(["c", "d"], type="link")
        results = hb.edges(since=cutoff)
        assert len(results) == 1
        assert results[0].id == e2.id

    def test_before_filter(self):
        hb = Hypabase()
        e1 = hb.edge(["a", "b"], type="link")
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.02)
        hb.edge(["c", "d"], type="link")
        results = hb.edges(before=cutoff)
        assert len(results) == 1
        assert results[0].id == e1.id

    def test_at_point_in_time(self):
        hb = Hypabase()
        past = datetime(2025, 1, 1, tzinfo=timezone.utc)

        e = hb.edge(["a", "b"], type="link", valid_at=past)
        # Expire it now (sets expired_at to current time ~2026-02-19)
        hb.expire_edge(e.id)

        # Query at a point between valid_at and expired_at (before expiration)
        query_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        results = hb.edges(at=query_time)
        assert len(results) == 1

        # Query at a point after expiration — not visible
        future_query = datetime(2027, 1, 1, tzinfo=timezone.utc)
        results = hb.edges(at=future_query)
        assert len(results) == 0

    def test_at_before_valid_at(self):
        hb = Hypabase()
        valid = datetime(2026, 6, 1, tzinfo=timezone.utc)
        hb.edge(["a", "b"], type="link", valid_at=valid)
        results = hb.edges(at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert len(results) == 0

    def test_combined_temporal_and_type_filter(self):
        hb = Hypabase()
        hb.edge(["a", "b"], type="knows")
        e2 = hb.edge(["c", "d"], type="works_with")
        hb.expire_edge(e2.id)
        results = hb.edges(type="works_with", active=True)
        assert len(results) == 0
        results = hb.edges(type="works_with", include_expired=True)
        assert len(results) == 1


class TestTemporalPersistence:
    def test_created_at_survives_save_cycle(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        e = hb.edge(["alice", "bob"], type="knows")
        original_created = e.created_at
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        edges = hb2.edges(include_expired=True)
        assert len(edges) == 1
        assert edges[0].created_at is not None
        assert abs(edges[0].created_at.timestamp() - original_created.timestamp()) < 0.01
        hb2.close()

    def test_valid_at_survives_save_cycle(self, tmp_db_path):
        valid = datetime(2026, 2, 18, tzinfo=timezone.utc)
        hb = Hypabase(tmp_db_path)
        hb.edge(["alice", "bob"], type="meeting", valid_at=valid)
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        edges = hb2.edges()
        assert len(edges) == 1
        assert edges[0].valid_at is not None
        assert abs(edges[0].valid_at.timestamp() - valid.timestamp()) < 0.01
        hb2.close()

    def test_expired_at_survives_save_cycle(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        e = hb.edge(["alice", "bob"], type="knows")
        hb.expire_edge(e.id)
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        edges = hb2.edges(include_expired=True)
        assert len(edges) == 1
        assert edges[0].expired_at is not None
        assert edges[0].is_active is False
        hb2.close()

    def test_node_timestamps_survive_save_cycle(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        n = hb.node("alice", type="person")
        original_created = n.created_at
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        alice = hb2.get_node("alice")
        assert alice is not None
        assert alice.created_at is not None
        assert abs(alice.created_at.timestamp() - original_created.timestamp()) < 0.01
        hb2.close()


class TestV4ToV5Migration:
    def _create_v4_database(self, path: str) -> None:
        """Create a v4 schema database with test data."""
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
                node_id TEXT,
                ref_edge_id TEXT,
                position INTEGER NOT NULL,
                direction TEXT,
                properties TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (edge_id, namespace, position),
                FOREIGN KEY (edge_id, namespace)
                    REFERENCES edges(id, namespace) ON DELETE CASCADE,
                CHECK (
                    (node_id IS NOT NULL AND ref_edge_id IS NULL) OR
                    (node_id IS NULL AND ref_edge_id IS NOT NULL)
                )
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
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '4')")
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

    def test_v4_to_v5_migration(self, tmp_db_path):
        """v4 database is migrated to v5, preserving all data and adding temporal columns."""
        self._create_v4_database(tmp_db_path)

        storage = SQLiteStorage(tmp_db_path)
        loaded = storage.load_namespace("default")
        storage.close()

        # Data preserved
        assert loaded.get_node("alice") is not None
        assert loaded.get_node("bob") is not None
        e1 = loaded.get_edge("e1")
        assert e1 is not None
        assert e1.type == "knows"
        assert e1.source == "manual"
        assert e1.confidence == 0.9

        # Migrated data retains NULL timestamps (no misleading backfill)
        alice = loaded.get_node("alice")
        assert alice.created_at is None
        assert alice.updated_at is None
        assert e1.created_at is None

        # Schema version is now 5
        conn = sqlite3.connect(tmp_db_path)
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        conn.close()
        assert version == "5"

    def test_v4_migration_creates_embeddings_table(self, tmp_db_path):
        self._create_v4_database(tmp_db_path)
        storage = SQLiteStorage(tmp_db_path)
        # Verify embeddings table exists
        count = storage._conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone()[0]
        assert count == 1
        storage.close()

    def test_v4_migration_creates_access_log_table(self, tmp_db_path):
        self._create_v4_database(tmp_db_path)
        storage = SQLiteStorage(tmp_db_path)
        count = storage._conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='access_log'"
        ).fetchone()[0]
        assert count == 1
        storage.close()


class TestCoreTemporalOps:
    def test_expire_edge_core(self):
        store = HypergraphCore()
        store.add_node(Node("a", "t"))
        store.add_node(Node("b", "t"))
        store.add_edge(Hyperedge(
            id="e1", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
        ))
        assert store.expire_edge("e1") is True
        e = store.get_edge("e1")
        assert e is not None
        assert e.expired_at is not None

    def test_expire_nonexistent_core(self):
        store = HypergraphCore()
        assert store.expire_edge("nope") is False

    def test_supersede_edge_core(self):
        store = HypergraphCore()
        store.add_node(Node("a", "t"))
        store.add_node(Node("b", "t"))
        store.add_edge(Hyperedge(
            id="e1", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
        ))
        result = store.supersede_edge(
            "e1",
            type="link_v2",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
        )
        assert result is not None
        old, new = result
        assert old.expired_at is not None
        assert new.type == "link_v2"
        assert new.id != "e1"

    def test_filter_temporal_active(self):
        store = HypergraphCore()
        store.add_node(Node("a", "t"))
        store.add_node(Node("b", "t"))
        e1 = Hyperedge(
            id="e1", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
        )
        e2 = Hyperedge(
            id="e2", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
            expired_at=time.time(),
        )
        active = store.filter_temporal([e1, e2], active=True)
        assert len(active) == 1
        assert active[0].id == "e1"

    def test_filter_temporal_include_expired(self):
        store = HypergraphCore()
        e1 = Hyperedge(
            id="e1", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
        )
        e2 = Hyperedge(
            id="e2", type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
            expired_at=time.time(),
        )
        result = store.filter_temporal([e1, e2], include_expired=True)
        assert len(result) == 2


class TestTemporalEdgeCases:
    def test_since_filter_preserves_null_created_at(self):
        """Edges with created_at=None survive since and before filters."""
        from hypabase.engine.core import Hyperedge, HypergraphCore, Incidence

        store = HypergraphCore()
        store.add_node(Node("a", "t"))
        store.add_node(Node("b", "t"))
        edge = Hyperedge(
            id="e_null",
            type="link",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
            created_at=None,
        )
        store.add_edge(edge)
        cutoff = time.time()
        result_since = store.filter_temporal([edge], since=cutoff)
        assert len(result_since) == 1
        assert result_since[0].id == "e_null"
        result_before = store.filter_temporal([edge], before=cutoff)
        assert len(result_before) == 1
        assert result_before[0].id == "e_null"

    def test_edge_updated_at_is_none(self):
        """New edges have updated_at=None on the model."""
        hb = Hypabase()
        e = hb.edge(["a", "b"], type="link")
        assert e.updated_at is None


class TestBackwardCompatibility:
    """Verify existing behavior is preserved."""

    def test_edges_default_hides_expired(self):
        hb = Hypabase()
        e = hb.edge(["a", "b"], type="link")
        hb.expire_edge(e.id)
        # Default behavior: active=True
        assert hb.edges() == []

    def test_edges_containing_hides_expired(self):
        hb = Hypabase()
        e = hb.edge(["a", "b"], type="link")
        hb.expire_edge(e.id)
        assert hb.edges(containing=["a"]) == []

    def test_edges_by_type_hides_expired(self):
        hb = Hypabase()
        e = hb.edge(["a", "b"], type="link")
        hb.expire_edge(e.id)
        assert hb.edges(type="link") == []
