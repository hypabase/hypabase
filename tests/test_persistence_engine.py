"""Tests for incremental persistence engine methods."""

import sqlite3
import time

import pytest

from hypabase import Hypabase
from hypabase.engine.core import Hyperedge, Incidence, Node
from hypabase.engine.storage import SQLiteStorage


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def storage(tmp_db_path):
    s = SQLiteStorage(tmp_db_path)
    yield s
    s.close()


class TestWriteNode:
    def test_write_node_persists_to_sqlite(self, storage, tmp_db_path):
        node = Node(id="alice", type="person", properties={"age": 30}, created_at=1.0, updated_at=2.0)
        storage.write_node("default", node)

        # Verify via raw SQL
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id, namespace, type, properties, created_at, updated_at FROM nodes WHERE id = ?",
            ("alice",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "alice"
        assert row[1] == "default"
        assert row[2] == "person"
        assert '"age": 30' in row[3]
        assert row[4] == 1.0
        assert row[5] == 2.0

    def test_write_node_upserts_existing(self, storage, tmp_db_path):
        node1 = Node(id="alice", type="person", properties={"age": 30})
        storage.write_node("default", node1)

        node2 = Node(id="alice", type="user", properties={"age": 31})
        storage.write_node("default", node2)

        conn = sqlite3.connect(tmp_db_path)
        rows = conn.execute("SELECT type, properties FROM nodes WHERE id = 'alice'").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "user"
        assert '"age": 31' in rows[0][1]

    def test_write_node_auto_commits(self, storage, tmp_db_path):
        node = Node(id="bob", type="person")
        storage.write_node("default", node)

        # Second connection should see the row (auto-committed)
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'bob'").fetchone()
        conn.close()
        assert row is not None


class TestWriteEdge:
    def test_write_edge_persists_all_tables(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
            source="test",
            confidence=0.9,
        )
        storage.write_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        # Check edges table
        erow = conn.execute("SELECT id, type, source, confidence FROM edges WHERE id = 'e1'").fetchone()
        assert erow is not None
        assert erow[1] == "knows"
        assert erow[2] == "test"
        assert erow[3] == 0.9

        # Check incidences table
        inc_rows = conn.execute(
            "SELECT node_id, position FROM incidences WHERE edge_id = 'e1' ORDER BY position"
        ).fetchall()
        assert len(inc_rows) == 2
        assert inc_rows[0][0] == "alice"
        assert inc_rows[1][0] == "bob"

        # Check vertex_set_index
        vsi_row = conn.execute(
            "SELECT edge_id FROM vertex_set_index WHERE edge_id = 'e1'"
        ).fetchone()
        assert vsi_row is not None
        conn.close()

    def test_write_edge_upserts_existing(self, storage, tmp_db_path):
        edge1 = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
        )
        storage.write_edge("default", edge1)

        # Write again with different incidences
        edge2 = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="carol")],
        )
        storage.write_edge("default", edge2)

        conn = sqlite3.connect(tmp_db_path)
        inc_rows = conn.execute(
            "SELECT node_id FROM incidences WHERE edge_id = 'e1' ORDER BY position"
        ).fetchall()
        conn.close()
        # Should be replaced, not duplicated
        assert len(inc_rows) == 2
        assert inc_rows[0][0] == "alice"
        assert inc_rows[1][0] == "carol"

    def test_write_edge_with_directed_incidences(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="link",
            incidences=[
                Incidence(node_id="a", direction="tail"),
                Incidence(node_id="b", direction="head"),
            ],
        )
        storage.write_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        rows = conn.execute(
            "SELECT node_id, direction FROM incidences WHERE edge_id = 'e1' ORDER BY position"
        ).fetchall()
        conn.close()
        assert rows[0] == ("a", "tail")
        assert rows[1] == ("b", "head")

    def test_write_edge_with_edge_ref(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="meta1",
            type="about",
            incidences=[
                Incidence(node_id="comment1"),
                Incidence(edge_ref_id="e1"),
            ],
        )
        storage.write_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        rows = conn.execute(
            "SELECT node_id, ref_edge_id FROM incidences WHERE edge_id = 'meta1' ORDER BY position"
        ).fetchall()
        conn.close()
        assert rows[0] == ("comment1", None)
        assert rows[1] == (None, "e1")


class TestRemoveNode:
    def test_remove_node_deletes_row(self, storage, tmp_db_path):
        node = Node(id="alice", type="person")
        storage.write_node("default", node)
        storage.remove_node("default", "alice")

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'alice'").fetchone()
        conn.close()
        assert row is None

    def test_remove_nonexistent_node_no_error(self, storage):
        # Should not raise
        storage.remove_node("default", "nonexistent")


class TestRemoveEdge:
    def test_remove_edge_deletes_all_related_rows(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
        )
        storage.write_edge("default", edge)
        storage.remove_edge("default", "e1")

        conn = sqlite3.connect(tmp_db_path)
        assert conn.execute("SELECT id FROM edges WHERE id = 'e1'").fetchone() is None
        assert conn.execute("SELECT edge_id FROM incidences WHERE edge_id = 'e1'").fetchone() is None
        assert conn.execute("SELECT edge_id FROM vertex_set_index WHERE edge_id = 'e1'").fetchone() is None
        conn.close()

    def test_remove_nonexistent_edge_no_error(self, storage):
        storage.remove_edge("default", "nonexistent")


class TestUpdateEdge:
    def test_update_edge_changes_expired_at(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
        )
        storage.write_edge("default", edge)

        edge.expired_at = 12345.0
        storage.update_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT expired_at FROM edges WHERE id = 'e1'").fetchone()
        conn.close()
        assert row[0] == 12345.0

    def test_update_edge_changes_confidence(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
            confidence=0.5,
        )
        storage.write_edge("default", edge)

        edge.confidence = 0.99
        storage.update_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT confidence FROM edges WHERE id = 'e1'").fetchone()
        conn.close()
        assert row[0] == 0.99

    def test_update_edge_preserves_incidences(self, storage, tmp_db_path):
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="alice"), Incidence(node_id="bob")],
        )
        storage.write_edge("default", edge)

        edge.confidence = 0.5
        storage.update_edge("default", edge)

        conn = sqlite3.connect(tmp_db_path)
        rows = conn.execute(
            "SELECT node_id FROM incidences WHERE edge_id = 'e1' ORDER BY position"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "alice"
        assert rows[1][0] == "bob"


class TestTransactions:
    def test_begin_commit_groups_writes(self, storage, tmp_db_path):
        storage.begin()
        for i in range(10):
            storage.write_node("default", Node(id=f"node_{i}", type="item"))
        storage.commit()

        conn = sqlite3.connect(tmp_db_path)
        count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        conn.close()
        assert count == 10

    def test_writes_invisible_before_commit(self, storage, tmp_db_path):
        storage.begin()
        storage.write_node("default", Node(id="secret", type="item"))

        # Second connection should NOT see the row (not committed yet)
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'secret'").fetchone()
        conn.close()
        assert row is None

        storage.commit()

        # Now it should be visible
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'secret'").fetchone()
        conn.close()
        assert row is not None

    def test_nested_begin_only_outermost_commits(self, storage, tmp_db_path):
        storage.begin()
        storage.begin()  # nested
        storage.write_node("default", Node(id="inner", type="item"))
        storage.commit()  # inner commit — should NOT flush

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'inner'").fetchone()
        conn.close()
        assert row is None  # not visible yet

        storage.commit()  # outer commit — should flush

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'inner'").fetchone()
        conn.close()
        assert row is not None

    def test_rollback_discards_uncommitted(self, storage, tmp_db_path):
        storage.begin()
        storage.write_node("default", Node(id="doomed", type="item"))
        storage.rollback()

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'doomed'").fetchone()
        conn.close()
        assert row is None

    def test_auto_commit_outside_transaction(self, storage, tmp_db_path):
        storage.write_node("default", Node(id="instant", type="item"))

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT id FROM nodes WHERE id = 'instant'").fetchone()
        conn.close()
        assert row is not None

    def test_commit_without_begin_raises(self, storage):
        with pytest.raises(RuntimeError, match="commit.*without matching begin"):
            storage.commit()

    def test_rollback_outside_transaction_is_noop(self, storage):
        # Should not raise
        storage.rollback()
        # Auto-commit still works after no-op rollback
        storage.write_node("default", Node(id="still_works", type="item"))
        assert storage._tx_depth == 0

    def test_writes_work_after_rollback(self, storage, tmp_db_path):
        storage.begin()
        storage.write_node("default", Node(id="doomed", type="item"))
        storage.rollback()

        # Subsequent write should auto-commit normally
        storage.write_node("default", Node(id="survivor", type="item"))

        conn = sqlite3.connect(tmp_db_path)
        assert conn.execute("SELECT id FROM nodes WHERE id = 'doomed'").fetchone() is None
        assert conn.execute("SELECT id FROM nodes WHERE id = 'survivor'").fetchone() is not None
        conn.close()


class TestAccessLogTransactions:
    def test_record_access_respects_transaction(self, storage, tmp_db_path):
        storage.begin()
        storage.record_access("default", "edge", "e1")

        # Second connection should NOT see the row (not committed yet)
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT ref_id FROM access_log WHERE ref_id = 'e1'"
        ).fetchone()
        conn.close()
        assert row is None

        storage.commit()

        # Now it should be visible
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT ref_id FROM access_log WHERE ref_id = 'e1'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_record_access_batch_respects_transaction(self, storage, tmp_db_path):
        storage.begin()
        storage.record_access_batch("default", "edge", ["e1", "e2"])

        # Second connection should NOT see the rows
        conn = sqlite3.connect(tmp_db_path)
        count = conn.execute("SELECT count(*) FROM access_log").fetchone()[0]
        conn.close()
        assert count == 0

        storage.commit()

        conn = sqlite3.connect(tmp_db_path)
        count = conn.execute("SELECT count(*) FROM access_log").fetchone()[0]
        conn.close()
        assert count == 2

    def test_record_access_auto_commits_outside_transaction(self, storage, tmp_db_path):
        storage.record_access("default", "edge", "e1")

        # Should be immediately visible from second connection
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT ref_id FROM access_log WHERE ref_id = 'e1'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_save_embedding_respects_transaction(self, storage, tmp_db_path):
        from hypabase.engine.vector import pack_embedding

        blob = pack_embedding([1.0, 0.0, 0.0, 0.0])
        storage.begin()
        storage.save_embedding(
            id="emb1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob, dimension=4, model="mock",
        )

        # Second connection should NOT see the embedding
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is None

        storage.commit()

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_save_embedding_auto_commits_outside_transaction(self, storage, tmp_db_path):
        from hypabase.engine.vector import pack_embedding

        blob = pack_embedding([1.0, 0.0, 0.0, 0.0])
        storage.save_embedding(
            id="emb1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob, dimension=4, model="mock",
        )

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_delete_embeddings_respects_transaction(self, storage, tmp_db_path):
        from hypabase.engine.vector import pack_embedding

        blob = pack_embedding([1.0, 0.0, 0.0, 0.0])
        storage.save_embedding(
            id="emb1", namespace="default", kind="node", ref_id="a",
            text="A", embedding=blob, dimension=4, model="mock",
        )

        # Verify committed and visible
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is not None

        # Delete inside a transaction — should not be visible externally yet
        storage.begin()
        storage.delete_embeddings("default", kind="node")

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is not None, "embedding should still be visible before commit"

        storage.commit()

        # After commit, the embedding should be gone
        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute(
            "SELECT id FROM embeddings WHERE id = 'emb1'"
        ).fetchone()
        conn.close()
        assert row is None


class TestRoundtrip:
    def test_write_then_load_namespace(self, storage):
        # Write nodes and edges incrementally
        storage.write_node("ns1", Node(id="a", type="person"))
        storage.write_node("ns1", Node(id="b", type="person"))
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
            source="test",
            confidence=0.8,
        )
        storage.write_edge("ns1", edge)

        # Load and verify
        core = storage.load_namespace("ns1")
        assert core.get_node("a") is not None
        assert core.get_node("b") is not None
        loaded_edge = core.get_edge("e1")
        assert loaded_edge is not None
        assert loaded_edge.type == "knows"
        assert loaded_edge.source == "test"
        assert loaded_edge.confidence == 0.8
        assert len(loaded_edge.incidences) == 2

    def test_incremental_matches_snapshot(self, tmp_path):
        db1 = str(tmp_path / "incremental.db")
        db2 = str(tmp_path / "snapshot.db")

        nodes = [
            Node(id="a", type="person", properties={"x": 1}, created_at=1.0, updated_at=2.0),
            Node(id="b", type="person", properties={"y": 2}, created_at=3.0, updated_at=4.0),
        ]
        edge = Hyperedge(
            id="e1",
            type="knows",
            incidences=[Incidence(node_id="a"), Incidence(node_id="b")],
            source="src",
            confidence=0.75,
            created_at=5.0,
        )

        # Incremental path
        s1 = SQLiteStorage(db1)
        for n in nodes:
            s1.write_node("default", n)
        s1.write_edge("default", edge)
        s1.close()

        # Snapshot path
        from hypabase.engine.core import HypergraphCore
        core = HypergraphCore()
        for n in nodes:
            core.add_node(n)
        core.add_edge(edge)
        s2 = SQLiteStorage(db2)
        s2.save_namespace("default", core)
        s2.close()

        # Compare raw SQL rows
        conn1 = sqlite3.connect(db1)
        conn2 = sqlite3.connect(db2)

        for table in ("nodes", "edges", "incidences", "vertex_set_index"):
            rows1 = sorted(conn1.execute(f"SELECT * FROM {table}").fetchall())
            rows2 = sorted(conn2.execute(f"SELECT * FROM {table}").fetchall())
            assert rows1 == rows2, f"Mismatch in {table}"

        conn1.close()
        conn2.close()


class TestIntegrationWithClient:
    def test_node_survives_close_reopen(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.node("alice", type="person", role="engineer")
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        node = hb2.get_node("alice")
        assert node is not None
        assert node.type == "person"
        assert node.properties["role"] == "engineer"
        hb2.close()

    def test_edge_survives_close_reopen(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        hb.edge(["alice", "bob", "carol"], type="group", source="test", confidence=0.9)
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        edges = hb2.edges(type="group")
        assert len(edges) == 1
        e = edges[0]
        assert set(e.node_ids) == {"alice", "bob", "carol"}
        assert e.source == "test"
        assert e.confidence == 0.9
        hb2.close()

    def test_batch_still_works(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        with hb.batch():
            for i in range(20):
                hb.edge([f"a_{i}", f"b_{i}"], type="link")
        hb.close()

        hb2 = Hypabase(tmp_db_path)
        assert hb2.stats().edge_count == 20
        hb2.close()

    def test_remember_100_no_batch_needed(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)
        start = time.time()
        for i in range(100):
            hb.edge([f"src_{i}", f"dst_{i}"], type="link", source="perf_test")
        elapsed = time.time() - start
        hb.close()

        # Should be well under 2s with incremental writes (was ~5s with snapshot)
        assert elapsed < 2.0, f"100 edges took {elapsed:.2f}s, expected < 2s"

        # Verify data survived
        hb2 = Hypabase(tmp_db_path)
        assert hb2.stats().edge_count == 100
        hb2.close()
