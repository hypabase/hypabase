"""SQLite persistence adapter for HypergraphCore.

This module persists HypergraphCore instances to/from SQLite. The in-memory
HypergraphCore (from core.py) is the real engine; this adapter handles
durable storage only.

The schema uses a ``namespace`` column in every table to support multiple
isolated hypergraphs in one SQLite file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from hypabase.engine.core import (
    Hyperedge,
    HypergraphCore,
    Incidence,
    Node,
)

_SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL DEFAULT 'unknown',
    properties TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (id, namespace)
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 1.0,
    properties TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (id, namespace)
);

CREATE TABLE IF NOT EXISTS incidences (
    edge_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    node_id TEXT,
    ref_edge_id TEXT,
    position INTEGER NOT NULL,
    direction TEXT,
    properties TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (edge_id, namespace, position),
    FOREIGN KEY (edge_id, namespace) REFERENCES edges(id, namespace) ON DELETE CASCADE,
    CHECK (
        (node_id IS NOT NULL AND ref_edge_id IS NULL) OR
        (node_id IS NULL AND ref_edge_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS vertex_set_index (
    vertex_set_hash TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (vertex_set_hash, edge_id, namespace),
    FOREIGN KEY (edge_id, namespace) REFERENCES edges(id, namespace) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_ns ON nodes(namespace);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_edges_ns ON edges(namespace);
CREATE INDEX IF NOT EXISTS idx_incidences_node ON incidences(node_id);
CREATE INDEX IF NOT EXISTS idx_incidences_edge ON incidences(edge_id);
CREATE INDEX IF NOT EXISTS idx_incidences_ns ON incidences(namespace);
"""


def _vertex_set_hash(node_ids: set[str]) -> str:
    key = "|".join(sorted(node_ids))
    return hashlib.sha256(key.encode()).hexdigest()


class SQLiteStorage:
    """SQLite persistence adapter for HypergraphCore.

    Supports namespace-scoped storage: each namespace's data is isolated
    by a ``namespace`` column in every table.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn
        # Check if meta table exists (i.e., schema already initialized)
        has_meta = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()[0]

        if has_meta:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"Database has meta table but no schema_version key. "
                    f"The database at '{self._path}' may be corrupted."
                )
            version = row[0]

            if version == "3":
                # Migrate v3 -> v4: rebuild incidences table to add
                # ref_edge_id column with CHECK constraint.
                # Validate existing data before migration
                invalid = conn.execute(
                    "SELECT COUNT(*) FROM incidences WHERE node_id IS NULL"
                ).fetchone()[0]
                if invalid > 0:
                    raise ValueError(
                        f"Migration v3->v4 failed: found {invalid} incidences "
                        f"with NULL node_id in v3 database"
                    )
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("""
                        CREATE TABLE incidences_v4 (
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
                        )
                    """)
                    conn.execute("""
                        INSERT INTO incidences_v4
                            (edge_id, namespace, node_id, position, direction, properties)
                        SELECT edge_id, namespace, node_id, position, direction, properties
                        FROM incidences
                    """)
                    conn.execute("DROP TABLE incidences")
                    conn.execute("ALTER TABLE incidences_v4 RENAME TO incidences")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_incidences_node"
                        " ON incidences(node_id)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_incidences_edge"
                        " ON incidences(edge_id)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_incidences_ns"
                        " ON incidences(namespace)"
                    )
                    conn.execute(
                        "UPDATE meta SET value = '4' WHERE key = 'schema_version'"
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return
            elif version == "4":
                # Already up to date
                return
            else:
                raise ValueError(
                    f"Unsupported schema version '{version}' in database "
                    f"'{self._path}'. Expected version 3 or 4. "
                    f"This database may have been created by a newer version of hypabase."
                )

        # Fresh database — create all tables with v4 schema
        conn.executescript(_SCHEMA_V4)
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '4')"
        )
        conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Namespace-scoped save/load ---

    def save(self, stores: dict[str, HypergraphCore]) -> None:
        """Persist all namespaces to SQLite (full overwrite per namespace)."""
        conn = self._conn
        # Get existing namespaces in DB
        existing_ns = {
            row[0]
            for row in conn.execute("SELECT DISTINCT namespace FROM nodes").fetchall()
        } | {
            row[0]
            for row in conn.execute("SELECT DISTINCT namespace FROM edges").fetchall()
        }
        # Delete namespaces that are no longer in stores
        for ns in existing_ns:
            if ns not in stores:
                self._delete_namespace_data(ns)
        # Save each namespace
        for ns, store in stores.items():
            self.save_namespace(ns, store)

    def load(self) -> dict[str, HypergraphCore]:
        """Load all namespaces from SQLite."""
        namespaces = self.list_namespaces()
        if not namespaces:
            return {"default": HypergraphCore()}
        return {ns: self.load_namespace(ns) for ns in namespaces}

    def save_namespace(self, namespace: str, store: HypergraphCore) -> None:
        """Persist a single namespace to SQLite (full overwrite for that namespace)."""
        conn = self._conn
        self._delete_namespace_data(namespace)

        for node in store.get_all_nodes():
            conn.execute(
                "INSERT INTO nodes (id, namespace, type, properties) VALUES (?, ?, ?, ?)",
                (node.id, namespace, node.type, json.dumps(node.properties)),
            )

        for edge in store.get_all_edges():
            conn.execute(
                "INSERT INTO edges (id, namespace, type, source, confidence, properties)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    edge.id,
                    namespace,
                    edge.type,
                    edge.source,
                    edge.confidence,
                    json.dumps(edge.properties),
                ),
            )
            for pos, inc in enumerate(edge.incidences):
                conn.execute(
                    "INSERT INTO incidences"
                    " (edge_id, namespace, node_id, ref_edge_id, position, direction, properties)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        edge.id,
                        namespace,
                        inc.node_id,
                        inc.edge_ref_id,
                        pos,
                        inc.direction,
                        json.dumps(inc.properties),
                    ),
                )
            if edge.node_set:
                vsh = _vertex_set_hash(edge.node_set)
                conn.execute(
                    "INSERT INTO vertex_set_index (vertex_set_hash, edge_id, namespace)"
                    " VALUES (?, ?, ?)",
                    (vsh, edge.id, namespace),
                )

        conn.commit()

    def load_namespace(self, namespace: str) -> HypergraphCore:
        """Load a single namespace from SQLite."""
        store = HypergraphCore()
        conn = self._conn

        for row in conn.execute(
            "SELECT id, type, properties FROM nodes WHERE namespace = ?",
            (namespace,),
        ).fetchall():
            store.add_node(Node(id=row[0], type=row[1], properties=json.loads(row[2])))

        edge_rows = conn.execute(
            "SELECT id, type, source, confidence, properties FROM edges WHERE namespace = ?",
            (namespace,),
        ).fetchall()

        for erow in edge_rows:
            edge_id, etype, source, confidence, props_json = erow
            inc_rows = conn.execute(
                "SELECT node_id, ref_edge_id, direction, properties FROM incidences"
                " WHERE edge_id = ? AND namespace = ? ORDER BY position",
                (edge_id, namespace),
            ).fetchall()
            incidences = [
                Incidence(
                    node_id=ir[0],
                    edge_ref_id=ir[1],
                    direction=ir[2],
                    properties=json.loads(ir[3]),
                )
                for ir in inc_rows
            ]
            store.add_edge(
                Hyperedge(
                    id=edge_id,
                    type=etype,
                    incidences=incidences,
                    properties=json.loads(props_json),
                    source=source,
                    confidence=confidence,
                )
            )

        return store

    def list_namespaces(self) -> list[str]:
        """List all namespaces that have data in SQLite."""
        conn = self._conn
        ns_set: set[str] = set()
        for row in conn.execute("SELECT DISTINCT namespace FROM nodes").fetchall():
            ns_set.add(row[0])
        for row in conn.execute("SELECT DISTINCT namespace FROM edges").fetchall():
            ns_set.add(row[0])
        return sorted(ns_set)

    def delete_namespace(self, namespace: str) -> None:
        """Delete all data for a namespace."""
        self._delete_namespace_data(namespace)
        self._conn.commit()

    def _delete_namespace_data(self, namespace: str) -> None:
        """Delete all rows for a namespace (no commit)."""
        conn = self._conn
        conn.execute("DELETE FROM incidences WHERE namespace = ?", (namespace,))
        conn.execute("DELETE FROM vertex_set_index WHERE namespace = ?", (namespace,))
        conn.execute("DELETE FROM edges WHERE namespace = ?", (namespace,))
        conn.execute("DELETE FROM nodes WHERE namespace = ?", (namespace,))
