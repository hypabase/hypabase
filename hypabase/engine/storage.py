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
import logging
import sqlite3
import time
from pathlib import Path

from hypabase.engine.core import (
    Hyperedge,
    HypergraphCore,
    Incidence,
    Node,
)
from hypabase.engine.persistence_engine import PersistenceEngine

logger = logging.getLogger(__name__)


def _safe_json_loads(raw: str, context: str) -> dict[str, object]:
    """Parse JSON with fallback to empty dict on decode errors."""
    try:
        result: dict[str, object] = json.loads(raw)
        return result
    except json.JSONDecodeError:
        logger.error("Corrupt JSON in %s, using empty dict: %s", context, raw[:200])
        return {}


_SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL DEFAULT 'unknown',
    properties TEXT NOT NULL DEFAULT '{}',
    created_at REAL,
    updated_at REAL,
    PRIMARY KEY (id, namespace)
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 1.0,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at REAL,
    valid_at REAL,
    expired_at REAL,
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

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS access_log (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE (namespace, kind, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_ns ON nodes(namespace);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_edges_ns ON edges(namespace);
CREATE INDEX IF NOT EXISTS idx_edges_expired ON edges(expired_at);
CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_at);
CREATE INDEX IF NOT EXISTS idx_edges_created ON edges(created_at);
CREATE INDEX IF NOT EXISTS idx_incidences_node ON incidences(node_id);
CREATE INDEX IF NOT EXISTS idx_incidences_edge ON incidences(edge_id);
CREATE INDEX IF NOT EXISTS idx_incidences_ns ON incidences(namespace);
CREATE INDEX IF NOT EXISTS idx_embeddings_ns_kind ON embeddings(namespace, kind);
CREATE INDEX IF NOT EXISTS idx_embeddings_kind ON embeddings(kind, ref_id);
CREATE INDEX IF NOT EXISTS idx_access_log_ns ON access_log(namespace);
CREATE INDEX IF NOT EXISTS idx_access_log_ref ON access_log(kind, ref_id);
"""


def _vertex_set_hash(node_ids: set[str]) -> str:
    key = "|".join(sorted(node_ids))
    return hashlib.sha256(key.encode()).hexdigest()


class SQLiteStorage(PersistenceEngine):
    """SQLite persistence adapter for HypergraphCore.

    Supports namespace-scoped storage: each namespace's data is isolated
    by a ``namespace`` column in every table.

    Implements the PersistenceEngine ABC with incremental write-through
    persistence. Writes auto-commit outside transactions; inside a
    begin/commit block, writes are buffered until commit.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._vec_dimension: int | None = None
        self._tx_depth: int = 0
        self._init_vec_extension()
        self._init_schema()
        self._detect_vec_dimension()

    def _init_vec_extension(self) -> None:
        """Load the sqlite-vec extension."""
        try:
            import sqlite_vec  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "sqlite-vec is required but not installed or failed to load. Install with: pip install sqlite-vec"
            ) from e
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)

    def _init_schema(self) -> None:
        conn = self._conn
        # Check if meta table exists (i.e., schema already initialized)
        has_meta = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()[0]

        if has_meta:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
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
                invalid = conn.execute("SELECT COUNT(*) FROM incidences WHERE node_id IS NULL").fetchone()[0]
                if invalid > 0:
                    raise ValueError(
                        f"Migration v3->v4 failed: found {invalid} incidences with NULL node_id in v3 database"
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
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidences_node ON incidences(node_id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidences_edge ON incidences(edge_id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidences_ns ON incidences(namespace)")
                    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                # Fall through to v4→v5 migration
                self._migrate_v4_to_v5()
                return
            elif version in ("4", "5"):
                if version == "4":
                    self._migrate_v4_to_v5()
                else:
                    # Clean up stale single-column index from early v5 migrations
                    conn.execute("DROP INDEX IF EXISTS idx_embeddings_ns")
                return
            else:
                raise ValueError(
                    f"Unsupported schema version '{version}' in database "
                    f"'{self._path}'. Expected version 3, 4, or 5. "
                    f"This database may have been created by a newer version of hypabase."
                )

        # Fresh database — create all tables with v5 schema
        conn.executescript(_SCHEMA_V5)
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '5')")
        conn.commit()

    def _migrate_v4_to_v5(self) -> None:
        """Migrate v4 schema to v5: add temporal columns, embeddings, and access_log tables."""
        conn = self._conn
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Add temporal columns to nodes
            conn.execute("ALTER TABLE nodes ADD COLUMN created_at REAL")
            conn.execute("ALTER TABLE nodes ADD COLUMN updated_at REAL")

            # Add temporal columns to edges
            conn.execute("ALTER TABLE edges ADD COLUMN created_at REAL")
            conn.execute("ALTER TABLE edges ADD COLUMN valid_at REAL")
            conn.execute("ALTER TABLE edges ADD COLUMN expired_at REAL")

            # Create temporal indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_expired ON edges(expired_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_created ON edges(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)")

            # Create embeddings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    kind TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    model TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_ns_kind ON embeddings(namespace, kind)")
            conn.execute("DROP INDEX IF EXISTS idx_embeddings_ns")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_kind ON embeddings(kind, ref_id)")

            # Create access_log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    kind TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 1,
                    UNIQUE (namespace, kind, ref_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_ns ON access_log(namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_ref ON access_log(kind, ref_id)")

            conn.execute("UPDATE meta SET value = '5' WHERE key = 'schema_version'")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    # --- Namespace-scoped save/load ---

    def save(self, stores: dict[str, HypergraphCore]) -> None:
        """Persist all namespaces to SQLite (full overwrite per namespace)."""
        conn = self._conn
        # Get existing namespaces in DB
        existing_ns = {row[0] for row in conn.execute("SELECT DISTINCT namespace FROM nodes").fetchall()} | {
            row[0] for row in conn.execute("SELECT DISTINCT namespace FROM edges").fetchall()
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
            created_at = node.created_at  # None is valid for migrated data
            updated_at = node.updated_at
            conn.execute(
                "INSERT INTO nodes (id, namespace, type, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    node.id,
                    namespace,
                    node.type,
                    json.dumps(node.properties),
                    created_at,
                    updated_at,
                ),
            )

        for edge in store.get_all_edges():
            created_at = edge.created_at  # None is valid for migrated data
            conn.execute(
                "INSERT INTO edges"
                " (id, namespace, type, source, confidence, properties,"
                "  created_at, valid_at, expired_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge.id,
                    namespace,
                    edge.type,
                    edge.source,
                    edge.confidence,
                    json.dumps(edge.properties),
                    created_at,
                    edge.valid_at,
                    edge.expired_at,
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
                    "INSERT INTO vertex_set_index (vertex_set_hash, edge_id, namespace) VALUES (?, ?, ?)",
                    (vsh, edge.id, namespace),
                )

        conn.commit()

    def load_namespace(self, namespace: str) -> HypergraphCore:
        """Load a single namespace from SQLite."""
        store = HypergraphCore()
        conn = self._conn

        for row in conn.execute(
            "SELECT id, type, properties, created_at, updated_at FROM nodes WHERE namespace = ?",
            (namespace,),
        ).fetchall():
            store.add_node(
                Node(
                    id=row[0],
                    type=row[1],
                    properties=_safe_json_loads(row[2], f"node {row[0]} properties"),
                    created_at=row[3],
                    updated_at=row[4],
                )
            )

        edge_rows = conn.execute(
            "SELECT id, type, source, confidence, properties,"
            " created_at, valid_at, expired_at"
            " FROM edges WHERE namespace = ?",
            (namespace,),
        ).fetchall()

        for erow in edge_rows:
            edge_id, etype, source, confidence, props_json = erow[:5]
            e_created_at, e_valid_at, e_expired_at = erow[5], erow[6], erow[7]
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
                    properties=_safe_json_loads(ir[3], f"incidence in edge {edge_id}"),
                )
                for ir in inc_rows
            ]
            store.add_edge(
                Hyperedge(
                    id=edge_id,
                    type=etype,
                    incidences=incidences,
                    properties=_safe_json_loads(props_json, f"edge {edge_id} properties"),
                    source=source,
                    confidence=confidence,
                    created_at=e_created_at,
                    valid_at=e_valid_at,
                    expired_at=e_expired_at,
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

    # --- Embedding storage ---

    def save_embedding(
        self,
        id: str,
        namespace: str,
        kind: str,
        ref_id: str,
        text: str,
        embedding: bytes,
        dimension: int,
        model: str,
    ) -> None:
        """Save or update an embedding."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings"
                " (id, namespace, kind, ref_id, text, embedding, dimension, model)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (id, namespace, kind, ref_id, text, embedding, dimension, model),
            )
            self._ensure_vec_table(dimension)
            row = self._conn.execute("SELECT rowid FROM embeddings WHERE id = ?", (id,)).fetchone()
            if row is None:
                raise RuntimeError(f"Embedding '{id}' not found after insert — database may be corrupted")
            rowid = row[0]
            # Delete old vec entry (handles upsert case)
            self._conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "INSERT INTO vec_embeddings(rowid, namespace_hash, embedding) VALUES (?, ?, ?)",
                (rowid, self._ns_hash(namespace), embedding),
            )
        except BaseException:
            # Roll back any implicit transaction so callers don't inherit a dirty state
            if self._tx_depth == 0 and self._conn.in_transaction:
                self._conn.rollback()
            raise
        self._auto_commit()

    def load_embeddings(
        self,
        namespace: str,
        kind: str | None = None,
    ) -> list[dict]:
        """Load embeddings for a namespace, optionally filtered by kind.

        Primarily used for testing and debugging — production search uses search_vec().
        """
        if kind is not None:
            rows = self._conn.execute(
                "SELECT id, kind, ref_id, text, embedding, dimension, model"
                " FROM embeddings WHERE namespace = ? AND kind = ?",
                (namespace, kind),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, kind, ref_id, text, embedding, dimension, model FROM embeddings WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "kind": r[1],
                "ref_id": r[2],
                "text": r[3],
                "embedding": r[4],
                "dimension": r[5],
                "model": r[6],
            }
            for r in rows
        ]

    def delete_embeddings(self, namespace: str, kind: str | None = None, ref_id: str | None = None) -> int:
        """Delete embeddings. Returns number of rows deleted."""
        # Collect affected rowids for vec table cleanup before deleting
        if self._vec_dimension is not None:
            if kind is not None and ref_id is not None:
                affected = self._conn.execute(
                    "SELECT rowid FROM embeddings WHERE namespace = ? AND kind = ? AND ref_id = ?",
                    (namespace, kind, ref_id),
                ).fetchall()
            elif kind is not None:
                affected = self._conn.execute(
                    "SELECT rowid FROM embeddings WHERE namespace = ? AND kind = ?",
                    (namespace, kind),
                ).fetchall()
            else:
                affected = self._conn.execute(
                    "SELECT rowid FROM embeddings WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
            for (rid,) in affected:
                self._conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (rid,))

        if kind is not None and ref_id is not None:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE namespace = ? AND kind = ? AND ref_id = ?",
                (namespace, kind, ref_id),
            )
        elif kind is not None:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE namespace = ? AND kind = ?",
                (namespace, kind),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE namespace = ?",
                (namespace,),
            )
        self._auto_commit()
        return cursor.rowcount

    # --- sqlite-vec support ---

    def _detect_vec_dimension(self) -> None:
        """Detect existing embedding dimension and ensure vec table is current."""
        has_embeddings = self._conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone()[0]
        if has_embeddings:
            row = self._conn.execute("SELECT dimension FROM embeddings LIMIT 1").fetchone()
            if row:
                # _vec_dimension is still None here, so _ensure_vec_table
                # will create/migrate the vec table and set _vec_dimension.
                self._ensure_vec_table(row[0])

    @staticmethod
    def _ns_hash(namespace: str) -> int:
        """Deterministic integer hash of namespace for vec0 partition key."""
        return int(hashlib.sha256(namespace.encode()).hexdigest()[:15], 16)

    def _ensure_vec_table(self, dimension: int) -> None:
        """Create the vec0 virtual table on first use and backfill existing embeddings."""
        if self._vec_dimension == dimension:
            return
        if self._vec_dimension is not None:
            raise ValueError(
                f"Embedding dimension mismatch: vec table has {self._vec_dimension} dimensions, "
                f"but got {dimension}. All embeddings must use the same dimension. "
                f"To switch embedding models, delete existing embeddings first."
            )
        # Check existing vec table schema via sqlite_master DDL
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
        ).fetchone()
        if row and "partition key" in row[0]:
            # Table already has the correct schema — no rebuild needed
            self._vec_dimension = dimension
            return
        if row:
            # Old table without partition key — drop before recreating
            self._conn.execute("DROP TABLE vec_embeddings")
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings"
            f" USING vec0("
            f"namespace_hash integer partition key,"
            f" embedding float[{dimension}] distance_metric=cosine)"
        )
        # Backfill existing embeddings into the fresh vec table
        rows = self._conn.execute(
            "SELECT rowid, namespace, embedding FROM embeddings WHERE dimension = ?",
            (dimension,),
        ).fetchall()
        self._conn.executemany(
            "INSERT INTO vec_embeddings(rowid, namespace_hash, embedding) VALUES (?, ?, ?)",
            [(rowid, self._ns_hash(ns), blob) for rowid, ns, blob in rows],
        )
        self._auto_commit()
        self._vec_dimension = dimension

    def rebuild_vec_index(self) -> int:
        """Rebuild vec_embeddings from the authoritative embeddings table.

        Returns number of embeddings re-indexed, or 0 if none exist.
        """
        row = self._conn.execute("SELECT dimension FROM embeddings LIMIT 1").fetchone()
        if row is None:
            return 0
        dimension = row[0]

        # Drop and recreate via _ensure_vec_table's backfill path
        existing = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
        ).fetchone()
        if existing:
            self._conn.execute("DROP TABLE vec_embeddings")
        self._vec_dimension = None
        self._ensure_vec_table(dimension)

        count: int = self._conn.execute(
            "SELECT count(*) FROM embeddings WHERE dimension = ?",
            (dimension,),
        ).fetchone()[0]
        return count

    def search_vec(
        self,
        namespace: str,
        query_embedding: bytes,
        *,
        limit: int = 10,
        kind: str | None = None,
        type_filter: str | None = None,
    ) -> list[dict]:
        """KNN search using sqlite-vec. Returns results with scores.

        Two-step approach: KNN from vec0 (scoped by namespace partition key),
        then metadata lookup for remaining filters.
        """
        # Over-fetch from KNN to compensate for post-hoc filtering.
        # With filters (kind/type), more KNN candidates get discarded,
        # so use a higher multiplier to avoid returning fewer than `limit`.
        fetch_limit = limit * 4 if (kind or type_filter) else limit * 2
        knn_rows = self._conn.execute(
            "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? AND k = ? AND namespace_hash = ?",
            (query_embedding, fetch_limit, self._ns_hash(namespace)),
        ).fetchall()

        if not knn_rows:
            return []

        rowids = [r[0] for r in knn_rows]
        dist_map = {r[0]: r[1] for r in knn_rows}

        placeholders = ",".join("?" for _ in rowids)

        # Build metadata query with optional type filter join
        _TABLE_FOR_KIND = {"edge": "edges", "node": "nodes"}
        if type_filter is not None and kind in _TABLE_FOR_KIND:
            ref_table = _TABLE_FOR_KIND[kind]
            meta_rows = self._conn.execute(
                f"SELECT e.rowid, e.id, e.kind, e.ref_id, e.text, e.dimension, e.model"
                f" FROM embeddings e"
                f" JOIN {ref_table} t ON e.ref_id = t.id AND e.namespace = t.namespace"
                f" WHERE e.namespace = ? AND e.kind = ? AND t.type = ?"
                f" AND e.rowid IN ({placeholders})",
                [namespace, kind, type_filter, *rowids],
            ).fetchall()
        elif kind is not None:
            meta_rows = self._conn.execute(
                f"SELECT rowid, id, kind, ref_id, text, dimension, model"
                f" FROM embeddings WHERE namespace = ? AND kind = ?"
                f" AND rowid IN ({placeholders})",
                [namespace, kind, *rowids],
            ).fetchall()
        else:
            meta_rows = self._conn.execute(
                f"SELECT rowid, id, kind, ref_id, text, dimension, model"
                f" FROM embeddings WHERE namespace = ?"
                f" AND rowid IN ({placeholders})",
                [namespace, *rowids],
            ).fetchall()

        results = []
        for r in meta_rows:
            rid = r[0]
            distance = dist_map.get(rid, 1.0)
            # Cosine distance is in [0, 2] for unit vectors; 1-distance gives [-1, 1].
            # Clamp to [0, 1]: min guards float rounding above 1, max floors
            # dissimilar (negative) scores to 0 since they're never useful in search.
            score = max(0.0, min(1.0, 1.0 - distance))
            results.append(
                {
                    "id": r[1],
                    "kind": r[2],
                    "ref_id": r[3],
                    "text": r[4],
                    "dimension": r[5],
                    "model": r[6],
                    "score": score,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # --- Access log ---

    def record_access(self, namespace: str, kind: str, ref_id: str) -> None:
        """Record an access event, incrementing the counter."""
        now = time.time()
        self._conn.execute(
            "INSERT INTO access_log (id, namespace, kind, ref_id, last_accessed, access_count)"
            " VALUES (?, ?, ?, ?, ?, 1)"
            " ON CONFLICT (namespace, kind, ref_id)"
            " DO UPDATE SET last_accessed = ?, access_count = access_count + 1",
            (f"{namespace}:{kind}:{ref_id}", namespace, kind, ref_id, now, now),
        )
        self._auto_commit()

    def record_access_batch(self, namespace: str, kind: str, ref_ids: list[str]) -> None:
        """Record access events for multiple items in a single commit."""
        if not ref_ids:
            return
        now = time.time()
        self._conn.executemany(
            "INSERT INTO access_log (id, namespace, kind, ref_id, last_accessed, access_count)"
            " VALUES (?, ?, ?, ?, ?, 1)"
            " ON CONFLICT (namespace, kind, ref_id)"
            " DO UPDATE SET last_accessed = ?, access_count = access_count + 1",
            [(f"{namespace}:{kind}:{rid}", namespace, kind, rid, now, now) for rid in ref_ids],
        )
        self._auto_commit()

    def get_access_stats(self, namespace: str, kind: str, ref_id: str) -> dict | None:
        """Get access stats for a specific item."""
        row = self._conn.execute(
            "SELECT last_accessed, access_count FROM access_log WHERE namespace = ? AND kind = ? AND ref_id = ?",
            (namespace, kind, ref_id),
        ).fetchone()
        if row is None:
            return None
        return {"last_accessed": row[0], "access_count": row[1]}

    def get_batch_access_stats(self, namespace: str, kind: str, ref_ids: list[str]) -> dict[str, dict]:
        """Get access stats for multiple items in a single query.

        Returns a dict mapping ref_id -> stats dict. Missing items are omitted.
        """
        if not ref_ids:
            return {}
        placeholders = ",".join("?" for _ in ref_ids)
        rows = self._conn.execute(
            f"SELECT ref_id, last_accessed, access_count FROM access_log"
            f" WHERE namespace = ? AND kind = ? AND ref_id IN ({placeholders})",
            [namespace, kind, *ref_ids],
        ).fetchall()
        return {r[0]: {"last_accessed": r[1], "access_count": r[2]} for r in rows}

    def get_all_access_stats(self, namespace: str) -> list[dict]:
        """Get all access stats for a namespace."""
        rows = self._conn.execute(
            "SELECT kind, ref_id, last_accessed, access_count FROM access_log WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        return [
            {
                "kind": r[0],
                "ref_id": r[1],
                "last_accessed": r[2],
                "access_count": r[3],
            }
            for r in rows
        ]

    # --- Incremental write methods ---

    def _auto_commit(self) -> None:
        """Commit if not inside a transaction."""
        if self._tx_depth == 0:
            self._conn.commit()

    def write_node(self, namespace: str, node: Node) -> None:
        """Persist a single node (INSERT OR REPLACE)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes"
            " (id, namespace, type, properties, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                node.id,
                namespace,
                node.type,
                json.dumps(node.properties),
                node.created_at,
                node.updated_at,
            ),
        )
        self._auto_commit()

    def write_edge(self, namespace: str, edge: Hyperedge) -> None:
        """Persist a single edge with all incidences and vertex-set index."""
        conn = self._conn
        # Remove old incidences and vertex-set index for this edge
        conn.execute(
            "DELETE FROM incidences WHERE edge_id = ? AND namespace = ?",
            (edge.id, namespace),
        )
        conn.execute(
            "DELETE FROM vertex_set_index WHERE edge_id = ? AND namespace = ?",
            (edge.id, namespace),
        )
        # Upsert the edge row
        conn.execute(
            "INSERT OR REPLACE INTO edges"
            " (id, namespace, type, source, confidence, properties,"
            "  created_at, valid_at, expired_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                edge.id,
                namespace,
                edge.type,
                edge.source,
                edge.confidence,
                json.dumps(edge.properties),
                edge.created_at,
                edge.valid_at,
                edge.expired_at,
            ),
        )
        # Insert incidences
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
        # Insert vertex-set index
        if edge.node_set:
            vsh = _vertex_set_hash(edge.node_set)
            conn.execute(
                "INSERT OR IGNORE INTO vertex_set_index (vertex_set_hash, edge_id, namespace) VALUES (?, ?, ?)",
                (vsh, edge.id, namespace),
            )
        self._auto_commit()

    def remove_node(self, namespace: str, node_id: str) -> None:
        """Delete a single node from SQLite."""
        self._conn.execute(
            "DELETE FROM nodes WHERE id = ? AND namespace = ?",
            (node_id, namespace),
        )
        self._auto_commit()

    def remove_edge(self, namespace: str, edge_id: str) -> None:
        """Delete a single edge and its incidences/index from SQLite.

        Child rows in incidences and vertex_set_index are removed by
        ON DELETE CASCADE.
        """
        self._conn.execute(
            "DELETE FROM edges WHERE id = ? AND namespace = ?",
            (edge_id, namespace),
        )
        self._auto_commit()

    def update_edge(self, namespace: str, edge: Hyperedge) -> None:
        """Update scalar fields on an existing edge without rewriting incidences."""
        self._conn.execute(
            "UPDATE edges SET source = ?, confidence = ?, properties = ?,"
            " expired_at = ?, valid_at = ?"
            " WHERE id = ? AND namespace = ?",
            (
                edge.source,
                edge.confidence,
                json.dumps(edge.properties),
                edge.expired_at,
                edge.valid_at,
                edge.id,
                namespace,
            ),
        )
        self._auto_commit()

    # --- Transaction methods ---

    def begin(self) -> None:
        """Begin a transaction (or increment nesting depth)."""
        if self._tx_depth == 0:
            self._conn.execute("BEGIN IMMEDIATE")
        self._tx_depth += 1

    def commit(self) -> None:
        """Commit the transaction (or decrement nesting depth)."""
        if self._tx_depth <= 0:
            raise RuntimeError("commit() called without matching begin()")
        self._tx_depth -= 1
        if self._tx_depth == 0:
            self._conn.commit()

    def rollback(self) -> None:
        """Rollback the transaction, discarding all uncommitted writes."""
        if self._tx_depth == 0:
            return
        try:
            self._conn.rollback()
        finally:
            self._tx_depth = 0

    # --- Properties ---

    @property
    def vec_dimension(self) -> int | None:
        """Current vector dimension, or None if not initialized."""
        return self._vec_dimension
