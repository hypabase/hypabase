"""Hypabase client — the primary interface for interacting with a hypergraph."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from hypabase.engine.core import (
    Hyperedge as CoreEdge,
)
from hypabase.engine.core import (
    HypergraphCore,
)
from hypabase.engine.core import (
    Incidence as CoreIncidence,
)
from hypabase.engine.core import (
    Node as CoreNode,
)
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.engine.persistence_engine import PersistenceEngine
from hypabase.engine.storage import SQLiteStorage
from hypabase.models import Edge, HypergraphStats, Incidence, Node, ValidationResult

logger = logging.getLogger(__name__)

# --- Conversion helpers: engine core types <-> pydantic models ---


def _ts_to_dt(ts: float | None) -> datetime | None:
    """Convert a Unix timestamp to a timezone-aware datetime, or None."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC)


def _dt_to_ts(dt: datetime | None) -> float | None:
    """Convert a datetime to a Unix timestamp, or None."""
    if dt is None:
        return None
    return dt.timestamp()


def _core_node_to_model(cn: CoreNode) -> Node:
    return Node(
        id=cn.id,
        type=cn.type,
        properties=cn.properties,
        created_at=_ts_to_dt(cn.created_at),
        updated_at=_ts_to_dt(cn.updated_at),
    )


def _core_edge_to_model(ce: CoreEdge) -> Edge:
    return Edge(
        id=ce.id,
        type=ce.type,
        incidences=[
            Incidence(
                node_id=inc.node_id,
                edge_ref_id=inc.edge_ref_id,
                direction=cast(Literal["head", "tail"] | None, inc.direction),
                properties=inc.properties,
            )
            for inc in ce.incidences
        ],
        directed=ce.is_directed,
        source=ce.source,
        confidence=ce.confidence,
        properties=ce.properties,
        created_at=_ts_to_dt(ce.created_at),
        updated_at=None,
        valid_at=_ts_to_dt(ce.valid_at),
        expired_at=_ts_to_dt(ce.expired_at),
    )


class Hypabase:
    """Hypergraph client.

    The primary interface for creating, querying, and traversing hypergraphs.
    Supports in-memory and local SQLite backends.

    Constructor patterns:
        - ``Hypabase()`` — in-memory, ephemeral (SQLite ``:memory:``)
        - ``Hypabase("file.db")`` — local persistent SQLite file
        - ``Hypabase("https://...")`` — cloud backend (not yet supported, raises NotImplementedError)

    Example:
        ```python
        hb = Hypabase()                          # in-memory
        hb = Hypabase("myproject.db")             # local SQLite file

        # Namespace isolation
        drugs = hb.database("drugs")
        sessions = hb.database("sessions")
        ```
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        key: str | None = None,
        database: str = "default",
        embedder: EmbeddingProvider | None = None,
        auto_embed: bool = False,
        # Private: shared state for namespace views
        _storage: PersistenceEngine | None = ...,  # type: ignore[assignment]
        _stores: dict[str, HypergraphCore] | None = None,
        _embedder: EmbeddingProvider | None = ...,  # type: ignore[assignment]
    ) -> None:
        path_str = str(path) if path else None

        # Detect cloud URLs
        if path_str and (path_str.startswith("http://") or path_str.startswith("https://")):
            raise NotImplementedError(
                "Cloud backends are not yet supported. "
                "Use Hypabase() for in-memory or Hypabase('file.db') for local SQLite."
            )

        # Check if this is a namespace view (internal construction)
        if _storage is not ...:
            # Namespace view — share storage and stores dict
            self._path = path_str
            self._storage = _storage
            self._stores = _stores or {"default": HypergraphCore()}
            self._current_ns = database
            self._embedder = _embedder if _embedder is not ... else embedder
            # Ensure the namespace store exists
            if self._current_ns not in self._stores:
                self._stores[self._current_ns] = HypergraphCore()
        else:
            # Normal construction
            self._path = path_str
            self._current_ns = database
            self._embedder = embedder
            if self._path:
                self._storage = SQLiteStorage(self._path)
                self._stores = self._storage.load()
            else:
                self._storage = None
                self._stores = {}
            # Ensure the requested namespace exists
            if self._current_ns not in self._stores:
                self._stores[self._current_ns] = HypergraphCore()

        self._context_source: str | None = None
        self._context_confidence: float | None = None
        self._auto_embed: bool = auto_embed

    @property
    def _store(self) -> HypergraphCore:
        """Resolve the current namespace's store."""
        return self._stores[self._current_ns]

    @property
    def storage(self) -> PersistenceEngine | None:
        """The underlying storage adapter, or None if in-memory only."""
        return self._storage

    @property
    def current_namespace(self) -> str:
        """The active namespace name."""
        return self._current_ns

    @property
    def embedder(self) -> Any:
        """The configured embedding provider, or None."""
        return self._embedder

    def close(self) -> None:
        """Close the database connection.

        Saves pending changes and releases the SQLite connection.
        No-op for in-memory instances.
        """
        if self._storage:
            try:
                self._storage.save(self._stores)
            finally:
                self._storage.close()

    def save(self) -> None:
        """Persist all namespaces to SQLite as a full snapshot.

        No-op for in-memory instances. Individual mutations auto-commit
        incrementally; this method performs a full overwrite of all
        namespace data.
        """
        if self._storage:
            self._storage.save(self._stores)

    def __enter__(self) -> Hypabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Namespace management ---

    @property
    def current_database(self) -> str:
        """Current namespace name."""
        return self._current_ns

    def database(self, name: str) -> Hypabase:
        """Return a scoped view into a named namespace.

        The returned instance shares the same SQLite connection and stores
        dict, but reads/writes only the given namespace's data.

        Args:
            name: Namespace name.

        Returns:
            A new Hypabase instance scoped to the namespace.
        """
        if name not in self._stores:
            # Load from SQLite if persistent and namespace exists on disk
            if self._storage and name in (self._storage.list_namespaces()):
                self._stores[name] = self._storage.load_namespace(name)
            else:
                self._stores[name] = HypergraphCore()
        return Hypabase(
            self._path,
            database=name,
            _storage=self._storage,
            _stores=self._stores,
            _embedder=self._embedder,
        )

    def databases(self) -> list[str]:
        """List all namespaces.

        Returns:
            Sorted list of namespace names.
        """
        ns_set = set(self._stores.keys())
        # Also include any namespaces on disk not yet loaded
        if self._storage:
            ns_set.update(self._storage.list_namespaces())
        return sorted(ns_set)

    def delete_database(self, name: str) -> bool:
        """Delete a namespace and all its data.

        Args:
            name: Namespace to delete.

        Returns:
            ``True`` if the namespace existed, ``False`` otherwise.
        """
        existed = name in self._stores
        if self._storage:
            existed = existed or name in self._storage.list_namespaces()
            self._storage.delete_namespace(name)
        if name in self._stores:
            del self._stores[name]
        return existed

    # --- Provenance context ---

    @contextmanager
    def context(self, *, source: str, confidence: float = 1.0) -> Generator[None, None, None]:
        """Set default provenance for all edges created within the block.

        Edges created inside the context inherit ``source`` and ``confidence``
        unless overridden per-edge. Contexts can be nested; the innermost wins.

        Args:
            source: Provenance source string (e.g., ``"gpt-4o_extraction"``).
            confidence: Default confidence score, 0.0-1.0.

        Example:
            ```python
            with hb.context(source="clinical_records", confidence=0.95):
                hb.edge(["a", "b"], type="link")  # inherits provenance
            ```
        """
        prev_source = self._context_source
        prev_confidence = self._context_confidence
        self._context_source = source
        self._context_confidence = confidence
        try:
            yield
        finally:
            self._context_source = prev_source
            self._context_confidence = prev_confidence

    # --- Nodes ---

    def node(self, id: str, *, type: str = "unknown", **properties: Any) -> Node:
        """Create or update a node.

        If a node with the given ID exists, its type and properties are updated.
        Otherwise a new node is created.

        Args:
            id: Unique node identifier.
            type: Node classification (e.g., ``"doctor"``, ``"patient"``).
            **properties: Arbitrary key-value metadata stored on the node.

        Returns:
            The created or updated Node.

        Raises:
            ValueError: If ``id`` is an empty string.
        """
        if not id:
            raise ValueError("Node ID must be a non-empty string")
        now = time.time()
        existing = self._store.get_node(id)
        if existing:
            existing.type = type
            if properties:
                existing.properties.update(properties)
            existing.updated_at = now
        else:
            self._store.add_node(CoreNode(id=id, type=type, properties=properties, created_at=now, updated_at=now))
        core_node = self._store.get_node(id)
        if core_node is None:
            raise RuntimeError(f"Internal consistency error: Node {id!r} should exist after add_node")
        if self._storage:
            self._storage.write_node(self._current_ns, core_node)
        return _core_node_to_model(core_node)

    def get_node(self, id: str) -> Node | None:
        """Get a node by ID.

        Args:
            id: The node ID to look up.

        Returns:
            The Node if found, or ``None``.
        """
        cn = self._store.get_node(id)
        return _core_node_to_model(cn) if cn else None

    def nodes(self, *, type: str | None = None) -> list[Node]:
        """Query nodes, optionally filtered by type.

        Args:
            type: If provided, return only nodes of this type.

        Returns:
            List of matching nodes.
        """
        if type is not None:
            return [_core_node_to_model(n) for n in self._store.get_nodes_by_type(type)]
        return [_core_node_to_model(n) for n in self._store.get_all_nodes()]

    def find_nodes(self, **properties: Any) -> list[Node]:
        """Find nodes matching all specified properties.

        Args:
            **properties: Key-value pairs that must match node properties.

        Returns:
            List of matching nodes.

        Example:
            ```python
            hb.find_nodes(role="admin", active=True)
            ```
        """
        return [_core_node_to_model(n) for n in self._store.find_nodes(**properties)]

    def has_node(self, id: str) -> bool:
        """Check if a node exists.

        Args:
            id: The node ID to check.

        Returns:
            ``True`` if the node exists, ``False`` otherwise.
        """
        return self._store.has_node(id)

    def delete_node(self, id: str, *, cascade: bool = False) -> bool:
        """Delete a node by ID.

        Args:
            id: The node ID to delete.
            cascade: If ``True``, also delete all incident edges.

        Returns:
            ``True`` if the node existed and was deleted, ``False`` otherwise.
        """
        if cascade:
            edge_ids = list(self._store._node_to_edges.get(id, set()))
            deleted, _ = self._store.delete_node_cascade(id)
            if deleted and self._storage:
                for eid in edge_ids:
                    self._storage.remove_edge(self._current_ns, eid)
                self._storage.remove_node(self._current_ns, id)
            return deleted
        result = self._store.delete_node(id)
        if result and self._storage:
            self._storage.remove_node(self._current_ns, id)
        return result

    def delete_node_cascade(self, node_id: str) -> tuple[bool, int]:
        """Delete a node and all its incident edges.

        .. deprecated:: 0.2.0
            Use ``delete_node(id, cascade=True)`` instead.

        Args:
            node_id: The node ID to delete.

        Returns:
            Tuple of ``(node_was_deleted, number_of_edges_deleted)``.
        """
        edge_ids = list(self._store._node_to_edges.get(node_id, set()))
        result = self._store.delete_node_cascade(node_id)
        if result[0] and self._storage:
            for eid in edge_ids:
                self._storage.remove_edge(self._current_ns, eid)
            self._storage.remove_node(self._current_ns, node_id)
        return result

    # --- Edges ---

    def edge(
        self,
        nodes: list[str] | None = None,
        *,
        type: str,
        directed: bool = False,
        source: str | None = None,
        confidence: float | None = None,
        properties: dict[str, Any] | None = None,
        id: str | None = None,
        valid_at: datetime | None = None,
        roles: list[str | None] | None = None,
        incidences: list[dict[str, Any]] | None = None,
    ) -> Edge:
        """Create a hyperedge linking two or more nodes in one relationship.

        Nodes are auto-created if they don't exist. Provenance values
        fall back to the active ``context()`` block if not set explicitly.

        There are two ways to specify participants:

        1. **nodes** (simple): A list of node IDs. Use ``roles`` for kāraka.
        2. **incidences** (advanced): A list of dicts, each with ``node_id``
           or ``edge_ref_id``, plus optional ``properties``. Supports mixed
           node and edge-ref incidences (metagraph patterns).

        Exactly one of ``nodes`` or ``incidences`` must be provided.

        Args:
            nodes: Node IDs to connect. Must contain at least 2.
            type: Edge type (e.g., ``"treatment"``, ``"concept_link"``).
            directed: If ``True``, first node is tail, last is head.
            source: Provenance source. Falls back to context or ``"unknown"``.
            confidence: Confidence score 0.0-1.0. Falls back to context or ``1.0``.
            properties: Arbitrary key-value metadata.
            id: Optional edge ID. Auto-generated UUID if omitted.
            roles: Optional per-node semantic roles (kāraka). Only valid with
                ``nodes``. ``len(roles)`` must equal ``len(nodes)``.
            incidences: List of incidence dicts. Each dict must have either
                ``node_id`` (str) or ``edge_ref_id`` (str), and may have
                ``properties`` (dict). Cannot be used with ``nodes``/``roles``.

        Returns:
            The created Edge.

        Raises:
            ValueError: If arguments are invalid.

        Example:
            ```python
            hb.edge(
                ["dr_smith", "patient_123", "aspirin"],
                type="treatment",
                source="clinical_records",
                confidence=0.95,
                roles=["subject", "object", "instrument"],
            )
            ```
        """
        if nodes is not None and incidences is not None:
            raise ValueError("Provide either 'nodes' or 'incidences', not both.")
        if nodes is None and incidences is None:
            raise ValueError("Provide either 'nodes' or 'incidences'.")

        edge_id = id or str(uuid.uuid4())
        resolved_source = source or self._context_source or "unknown"
        resolved_confidence = (
            confidence
            if confidence is not None
            else (self._context_confidence if self._context_confidence is not None else 1.0)
        )

        if incidences is not None:
            # Advanced path: mixed node_id + edge_ref_id incidences
            if roles is not None:
                raise ValueError("'roles' cannot be used with 'incidences'. Put roles in incidence properties.")
            if len(incidences) < 2:
                raise ValueError("A hyperedge must have at least 2 incidences.")

            core_incidences: list[CoreIncidence] = []
            for inc_dict in incidences:
                nid = inc_dict.get("node_id")
                erid = inc_dict.get("edge_ref_id")
                inc_props = dict(inc_dict.get("properties", {}))

                if nid is not None:
                    # Auto-create node if needed
                    if not self._store.get_node(nid):
                        new_node = CoreNode(id=nid, type="unknown")
                        self._store.add_node(new_node)
                        if self._storage:
                            self._storage.write_node(self._current_ns, new_node)
                    core_incidences.append(CoreIncidence(node_id=nid, properties=inc_props))
                elif erid is not None:
                    core_incidences.append(CoreIncidence(edge_ref_id=erid, properties=inc_props))
                else:
                    raise ValueError("Each incidence must have either 'node_id' or 'edge_ref_id'.")
        else:
            # Original path: nodes list
            if nodes is None:
                raise ValueError("Either 'nodes' or 'incidences' must be provided.")
            if len(nodes) < 2:
                raise ValueError("A hyperedge must connect at least 2 nodes.")
            if any(not n for n in nodes):
                raise ValueError("Node IDs must be non-empty strings")
            if roles is not None and len(roles) != len(nodes):
                raise ValueError(f"roles length ({len(roles)}) must match nodes length ({len(nodes)})")

            # Auto-create nodes
            for node_id in nodes:
                if not self._store.get_node(node_id):
                    new_node = CoreNode(id=node_id, type="unknown")
                    self._store.add_node(new_node)
                    if self._storage:
                        self._storage.write_node(self._current_ns, new_node)

            # Build incidences — for directed edges, first node is tail, last is head
            def _inc_props(idx: int) -> dict[str, Any]:
                if roles is not None and roles[idx] is not None:
                    return {"role": roles[idx]}
                return {}

            if directed:
                core_incidences = [
                    CoreIncidence(node_id=nodes[0], direction="tail", properties=_inc_props(0)),
                    *[CoreIncidence(node_id=nodes[i], properties=_inc_props(i)) for i in range(1, len(nodes) - 1)],
                    CoreIncidence(node_id=nodes[-1], direction="head", properties=_inc_props(len(nodes) - 1)),
                ]
            else:
                core_incidences = [CoreIncidence(node_id=n, properties=_inc_props(i)) for i, n in enumerate(nodes)]

        core_edge = CoreEdge(
            id=edge_id,
            type=type,
            incidences=core_incidences,
            properties=properties or {},
            source=resolved_source,
            confidence=resolved_confidence,
            created_at=time.time(),
            valid_at=_dt_to_ts(valid_at),
        )

        # Use upsert to handle existing edge IDs
        self._store.upsert_edge(core_edge)
        stored_edge = self._store.get_edge(edge_id)
        if stored_edge is None:
            raise RuntimeError(f"Internal consistency error: Edge {edge_id!r} should exist after upsert_edge")
        if self._storage:
            self._storage.write_edge(self._current_ns, stored_edge)
        return _core_edge_to_model(stored_edge)

    def get_edge(self, id: str) -> Edge | None:
        """Get an edge by ID.

        Args:
            id: The edge ID to look up.

        Returns:
            The Edge if found, or ``None``.
        """
        ce = self._store.get_edge(id)
        return _core_edge_to_model(ce) if ce else None

    def edges(
        self,
        *,
        containing: list[str] | None = None,
        type: str | None = None,
        match_all: bool = False,
        source: str | None = None,
        min_confidence: float | None = None,
        active: bool = True,
        include_expired: bool = False,
        since: datetime | None = None,
        before: datetime | None = None,
        at: datetime | None = None,
    ) -> list[Edge]:
        """Query edges by contained nodes, type, source, confidence, and temporal criteria.

        All filters are combined with AND logic.

        Args:
            containing: Node IDs that must appear in the edge.
            type: Filter to edges of this type.
            match_all: If ``True``, edges must contain *all* nodes in
                ``containing``. If ``False`` (default), any match suffices.
            source: Filter to edges from this provenance source.
            min_confidence: Filter to edges with confidence >= this value.
            active: If ``True`` (default), only return non-expired edges.
            include_expired: If ``True``, include expired edges (overrides ``active``).
            since: Only edges created at or after this time.
            before: Only edges created before this time.
            at: Point-in-time query: edges that were valid at this moment.

        Returns:
            List of matching edges.

        Example:
            ```python
            hb.edges(containing=["patient_123"], min_confidence=0.9)
            ```
        """
        if containing:
            core_edges = self._store.get_edges_containing(
                set(containing),
                match_all=match_all,
            )
        elif type:
            core_edges = self._store.get_edges_by_type(type)
        else:
            core_edges = self._store.get_all_edges()

        if type and containing:
            core_edges = [e for e in core_edges if e.type == type]
        if source is not None:
            core_edges = [e for e in core_edges if e.source == source]
        if min_confidence is not None:
            core_edges = [e for e in core_edges if e.confidence >= min_confidence]

        # Temporal filtering
        core_edges = self._store.filter_temporal(
            core_edges,
            active=active,
            include_expired=include_expired,
            since=_dt_to_ts(since),
            before=_dt_to_ts(before),
            at=_dt_to_ts(at),
        )

        return [_core_edge_to_model(e) for e in core_edges]

    def find_edges(self, **properties: Any) -> list[Edge]:
        """Find edges matching all specified properties.

        Args:
            **properties: Key-value pairs that must match edge properties.

        Returns:
            List of matching edges.
        """
        return [_core_edge_to_model(e) for e in self._store.find_edges(**properties)]

    def has_edge_with_nodes(
        self,
        node_ids: set[str],
        edge_type: str | None = None,
    ) -> bool:
        """Check if an edge with the exact vertex set exists.

        Args:
            node_ids: Exact set of node IDs.
            edge_type: If provided, also filter by edge type.

        Returns:
            ``True`` if matching edge exists.
        """
        return self._store.has_edge_with_nodes(node_ids, edge_type)

    def sources(self) -> list[dict[str, Any]]:
        """Summarize provenance sources across all edges.

        Returns:
            List of dicts with keys ``"source"``, ``"edge_count"``,
            and ``"avg_confidence"`` for each unique source.

        Example:
            ```python
            hb.sources()
            # [{"source": "clinical_records", "edge_count": 2, "avg_confidence": 0.95}]
            ```
        """
        all_edges = self._store.get_all_edges()
        source_data: dict[str, list[float]] = {}
        for e in all_edges:
            source_data.setdefault(e.source, []).append(e.confidence)
        return [
            {
                "source": src,
                "edge_count": len(confs),
                "avg_confidence": round(sum(confs) / len(confs), 4),
            }
            for src, confs in sorted(source_data.items())
        ]

    def edges_by_vertex_set(self, nodes: list[str]) -> list[Edge]:
        """O(1) lookup: find edges with exactly this set of nodes.

        Uses the in-memory vertex-set hash index for constant-time lookup.
        Order of ``nodes`` does not matter.

        Args:
            nodes: The exact set of node IDs to match.

        Returns:
            Edges whose node set matches exactly.
        """
        core_edges = self._store.get_edges_by_node_set(set(nodes))
        return [_core_edge_to_model(e) for e in core_edges]

    def delete_edge(self, id: str) -> bool:
        """Delete an edge by ID.

        Args:
            id: The edge ID to delete.

        Returns:
            ``True`` if the edge existed and was deleted, ``False`` otherwise.
        """
        result = self._store.delete_edge(id)
        if result and self._storage:
            self._storage.remove_edge(self._current_ns, id)
        return result

    def expire_edge(self, id: str) -> Edge | None:
        """Expire an edge, marking it as no longer active.

        The edge is not deleted — it remains queryable via ``include_expired=True``.

        Args:
            id: The edge ID to expire.

        Returns:
            The expired Edge, or ``None`` if not found.
        """
        if not self._store.expire_edge(id):
            return None
        ce = self._store.get_edge(id)
        if ce is None:
            raise RuntimeError(f"Edge {id!r} should exist after expire_edge")
        if self._storage:
            self._storage.update_edge(self._current_ns, ce)
        return _core_edge_to_model(ce)

    def supersede_edge(
        self,
        old_edge_id: str,
        nodes: list[str],
        *,
        type: str,
        source: str | None = None,
        confidence: float | None = None,
        properties: dict[str, Any] | None = None,
        valid_at: datetime | None = None,
    ) -> tuple[Edge, Edge] | None:
        """Expire an edge and create a replacement atomically.

        Args:
            old_edge_id: The edge to expire.
            nodes: Node IDs for the new edge.
            type: Edge type for the new edge.
            source: Provenance source for the new edge.
            confidence: Confidence for the new edge.
            properties: Properties for the new edge.
            valid_at: When the new fact became true.

        Returns:
            Tuple of (expired_edge, new_edge), or ``None`` if old edge not found.
        """
        if len(nodes) < 2:
            raise ValueError("A hyperedge must connect at least 2 nodes.")
        for node_id in nodes:
            if not self._store.get_node(node_id):
                new_node = CoreNode(id=node_id, type="unknown")
                self._store.add_node(new_node)
                if self._storage:
                    self._storage.write_node(self._current_ns, new_node)

        resolved_source = source or self._context_source or "unknown"
        resolved_confidence = (
            confidence
            if confidence is not None
            else (self._context_confidence if self._context_confidence is not None else 1.0)
        )

        incidences = [CoreIncidence(node_id=n) for n in nodes]
        result = self._store.supersede_edge(
            old_edge_id,
            type=type,
            incidences=incidences,
            properties=properties or {},
            source=resolved_source,
            confidence=resolved_confidence,
            valid_at=_dt_to_ts(valid_at),
        )
        if result is None:
            return None
        old_core, new_core = result
        if self._storage:
            self._storage.update_edge(self._current_ns, old_core)
            self._storage.write_edge(self._current_ns, new_core)
        return (_core_edge_to_model(old_core), _core_edge_to_model(new_core))

    # --- Traversal ---

    def neighbors(self, node_id: str, *, edge_types: list[str] | None = None) -> list[Node]:
        """Find all nodes connected to the given node via shared edges.

        The query node itself is excluded from the results.

        Args:
            node_id: The node to find neighbors of.
            edge_types: If provided, only traverse edges of these types.

        Returns:
            List of neighboring nodes.
        """
        neighbor_ids = self._store.get_neighbor_nodes(
            node_id,
            edge_types=edge_types,
            exclude_self=True,
        )
        return [_core_node_to_model(n) for nid in neighbor_ids if (n := self._store.get_node(nid)) is not None]

    def paths(
        self,
        start: str,
        end: str,
        *,
        max_hops: int = 5,
        edge_types: list[str] | None = None,
    ) -> list[list[str]]:
        """Find paths between two nodes through hyperedges.

        Uses breadth-first search. Each path is a list of node IDs
        from ``start`` to ``end``.

        Args:
            start: Starting node ID.
            end: Target node ID.
            max_hops: Maximum number of hops (default 5).
            edge_types: If provided, only traverse edges of these types.

        Returns:
            List of paths, where each path is a list of node IDs.

        Example:
            ```python
            paths = hb.paths("dr_smith", "mercy_hospital")
            # [["dr_smith", "patient_123", "mercy_hospital"]]
            ```
        """
        if start == end:
            return [[start]]

        visited: set[str] = {start}
        queue: deque[list[str]] = deque([[start]])
        results: list[list[str]] = []

        while queue:
            path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue

            current = path[-1]
            for nid in self._store.get_neighbor_nodes(
                current,
                edge_types=edge_types,
                exclude_self=True,
            ):
                if nid == end:
                    results.append([*path, end])
                elif nid not in visited:
                    visited.add(nid)
                    queue.append([*path, nid])

        return results

    # --- New methods powered by HypergraphCore ---

    def find_paths(
        self,
        start_nodes: set[str],
        end_nodes: set[str],
        *,
        max_hops: int = 3,
        max_paths: int = 10,
        min_intersection: int = 1,
        edge_types: list[str] | None = None,
        direction_mode: str = "undirected",
    ) -> list[list[Edge]]:
        """Find paths between two groups of nodes through shared edges.

        Returns paths as sequences of edges. Supports set-based start/end
        nodes and configurable overlap requirements.

        Args:
            start_nodes: Set of possible starting node IDs.
            end_nodes: Set of possible ending node IDs.
            max_hops: Maximum path length in edges (default 3).
            max_paths: Maximum number of paths to return (default 10).
            min_intersection: Minimum node overlap between consecutive edges (default 1).
            edge_types: If provided, only traverse edges of these types.
            direction_mode: ``"undirected"`` (default), ``"forward"``, or ``"backward"``.

        Returns:
            List of paths, where each path is a list of Edge objects.
        """
        core_paths = self._store.find_paths(
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            max_hops=max_hops,
            max_paths=max_paths,
            min_intersection=min_intersection,
            edge_types=edge_types,
            direction_mode=direction_mode,
        )
        return [[_core_edge_to_model(e) for e in path] for path in core_paths]

    def node_degree(self, node_id: str, *, edge_types: list[str] | None = None) -> int:
        """Count how many edges touch a node.

        Args:
            node_id: The node to measure.
            edge_types: If provided, only count edges of these types.

        Returns:
            The degree (edge count) of the node.
        """
        return self._store.node_degree(node_id, edge_types=edge_types)

    def top_nodes_by_degree(self, k: int = 20) -> list[tuple[Node, int]]:
        """Return the top-k most connected nodes.

        Args:
            k: Number of top nodes to return (default 20).

        Returns:
            List of (Node, degree) tuples, most connected first.
        """
        return [(_core_node_to_model(cn), degree) for cn, degree in self._store.top_nodes_by_degree(k)]

    def edge_cardinality(self, edge_id: str) -> int:
        """Count how many distinct nodes an edge contains.

        Args:
            edge_id: The edge to measure.

        Returns:
            Count of distinct node IDs in the edge.
        """
        return self._store.edge_cardinality(edge_id)

    def hyperedge_degree(
        self,
        node_set: set[str],
        *,
        edge_type: str | None = None,
    ) -> int:
        """Add up the edge counts of every node in a set.

        Args:
            node_set: Set of node IDs to aggregate.
            edge_type: If provided, only count edges of this type.

        Returns:
            Sum of individual node degrees.
        """
        return self._store.hyperedge_degree(node_set, edge_type=edge_type)

    def validate(self) -> ValidationResult:
        """Check the hypergraph for internal consistency.

        Returns:
            A ``ValidationResult`` with ``valid``, ``errors``, and ``warnings`` fields.
        """
        result = self._store.validate()
        return ValidationResult(
            valid=result["valid"],
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
        )

    def to_hif(self) -> dict:
        """Export the graph to HIF (Hypergraph Interchange Format).

        Returns:
            A dict representing the hypergraph in HIF JSON structure.
        """
        return self._store.to_hif()

    @classmethod
    def from_hif(cls, hif_data: dict) -> Hypabase:
        """Build a new Hypabase instance from HIF (Hypergraph Interchange Format) data.

        Creates an in-memory instance populated from the HIF structure.

        Args:
            hif_data: A dict in HIF JSON format.

        Returns:
            A new Hypabase instance containing the imported data.
        """
        hb = cls()
        hb._stores[hb._current_ns] = HypergraphCore.from_hif(hif_data)
        return hb

    def upsert_edge_by_vertex_set(
        self,
        node_ids: set[str],
        edge_type: str,
        properties: dict[str, Any] | None = None,
        *,
        source: str | None = None,
        confidence: float | None = None,
        merge_fn: Any = None,
    ) -> Edge:
        """Create or update an edge matched by its exact set of nodes.

        Finds an existing edge with the same nodes, or creates a new one.
        Useful for idempotent ingestion.

        Args:
            node_ids: Set of node IDs for the edge.
            edge_type: Edge type string.
            properties: Key-value metadata. Merged on update.
            source: Provenance source. Falls back to context or ``"unknown"``.
            confidence: Confidence score 0.0-1.0. Falls back to context or ``1.0``.
            merge_fn: Optional callable ``(existing_props, new_props) -> merged_props``
                for custom property merging on update.

        Returns:
            The created or updated Edge.
        """
        resolved_source = source or self._context_source or "unknown"
        resolved_confidence = (
            confidence
            if confidence is not None
            else (self._context_confidence if self._context_confidence is not None else 1.0)
        )
        core_edge = self._store.upsert_edge_by_node_set(
            node_ids=node_ids,
            edge_type=edge_type,
            properties=properties or {},
            source=resolved_source,
            confidence=resolved_confidence,
            merge_fn=merge_fn,
        )
        if self._storage:
            self._storage.write_edge(self._current_ns, core_edge)
        return _core_edge_to_model(core_edge)

    def edges_of_node(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
    ) -> list[Edge]:
        """Get all edges incident to a node.

        Args:
            node_id: The node to query.
            edge_types: If provided, only return edges of these types.

        Returns:
            List of edges containing this node.
        """
        return [_core_edge_to_model(e) for e in self._store.get_edges_of_node(node_id, edge_types=edge_types)]

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Group write operations in an atomic transaction.

        On success the SQLite transaction commits and in-memory state is kept.
        On any failure, SQLite rolls back and in-memory state is reloaded
        from disk to restore consistency.

        In-memory-only instances (no storage) are unaffected — partial
        changes remain since there is no durable state to reload from.

        Batches can nest; only the outermost batch triggers commit/rollback.

        Example:
            ```python
            with hb.batch():
                for i in range(1000):
                    hb.edge([f"entity_{i}", "catalog"], type="belongs_to")
            # Single save at the end
            ```
        """
        with self._store.batch():
            if self._storage:
                self._storage.begin()
            try:
                yield
            except BaseException:
                if self._storage:
                    self._storage.rollback()
                    try:
                        self._reload_namespace()
                    except Exception:
                        logger.error("Failed to reload namespace after rollback", exc_info=True)
                raise
            else:
                if self._storage:
                    try:
                        self._storage.commit()
                    except BaseException:
                        self._storage.rollback()
                        self._reload_namespace()
                        raise

    def _reload_namespace(self) -> None:
        """Reload the current namespace from SQLite after a failed batch."""
        if self._storage:
            try:
                self._stores[self._current_ns] = self._storage.load_namespace(self._current_ns)
            except Exception as reload_err:
                import warnings

                warnings.warn(
                    f"Failed to reload namespace '{self._current_ns}' after rollback: "
                    f"{reload_err!r}. In-memory state may be inconsistent. "
                    f"Close and reopen the database to restore consistency.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def rebuild_search_index(self) -> int:
        """Rebuild the vector search index from stored embeddings.

        Use this if search results seem incorrect or after recovering
        from a storage error. No-op for in-memory instances.

        Returns:
            Number of embeddings re-indexed.
        """
        if self._storage is None:
            return 0
        rebuild = getattr(self._storage, "rebuild_vec_index", None)
        if rebuild is None:
            return 0
        result: int = rebuild()
        return result

    # --- Vector search ---

    def embed_node(self, node_id: str, text: str | None = None) -> bool:
        """Generate and store an embedding for a node.

        Args:
            node_id: The node to embed.
            text: Text to embed. Defaults to the node ID.

        Returns:
            True if embedding was stored, False if embedder is not configured or node not found.
        """
        if self._embedder is None or self._storage is None:
            return False
        cn = self._store.get_node(node_id)
        if cn is None:
            return False
        embed_text = text or node_id
        vec = self._embedder.embed(embed_text)
        from hypabase.engine.vector import pack_embedding

        blob = pack_embedding(vec)
        emb_id = f"node:{self._current_ns}:{node_id}"
        self._storage.save_embedding(
            id=emb_id,
            namespace=self._current_ns,
            kind="node",
            ref_id=node_id,
            text=embed_text,
            embedding=blob,
            dimension=self._embedder.dimension,
            model=getattr(self._embedder, "_model_name", None) or str(type(self._embedder).__name__),
        )
        return True

    def embed_edge(self, edge_id: str, text: str | None = None) -> bool:
        """Generate and store an embedding for an edge.

        Args:
            edge_id: The edge to embed.
            text: Text to embed. Defaults to a string of the edge's node IDs joined with spaces.

        Returns:
            True if embedding was stored, False if embedder is not configured or edge not found.
        """
        if self._embedder is None or self._storage is None:
            return False
        ce = self._store.get_edge(edge_id)
        if ce is None:
            return False
        if text is None:
            text = " ".join(ce.nodes)
        if not text:
            return False
        vec = self._embedder.embed(text)
        from hypabase.engine.vector import pack_embedding

        blob = pack_embedding(vec)
        emb_id = f"edge:{self._current_ns}:{edge_id}"
        self._storage.save_embedding(
            id=emb_id,
            namespace=self._current_ns,
            kind="edge",
            ref_id=edge_id,
            text=text,
            embedding=blob,
            dimension=self._embedder.dimension,
            model=getattr(self._embedder, "_model_name", None) or str(type(self._embedder).__name__),
        )
        return True

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        type: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Semantic search over embedded nodes and edges.

        Args:
            query: The search query text.
            limit: Maximum results to return.
            kind: Filter by kind ("node" or "edge"). None returns both.
            type: Filter by node/edge type.
            min_score: Minimum cosine similarity score.

        Returns:
            List of dicts with keys: kind, ref_id, text, score, and optionally the full object.
        """
        if self._embedder is None or self._storage is None:
            return []

        from hypabase.engine.vector import pack_embedding

        query_vec = self._embedder.embed(query)
        query_blob = pack_embedding(query_vec)

        if self._storage.vec_dimension is None:
            # No embeddings stored yet
            return []

        # When kind and type are both specified, filter at the SQL level
        sql_type_filter = type if kind in ("node", "edge") and type is not None else None
        raw_results = self._storage.search_vec(
            self._current_ns,
            query_blob,
            limit=limit * 2,
            kind=kind,
            type_filter=sql_type_filter,
        )
        results = []
        for r in raw_results:
            score = r["score"]
            if score < min_score:
                continue
            result: dict[str, Any] = {
                "kind": r["kind"],
                "ref_id": r["ref_id"],
                "text": r["text"],
                "score": round(score, 6),
            }
            if r["kind"] == "node":
                cn = self._store.get_node(r["ref_id"])
                if cn is not None:
                    if type is not None and sql_type_filter is None and cn.type != type:
                        continue
                    result["node"] = _core_node_to_model(cn)
            elif r["kind"] == "edge":
                ce = self._store.get_edge(r["ref_id"])
                if ce is not None:
                    if type is not None and sql_type_filter is None and ce.type != type:
                        continue
                    result["edge"] = _core_edge_to_model(ce)
            results.append(result)
        return results[:limit]

    # --- Stats ---

    def stats(self) -> HypergraphStats:
        """Get node and edge counts by type.

        Returns:
            A ``HypergraphStats`` with ``node_count``, ``edge_count``,
            ``nodes_by_type``, and ``edges_by_type`` fields.
        """
        s = self._store.stats()
        return HypergraphStats(
            node_count=s["num_nodes"],
            edge_count=s["num_edges"],
            nodes_by_type=s.get("nodes_by_type", {}),
            edges_by_type=s.get("edges_by_type", {}),
        )
