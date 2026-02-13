"""Hypabase client — the primary interface for interacting with a hypergraph."""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from hypabase.engine.storage import SQLiteStorage
from hypabase.models import Edge, HypergraphStats, Incidence, Node, ValidationResult

# --- Conversion helpers: engine core types <-> pydantic models ---


def _core_node_to_model(cn: CoreNode) -> Node:
    return Node(id=cn.id, type=cn.type, properties=cn.properties)


def _core_edge_to_model(ce: CoreEdge) -> Edge:
    return Edge(
        id=ce.id,
        type=ce.type,
        incidences=[
            Incidence(
                node_id=inc.node_id,
                edge_ref_id=inc.edge_ref_id,
                direction=inc.direction,
                properties=inc.properties,
            )
            for inc in ce.incidences
        ],
        directed=ce.is_directed,
        source=ce.source,
        confidence=ce.confidence,
        properties=ce.properties,
    )


class Hypabase:
    """A hypergraph database client.

    The primary interface for creating, querying, and traversing hypergraphs.
    Supports in-memory and local SQLite backends.

    Constructor patterns:
        - ``Hypabase()`` — in-memory, ephemeral (SQLite ``:memory:``)
        - ``Hypabase("file.db")`` — local persistent SQLite file
        - ``Hypabase("https://...")`` — cloud backend (Phase 3, raises NotImplementedError)

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
        # Private: shared state for namespace views
        _storage: SQLiteStorage | None = ...,  # type: ignore[assignment]
        _stores: dict[str, HypergraphCore] | None = None,
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
            # Ensure the namespace store exists
            if self._current_ns not in self._stores:
                self._stores[self._current_ns] = HypergraphCore()
        else:
            # Normal construction
            self._path = path_str
            self._current_ns = database
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
        self._batch_depth: int = 0

    @property
    def _store(self) -> HypergraphCore:
        """Resolve the current namespace's store."""
        return self._stores[self._current_ns]

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
        """Persist current state to SQLite.

        No-op for in-memory instances. Normally called automatically
        after each mutation; use this for explicit manual saves.
        """
        if self._storage:
            self._storage.save(self._stores)

    def _auto_save(self) -> None:
        """Persist to SQLite if file-backed and not inside a batch."""
        if self._storage and self._batch_depth == 0:
            self._storage.save_namespace(self._current_ns, self._store)

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
        existing = self._store.get_node(id)
        if existing:
            existing.type = type
            if properties:
                existing.properties.update(properties)
        else:
            self._store.add_node(CoreNode(id=id, type=type, properties=properties))
        core_node = self._store.get_node(id)
        assert core_node is not None, f"Node {id!r} should exist after add_node"
        self._auto_save()
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
            deleted, _ = self._store.delete_node_cascade(id)
            self._auto_save()
            return deleted
        result = self._store.delete_node(id)
        self._auto_save()
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
        result = self._store.delete_node_cascade(node_id)
        self._auto_save()
        return result

    # --- Edges ---

    def edge(
        self,
        nodes: list[str],
        *,
        type: str,
        directed: bool = False,
        source: str | None = None,
        confidence: float | None = None,
        properties: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> Edge:
        """Create a hyperedge linking two or more nodes in one relationship.

        Nodes are auto-created if they don't exist. Provenance values
        fall back to the active ``context()`` block if not set explicitly.

        Args:
            nodes: Node IDs to connect. Must contain at least 2.
            type: Edge type (e.g., ``"treatment"``, ``"concept_link"``).
            directed: If ``True``, first node is tail, last is head.
            source: Provenance source. Falls back to context or ``"unknown"``.
            confidence: Confidence score 0.0-1.0. Falls back to context or ``1.0``.
            properties: Arbitrary key-value metadata.
            id: Optional edge ID. Auto-generated UUID if omitted.

        Returns:
            The created Edge.

        Raises:
            ValueError: If fewer than 2 nodes or any node ID is empty.

        Example:
            ```python
            hb.edge(
                ["dr_smith", "patient_123", "aspirin"],
                type="treatment",
                source="clinical_records",
                confidence=0.95,
            )
            ```
        """
        if len(nodes) < 2:
            raise ValueError("A hyperedge must connect at least 2 nodes.")
        if any(not n for n in nodes):
            raise ValueError("Node IDs must be non-empty strings")

        edge_id = id or str(uuid.uuid4())
        resolved_source = source or self._context_source or "unknown"
        resolved_confidence = (
            confidence
            if confidence is not None
            else (self._context_confidence if self._context_confidence is not None else 1.0)
        )

        # Auto-create nodes
        for node_id in nodes:
            if not self._store.get_node(node_id):
                self._store.add_node(CoreNode(id=node_id, type="unknown"))

        # Build incidences — for directed edges, first node is tail, last is head
        if directed:
            incidences = [
                CoreIncidence(node_id=nodes[0], direction="tail"),
                *[CoreIncidence(node_id=n) for n in nodes[1:-1]],
                CoreIncidence(node_id=nodes[-1], direction="head"),
            ]
        else:
            incidences = [CoreIncidence(node_id=n) for n in nodes]

        core_edge = CoreEdge(
            id=edge_id,
            type=type,
            incidences=incidences,
            properties=properties or {},
            source=resolved_source,
            confidence=resolved_confidence,
        )

        # Use upsert to handle existing edge IDs
        self._store.upsert_edge(core_edge)
        stored_edge = self._store.get_edge(edge_id)
        assert stored_edge is not None, f"Edge {edge_id!r} should exist after upsert_edge"
        self._auto_save()
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
    ) -> list[Edge]:
        """Query edges by contained nodes, type, source, and/or confidence.

        All filters are combined with AND logic.

        Args:
            containing: Node IDs that must appear in the edge.
            type: Filter to edges of this type.
            match_all: If ``True``, edges must contain *all* nodes in
                ``containing``. If ``False`` (default), any match suffices.
            source: Filter to edges from this provenance source.
            min_confidence: Filter to edges with confidence >= this value.

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

        Uses the SHA-256 vertex-set hash index for constant-time lookup.
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
        self._auto_save()
        return result

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
        return [
            _core_node_to_model(n)
            for nid in neighbor_ids
            if (n := self._store.get_node(nid)) is not None
        ]

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
        self._auto_save()
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
        return [
            _core_edge_to_model(e)
            for e in self._store.get_edges_of_node(node_id, edge_types=edge_types)
        ]

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Group write operations and save them all at once.

        Reduces disk I/O for bulk inserts. Batches can nest; only the
        outermost batch triggers a save.

        Note:
            Provides batched persistence, **not** transaction rollback. If an
            exception occurs mid-batch, partial in-memory changes remain and
            are persisted when the batch exits.

        Example:
            ```python
            with hb.batch():
                for i in range(1000):
                    hb.edge([f"entity_{i}", "catalog"], type="belongs_to")
            # Single save at the end
            ```
        """
        with self._store.batch():
            self._batch_depth += 1
            try:
                yield
            finally:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._auto_save()

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
