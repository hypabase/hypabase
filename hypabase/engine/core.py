"""Core hypergraph data structures and operations.

A minimal, HIF-compatible hypergraph implementation optimized for semantic layer use cases.
Domain-agnostic: no SQL, embeddings, or semantic-layer concepts in the core.
Supports metagraph patterns where edges can reference other edges via edge_ref_id.

Thread Safety:
    This module is thread-safe. All operations on HypergraphCore are protected by
    an internal RLock (reentrant lock), allowing safe concurrent access from multiple
    threads without external synchronization.

    For atomic batch operations, use the batch() context manager:
        with store.batch():
            store.add_node(node1)
            store.add_node(node2)
            store.add_edge(edge)

References:
- HIF: Hypergraph Interchange Format (Coll et al., 2025)
- Bretto: "Hypergraph Theory: An Introduction" (2013)
- Stewart & Buehler: Intersection-constrained path finding (2026)
"""

import threading
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """An entity in the hypergraph.

    Attributes:
        id: Unique identifier for the node
        type: Extensible type string (e.g., "table", "column", "concept")
        properties: Arbitrary key-value metadata

    Raises:
        TypeError: If id or type is not a string
    """

    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError(f"Node id must be a string, got: {type(self.id).__name__}")
        if not isinstance(self.type, str):
            raise TypeError(f"Node type must be a string, got: {type(self.type).__name__}")


@dataclass
class Incidence:
    """A node's or edge's participation in a hyperedge, with optional direction.

    Follows HIF standard where direction is specified per-incidence, not per-edge.
    This allows mixed directed/undirected relationships within a single hyperedge.

    An incidence references either a node (node_id) or another edge (edge_ref_id),
    but not both. This enables metagraph support where edges can connect to other edges.

    Direction values (from HIF/Bretto):
        - None: Undirected participation
        - "tail": Participant is a sender/source (e+ in Bretto notation)
        - "head": Participant is a receiver/target (e- in Bretto notation)

    Attributes:
        node_id: ID of the participating node (mutually exclusive with edge_ref_id)
        edge_ref_id: ID of a referenced edge (mutually exclusive with node_id)
        direction: Optional direction ("head", "tail", or None)
        properties: Arbitrary key-value metadata for this incidence

    Raises:
        TypeError: If node_id or edge_ref_id is not a string
        ValueError: If neither or both of node_id/edge_ref_id are set,
                    or if direction is not None, "head", or "tail"
    """

    node_id: str | None = None
    edge_ref_id: str | None = None
    direction: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id is None and self.edge_ref_id is None:
            raise ValueError("Incidence must have either node_id or edge_ref_id")
        if self.node_id is not None and self.edge_ref_id is not None:
            raise ValueError("Incidence cannot have both node_id and edge_ref_id")
        if self.node_id is not None and not isinstance(self.node_id, str):
            raise TypeError(
                f"Incidence node_id must be a string, got: {type(self.node_id).__name__}"
            )
        if self.edge_ref_id is not None and not isinstance(self.edge_ref_id, str):
            raise TypeError(
                f"Incidence edge_ref_id must be a string, got: {type(self.edge_ref_id).__name__}"
            )
        if self.direction is not None and self.direction not in ("head", "tail"):
            raise ValueError(
                f"Incidence direction must be None, 'head', or 'tail', got: {self.direction!r}"
            )


@dataclass
class Hyperedge:
    """An n-ary relationship between nodes and/or other edges.

    Uses incidence-based representation for HIF compatibility.
    Supports both directed and undirected hyperedges.

    Attributes:
        id: Unique identifier for the edge
        type: Extensible type string (e.g., "foreign_key", "concept_mapping")
        incidences: List of node or edge-ref participations with optional directions
        properties: Arbitrary key-value metadata
        source: Provenance - where this edge came from
        confidence: Quality score from 0.0 to 1.0

    Raises:
        TypeError: If id or type is not a string
        ValueError: If confidence is not between 0.0 and 1.0
    """

    id: str
    type: str
    incidences: list[Incidence]
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError(f"Hyperedge id must be a string, got: {type(self.id).__name__}")
        if not isinstance(self.type, str):
            raise TypeError(f"Hyperedge type must be a string, got: {type(self.type).__name__}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Hyperedge confidence must be between 0.0 and 1.0, got: {self.confidence}"
            )

    @property
    def nodes(self) -> list[str]:
        """All participating node IDs (for intersection calculations)."""
        return [inc.node_id for inc in self.incidences if inc.node_id is not None]

    @property
    def node_set(self) -> set[str]:
        """All participating node IDs as a set."""
        return {inc.node_id for inc in self.incidences if inc.node_id is not None}

    @property
    def edge_refs(self) -> list[str]:
        """All referenced edge IDs (for metagraph traversal)."""
        return [inc.edge_ref_id for inc in self.incidences if inc.edge_ref_id is not None]

    @property
    def head_nodes(self) -> list[str]:
        """Node IDs marked as head/receivers/targets."""
        return [
            inc.node_id
            for inc in self.incidences
            if inc.direction == "head" and inc.node_id is not None
        ]

    @property
    def tail_nodes(self) -> list[str]:
        """Node IDs marked as tail/senders/sources."""
        return [
            inc.node_id
            for inc in self.incidences
            if inc.direction == "tail" and inc.node_id is not None
        ]

    @property
    def is_directed(self) -> bool:
        """True if any incidence has a direction."""
        return any(inc.direction is not None for inc in self.incidences)


class HypergraphCore:
    """Hypergraph storage with indexed operations and path finding.

    Implements Stewart & Buehler's intersection-constrained path finding
    with support for directed traversal modes.

    Design principles:
    - HIF-compatible data model
    - Domain-agnostic (no SQL knowledge)
    - Fast indexed lookups
    - Provenance tracking
    - Metagraph support (edges can reference other edges via edge_ref_id)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Hyperedge] = {}
        # Indexes for fast lookup
        self._node_to_edges: dict[str, set[str]] = defaultdict(set)
        self._nodes_by_type: dict[str, set[str]] = defaultdict(set)
        self._edges_by_type: dict[str, set[str]] = defaultdict(set)
        # Vertex-set index for O(1) Cog-RAG style lookup
        # Maps frozenset of node IDs -> set of edge IDs (multiple edges can share same node set)
        self._edges_by_node_set: dict[frozenset[str], set[str]] = defaultdict(set)
        # Metagraph index: maps referenced edge ID -> set of edge IDs that reference it
        self._edge_to_edges: dict[str, set[str]] = defaultdict(set)
        # Reentrant lock for thread safety (reentrant because delete_node_cascade
        # calls delete_node and delete_edge internally)
        self._lock = threading.RLock()

    def __getstate__(self) -> dict[str, Any]:
        """Support for pickle/deepcopy - exclude the lock."""
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support for pickle/deepcopy - recreate the lock."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __deepcopy__(self, memo: dict) -> "HypergraphCore":
        """Support for copy.deepcopy - create new instance with copied data.

        Thread-safe: acquires lock during copy to prevent concurrent modifications.
        """
        import copy

        with self._lock:
            new_store = HypergraphCore.__new__(HypergraphCore)
            memo[id(self)] = new_store

            # Deep copy all state except the lock
            new_store._nodes = copy.deepcopy(self._nodes, memo)
            new_store._edges = copy.deepcopy(self._edges, memo)
            new_store._node_to_edges = copy.deepcopy(self._node_to_edges, memo)
            new_store._nodes_by_type = copy.deepcopy(self._nodes_by_type, memo)
            new_store._edges_by_type = copy.deepcopy(self._edges_by_type, memo)
            new_store._edges_by_node_set = copy.deepcopy(self._edges_by_node_set, memo)
            new_store._edge_to_edges = copy.deepcopy(self._edge_to_edges, memo)

            # Create a new lock for the copy
            new_store._lock = threading.RLock()

            return new_store

    # ========== Thread Safety ==========

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Hold lock for multiple operations - provides isolation, NOT rollback.

        Use this context manager when performing multiple operations that should
        appear atomic to other threads (they see either none or all changes).

        WARNING: This does NOT provide transaction rollback. If an exception
        occurs mid-batch, partial changes will persist. The lock is always
        released properly regardless of exceptions.

        Example:
            with store.batch():
                store.add_node(node1)
                store.add_node(node2)  # If this fails, node1 is still added
                store.add_edge(edge)

        Yields:
            None
        """
        with self._lock:
            yield

    # ========== Node Operations ==========

    def add_node(self, node: Node) -> None:
        """Add a node to the hypergraph.

        If a node with the same ID already exists, it will be overwritten
        and indexes will be updated accordingly.
        """
        with self._lock:
            existing = self._nodes.get(node.id)
            if existing is not None:
                # Always clean up old type index for consistency with add_edge behavior
                self._nodes_by_type[existing.type].discard(node.id)
                if not self._nodes_by_type[existing.type]:
                    del self._nodes_by_type[existing.type]
            self._nodes[node.id] = node
            self._nodes_by_type[node.type].add(node.id)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID, or None if not found."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Get all nodes of a specific type."""
        with self._lock:
            return [self._nodes[nid] for nid in self._nodes_by_type.get(node_type, set())]

    def find_nodes(self, **properties: Any) -> list[Node]:
        """Find nodes matching all specified properties."""
        with self._lock:
            results = []
            for node in self._nodes.values():
                if all(node.properties.get(k) == v for k, v in properties.items()):
                    results.append(node)
            return results

    def delete_node(self, node_id: str) -> bool:
        """Delete a node. Returns True if deleted, False if not found.

        Warning:
            Does NOT automatically remove the node from edges. Use
            delete_node_cascade() to remove a node and all its incident edges,
            or call delete_edge() first for each incident edge.

        Args:
            node_id: The node ID to delete

        Returns:
            True if node was deleted, False if not found
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            node = self._nodes[node_id]
            self._nodes_by_type[node.type].discard(node_id)
            # Clean up empty type sets to prevent memory leaks
            if not self._nodes_by_type[node.type]:
                del self._nodes_by_type[node.type]
            del self._nodes[node_id]
            return True

    def delete_node_cascade(self, node_id: str) -> tuple[bool, int]:
        """Delete a node and all its incident edges.

        Safely removes a node along with all hyperedges that contain it,
        maintaining referential integrity.

        Args:
            node_id: The node ID to delete

        Returns:
            Tuple of (node_deleted, edges_deleted_count)
        """
        with self._lock:
            if node_id not in self._nodes:
                return (False, 0)

            # Get all incident edge IDs before deletion
            edge_ids = list(self._node_to_edges.get(node_id, set()))

            # Delete all incident edges (lock is reentrant, so nested calls work)
            edges_deleted = 0
            for edge_id in edge_ids:
                if self.delete_edge(edge_id):
                    edges_deleted += 1

            # Delete the node itself (lock is reentrant)
            node_deleted = self.delete_node(node_id)

            return (node_deleted, edges_deleted)

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the hypergraph."""
        with self._lock:
            return list(self._nodes.values())

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the hypergraph.

        O(1) lookup. Used in Cog-RAG merge operations.

        Args:
            node_id: The node ID to check

        Returns:
            True if node exists, False otherwise
        """
        with self._lock:
            return node_id in self._nodes

    # ========== Edge Operations ==========

    def add_edge(self, edge: Hyperedge) -> None:
        """Add a hyperedge to the hypergraph.

        If an edge with the same ID already exists, it will be overwritten
        and indexes will be updated accordingly.
        """
        with self._lock:
            existing = self._edges.get(edge.id)
            if existing is not None:
                # Clean up old type index
                if existing.type != edge.type:
                    self._edges_by_type[existing.type].discard(edge.id)
                    if not self._edges_by_type[existing.type]:
                        del self._edges_by_type[existing.type]
                # Clean up old node-to-edge indexes
                old_node_ids = {
                    inc.node_id for inc in existing.incidences if inc.node_id is not None
                }
                new_node_ids = {
                    inc.node_id for inc in edge.incidences if inc.node_id is not None
                }
                for node_id in old_node_ids - new_node_ids:
                    self._node_to_edges[node_id].discard(edge.id)
                    if not self._node_to_edges[node_id]:
                        del self._node_to_edges[node_id]
                # Clean up old edge-ref indexes
                old_edge_refs = {
                    inc.edge_ref_id
                    for inc in existing.incidences
                    if inc.edge_ref_id is not None
                }
                new_edge_refs = {
                    inc.edge_ref_id
                    for inc in edge.incidences
                    if inc.edge_ref_id is not None
                }
                for ref_id in old_edge_refs - new_edge_refs:
                    self._edge_to_edges[ref_id].discard(edge.id)
                    if not self._edge_to_edges[ref_id]:
                        del self._edge_to_edges[ref_id]
                # Clean up old vertex-set index
                old_node_set_key = frozenset(existing.node_set)
                new_node_set_key = frozenset(edge.node_set)
                if old_node_set_key != new_node_set_key and old_node_set_key:
                    self._edges_by_node_set[old_node_set_key].discard(edge.id)
                    if not self._edges_by_node_set[old_node_set_key]:
                        del self._edges_by_node_set[old_node_set_key]

            self._edges[edge.id] = edge
            self._edges_by_type[edge.type].add(edge.id)
            for inc in edge.incidences:
                if inc.node_id is not None:
                    self._node_to_edges[inc.node_id].add(edge.id)
                if inc.edge_ref_id is not None:
                    self._edge_to_edges[inc.edge_ref_id].add(edge.id)
            # Index by vertex set for O(1) lookup (multiple edges can share same node set)
            # Skip for edge-ref-only edges (empty node set would cause collisions)
            node_set_key = frozenset(edge.node_set)
            if node_set_key:
                self._edges_by_node_set[node_set_key].add(edge.id)

    def get_edge(self, edge_id: str) -> Hyperedge | None:
        """Get a hyperedge by ID, or None if not found."""
        with self._lock:
            return self._edges.get(edge_id)

    def get_edges_by_type(self, edge_type: str) -> list[Hyperedge]:
        """Get all hyperedges of a specific type."""
        with self._lock:
            return [self._edges[eid] for eid in self._edges_by_type.get(edge_type, set())]

    def get_edges_containing(
        self,
        node_ids: set[str],
        match_all: bool = False,
    ) -> list[Hyperedge]:
        """Find hyperedges containing the given nodes.

        Args:
            node_ids: Set of node IDs to search for
            match_all: If True, edge must contain ALL nodes (intersection)
                      If False, edge must contain ANY node (union)

        Returns:
            List of matching hyperedges
        """
        with self._lock:
            if not node_ids:
                return []

            if match_all:
                # Intersection: edge must contain all specified nodes
                edge_sets = [self._node_to_edges.get(nid, set()) for nid in node_ids]
                if not edge_sets:
                    return []
                common_edges = set.intersection(*edge_sets)
                return [self._edges[eid] for eid in common_edges]
            else:
                # Union: edge must contain any specified node
                all_edges: set[str] = set()
                for nid in node_ids:
                    all_edges.update(self._node_to_edges.get(nid, set()))
                return [self._edges[eid] for eid in all_edges]

    def find_edges(self, **properties: Any) -> list[Hyperedge]:
        """Find hyperedges matching all specified properties."""
        with self._lock:
            results = []
            for edge in self._edges.values():
                if all(edge.properties.get(k) == v for k, v in properties.items()):
                    results.append(edge)
            return results

    def delete_edge(self, edge_id: str) -> bool:
        """Delete a hyperedge. Returns True if deleted, False if not found."""
        with self._lock:
            if edge_id not in self._edges:
                return False
            edge = self._edges[edge_id]
            self._edges_by_type[edge.type].discard(edge_id)
            # Clean up empty type sets to prevent memory leaks
            if not self._edges_by_type[edge.type]:
                del self._edges_by_type[edge.type]
            for inc in edge.incidences:
                if inc.node_id is not None:
                    self._node_to_edges[inc.node_id].discard(edge_id)
                    # Clean up empty node-to-edge sets
                    if not self._node_to_edges[inc.node_id]:
                        del self._node_to_edges[inc.node_id]
                if inc.edge_ref_id is not None:
                    self._edge_to_edges[inc.edge_ref_id].discard(edge_id)
                    if not self._edge_to_edges[inc.edge_ref_id]:
                        del self._edge_to_edges[inc.edge_ref_id]
            # Remove from vertex-set index
            node_set_key = frozenset(edge.node_set)
            if node_set_key and node_set_key in self._edges_by_node_set:
                self._edges_by_node_set[node_set_key].discard(edge_id)
                if not self._edges_by_node_set[node_set_key]:
                    del self._edges_by_node_set[node_set_key]
            # Clean up this edge as a referenced target in _edge_to_edges
            if edge_id in self._edge_to_edges:
                del self._edge_to_edges[edge_id]
            del self._edges[edge_id]
            return True

    def get_all_edges(self) -> list[Hyperedge]:
        """Get all hyperedges in the hypergraph."""
        with self._lock:
            return list(self._edges.values())

    # ========== Utility Methods ==========

    def get_neighbor_nodes(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        exclude_self: bool = True,
    ) -> list[str]:
        """Get 1-hop neighbor nodes connected via hyperedges.

        Only traverses node-based incidences; edge-ref incidences are ignored.

        Args:
            node_id: The node to find neighbors for
            edge_types: Filter to specific edge types (optional)
            exclude_self: If True, exclude the source node from results

        Returns:
            List of neighbor node IDs
        """
        with self._lock:
            neighbors: set[str] = set()
            edge_ids = self._node_to_edges.get(node_id, set())

            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge is None:
                    continue
                if edge_types and edge.type not in edge_types:
                    continue
                neighbors.update(edge.node_set)

            if exclude_self:
                neighbors.discard(node_id)

            return list(neighbors)

    def get_edges_of_node(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[Hyperedge]:
        """Get all hyperedges containing a node.

        Used for Cog-RAG diffusion: given a vertex, retrieve its incident edges.

        Args:
            node_id: The node to find edges for
            edge_types: Filter to specific edge types (optional)

        Returns:
            List of hyperedges containing the node
        """
        with self._lock:
            edge_ids = self._node_to_edges.get(node_id, set())
            if not edge_ids:
                return []

            if edge_types is None:
                return [edge for eid in edge_ids if (edge := self._edges.get(eid)) is not None]

            return [
                edge
                for eid in edge_ids
                if (edge := self._edges.get(eid)) is not None and edge.type in edge_types
            ]

    def get_edge_node_tuples_of_node(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> set[frozenset[str]]:
        """Get vertex sets of all hyperedges containing a node.

        Returns frozensets of node IDs representing the vertex tuples
        of incident edges. Used for Cog-RAG neighbor diffusion.

        Example:
            If node "A" participates in edges {A,B,C} and {A,D}:
            Returns {frozenset({"A","B","C"}), frozenset({"A","D"})}

        Args:
            node_id: The node to find incident edge tuples for
            edge_types: Filter to specific edge types (optional)

        Returns:
            Set of frozensets, each representing an incident edge's vertex set
        """
        with self._lock:
            edge_ids = self._node_to_edges.get(node_id, set())
            if not edge_ids:
                return set()

            result: set[frozenset[str]] = set()
            for eid in edge_ids:
                edge = self._edges.get(eid)
                if edge is None:
                    continue
                if edge_types is not None and edge.type not in edge_types:
                    continue
                result.add(frozenset(edge.node_set))
            return result

    def node_degree(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> int:
        """Get the number of hyperedges containing a node.

        Args:
            node_id: The node to check
            edge_types: Filter to specific edge types (optional)

        Returns:
            Edge count for the node
        """
        with self._lock:
            edge_ids = self._node_to_edges.get(node_id, set())

            if edge_types is None:
                return len(edge_ids)

            count = 0
            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge and edge.type in edge_types:
                    count += 1
            return count

    def edge_cardinality(self, edge_id: str) -> int:
        """Get the number of unique nodes in a hyperedge (excludes edge-ref members).

        Args:
            edge_id: The edge to check

        Returns:
            Node count for the edge, or 0 if edge not found
        """
        with self._lock:
            edge = self._edges.get(edge_id)
            if edge is None:
                return 0
            return len(edge.node_set)

    def hyperedge_degree(
        self,
        node_ids: set[str],
        edge_type: str | None = None,
    ) -> int:
        """Get the degree of a hyperedge (sum of vertex degrees).

        Used for ranking edges by connectivity/importance in Cog-RAG retrieval.
        Higher degree = edge connects more well-connected nodes.

        Args:
            node_ids: Set of node IDs identifying the hyperedge
            edge_type: Filter to specific edge type (optional)

        Returns:
            Sum of degrees of all vertices in the edge, or 0 if edge not found
        """
        with self._lock:
            edge = self.get_edge_by_node_set(node_ids, edge_type)
            if edge is None:
                return 0
            return sum(self.node_degree(nid) for nid in edge.node_set)

    def get_edge_by_node_set(
        self,
        node_ids: set[str],
        edge_type: str | None = None,
    ) -> Hyperedge | None:
        """Get a hyperedge by its exact vertex set.

        Uses O(1) lookup via the vertex-set index. This is useful for
        Cog-RAG style operations where you need to find an edge
        connecting a specific set of nodes.

        Note: If multiple edges share the same vertex set, returns the first
        matching one. Use get_edges_by_node_set() to get all matches.

        Args:
            node_ids: Exact set of node IDs
            edge_type: Filter to specific edge type (optional)

        Returns:
            Matching hyperedge or None if not found
        """
        with self._lock:
            edge_ids = self._edges_by_node_set.get(frozenset(node_ids))
            if not edge_ids:
                return None

            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge is None:
                    continue
                if edge_type is not None and edge.type != edge_type:
                    continue
                return edge

            return None

    def get_edges_by_node_set(
        self,
        node_ids: set[str],
        edge_type: str | None = None,
    ) -> list[Hyperedge]:
        """Get all hyperedges with the exact vertex set.

        Uses O(1) lookup via the vertex-set index, then filters by type.

        Args:
            node_ids: Exact set of node IDs
            edge_type: Filter to specific edge type (optional)

        Returns:
            List of matching hyperedges (may be empty)
        """
        with self._lock:
            edge_ids = self._edges_by_node_set.get(frozenset(node_ids))
            if not edge_ids:
                return []

            results = []
            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge is None:
                    continue
                if edge_type is not None and edge.type != edge_type:
                    continue
                results.append(edge)

            return results

    def has_edge_with_nodes(
        self,
        node_ids: set[str],
        edge_type: str | None = None,
    ) -> bool:
        """Check if a hyperedge exists with the exact vertex set.

        Args:
            node_ids: Exact set of node IDs
            edge_type: Filter to specific edge type (optional)

        Returns:
            True if matching edge exists
        """
        with self._lock:
            return self.get_edge_by_node_set(node_ids, edge_type) is not None

    def upsert_node(
        self,
        node: Node,
        merge_properties: bool = True,
    ) -> Node:
        """Insert or update a node.

        Args:
            node: The node to upsert
            merge_properties: If True, merge properties with existing node;
                            if False, replace properties entirely

        Returns:
            The upserted node
        """
        with self._lock:
            existing = self._nodes.get(node.id)

            if existing is None:
                self.add_node(node)
                return node

            # Update type index if type changed
            if existing.type != node.type:
                self._nodes_by_type[existing.type].discard(node.id)
                # Clean up empty type sets
                if not self._nodes_by_type[existing.type]:
                    del self._nodes_by_type[existing.type]
                self._nodes_by_type[node.type].add(node.id)

            if merge_properties:
                merged_props = dict(existing.properties)
                merged_props.update(node.properties)
                updated_node = Node(
                    id=node.id,
                    type=node.type,
                    properties=merged_props,
                )
            else:
                updated_node = node

            self._nodes[node.id] = updated_node
            return updated_node

    def upsert_edge(
        self,
        edge: Hyperedge,
        merge_fn: Callable[[Hyperedge, Hyperedge], Hyperedge] | None = None,
    ) -> Hyperedge:
        """Insert or update a hyperedge.

        Args:
            edge: The edge to upsert
            merge_fn: Optional function (existing, new) -> merged to customize merge.
                     If None and edge exists, the new edge replaces the old.

        Returns:
            The upserted edge

        Note:
            If merge_fn raises an exception, the original edge and all indexes
            remain intact (exception-safe).
        """
        with self._lock:
            existing = self._edges.get(edge.id)

            if existing is None:
                self.add_edge(edge)
                return edge

            # Compute merge result FIRST (safe to fail here - nothing modified yet)
            if merge_fn is not None:
                final_edge = merge_fn(existing, edge)
            else:
                final_edge = edge

            # NOW remove old indexes (point of no return)
            self._edges_by_type[existing.type].discard(edge.id)
            # Clean up empty type sets
            if not self._edges_by_type[existing.type]:
                del self._edges_by_type[existing.type]
            for inc in existing.incidences:
                if inc.node_id is not None:
                    self._node_to_edges[inc.node_id].discard(edge.id)
                    if not self._node_to_edges[inc.node_id]:
                        del self._node_to_edges[inc.node_id]
                if inc.edge_ref_id is not None:
                    self._edge_to_edges[inc.edge_ref_id].discard(edge.id)
                    if not self._edge_to_edges[inc.edge_ref_id]:
                        del self._edge_to_edges[inc.edge_ref_id]
            # Remove from vertex-set index
            old_node_set_key = frozenset(existing.node_set)
            if old_node_set_key and old_node_set_key in self._edges_by_node_set:
                self._edges_by_node_set[old_node_set_key].discard(edge.id)
                if not self._edges_by_node_set[old_node_set_key]:
                    del self._edges_by_node_set[old_node_set_key]

            # Re-add with updated indexes
            self._edges[final_edge.id] = final_edge
            self._edges_by_type[final_edge.type].add(final_edge.id)
            for inc in final_edge.incidences:
                if inc.node_id is not None:
                    self._node_to_edges[inc.node_id].add(final_edge.id)
                if inc.edge_ref_id is not None:
                    self._edge_to_edges[inc.edge_ref_id].add(final_edge.id)
            final_node_set_key = frozenset(final_edge.node_set)
            if final_node_set_key:
                self._edges_by_node_set[final_node_set_key].add(final_edge.id)

            return final_edge

    def upsert_edge_by_node_set(
        self,
        node_ids: set[str],
        edge_type: str,
        properties: dict[str, Any],
        merge_fn: Callable[[Hyperedge, Hyperedge], Hyperedge] | None = None,
        source: str = "unknown",
        confidence: float = 1.0,
    ) -> Hyperedge:
        """Upsert a hyperedge identified by its vertex set and type.

        If an edge with the same vertex set and type exists, it is updated
        (optionally merged using merge_fn). Otherwise, a new edge is created
        with an auto-generated UUID.

        This is the Cog-RAG pattern where (vertex_set, type) uniquely identifies
        an edge, enabling merge-on-collision semantics during extraction.

        Args:
            node_ids: Set of node IDs forming the hyperedge
            edge_type: Type of the hyperedge
            properties: Properties to set on the edge
            merge_fn: Optional (existing, new) -> merged function.
                      If None, new properties replace existing.
            source: Provenance string
            confidence: Quality score 0.0-1.0

        Returns:
            The upserted hyperedge
        """
        with self._lock:
            existing = self.get_edge_by_node_set(node_ids, edge_type)
            incidences = [Incidence(node_id=n) for n in node_ids]

            new_edge = Hyperedge(
                id=existing.id if existing else str(uuid.uuid4()),
                type=edge_type,
                incidences=incidences,
                properties=properties,
                source=source,
                confidence=confidence,
            )

            if existing:
                return self.upsert_edge(new_edge, merge_fn)
            else:
                self.add_edge(new_edge)
                return new_edge

    # ========== Path Finding (Stewart & Buehler Style) ==========

    def find_paths(
        self,
        start_nodes: set[str],
        end_nodes: set[str],
        min_intersection: int = 1,
        max_hops: int = 4,
        max_paths: int = 10,
        edge_types: list[str] | None = None,
        direction_mode: str = "undirected",
    ) -> list[list[Hyperedge]]:
        """Find paths through hyperedges connecting start to end nodes.

        Implements Stewart & Buehler's intersection-constrained path finding.

        Args:
            start_nodes: Nodes to start traversal from
            end_nodes: Nodes to reach
            min_intersection: Minimum shared nodes between adjacent edges (IS parameter)
            max_hops: Maximum number of edges in path
            max_paths: Maximum number of paths to return
            edge_types: Filter to specific edge types (optional)
            direction_mode: How to handle directed edges
                - "undirected": Ignore direction, use all nodes for intersection
                - "forward": Traverse tail→head (edge1.head ∩ edge2.tail >= IS)
                - "backward": Traverse head→tail (edge1.tail ∩ edge2.head >= IS)

        Returns:
            List of paths, where each path is a list of hyperedges
        """
        if direction_mode not in ("undirected", "forward", "backward"):
            raise ValueError(
                f"direction_mode must be 'undirected', 'forward', or 'backward', "
                f"got: {direction_mode!r}"
            )

        with self._lock:
            # Get starting edges (those containing any start node)
            start_edges = self.get_edges_containing(start_nodes, match_all=False)
            if edge_types:
                start_edges = [e for e in start_edges if e.type in edge_types]

            # Get target edges (those containing any end node)
            target_edge_ids = {
                e.id
                for e in self.get_edges_containing(end_nodes, match_all=False)
                if edge_types is None or e.type in edge_types
            }

            # Early exit if no path is possible
            if not start_edges or not target_edge_ids:
                return []

            # BFS for paths (using deque for O(1) popleft)
            found_paths: list[list[Hyperedge]] = []
            queue: deque[tuple[Hyperedge, list[Hyperedge]]] = deque(
                (edge, [edge]) for edge in start_edges
            )
            visited: set[str] = {edge.id for edge in start_edges}

            while queue and len(found_paths) < max_paths:
                current_edge, path = queue.popleft()

                # Check if we've reached a target
                if current_edge.id in target_edge_ids:
                    found_paths.append(path)
                    continue

                # Don't extend beyond max_hops
                if len(path) >= max_hops:
                    continue

                # Find adjacent edges based on direction mode
                adjacent = self._find_adjacent_edges(
                    current_edge,
                    min_intersection,
                    direction_mode,
                    edge_types,
                )

                for next_edge in adjacent:
                    if next_edge.id not in visited:
                        visited.add(next_edge.id)
                        queue.append((next_edge, path + [next_edge]))

            return found_paths

    def _find_adjacent_edges(
        self,
        edge: Hyperedge,
        min_intersection: int,
        direction_mode: str,
        edge_types: list[str] | None,
    ) -> list[Hyperedge]:
        """Find edges adjacent to the given edge based on intersection constraint.

        Note: This method assumes the caller holds the lock.
        """
        # Determine which nodes to use for intersection based on direction mode
        if direction_mode == "undirected":
            source_nodes = edge.node_set
        elif direction_mode == "forward":
            source_nodes = set(edge.head_nodes) if edge.head_nodes else edge.node_set
        else:  # backward
            source_nodes = set(edge.tail_nodes) if edge.tail_nodes else edge.node_set

        # Get candidate edges (any edge sharing at least one node)
        candidates = self.get_edges_containing(source_nodes, match_all=False)
        if edge_types:
            candidates = [e for e in candidates if e.type in edge_types]

        # Filter by intersection constraint
        adjacent = []
        for candidate in candidates:
            if candidate.id == edge.id:
                continue

            # Determine target nodes based on direction mode
            if direction_mode == "undirected":
                target_nodes = candidate.node_set
            elif direction_mode == "forward":
                target_nodes = (
                    set(candidate.tail_nodes) if candidate.tail_nodes else candidate.node_set
                )
            else:  # backward
                target_nodes = (
                    set(candidate.head_nodes) if candidate.head_nodes else candidate.node_set
                )

            # Check intersection constraint
            intersection_size = len(source_nodes & target_nodes)
            if intersection_size >= min_intersection:
                adjacent.append(candidate)

        return adjacent

    # ========== Statistics & Validation ==========

    def stats(self) -> dict[str, Any]:
        """Get hypergraph statistics.

        Returns:
            Dict with num_nodes, num_edges, nodes_by_type, edges_by_type
        """
        with self._lock:
            return {
                "num_nodes": len(self._nodes),
                "num_edges": len(self._edges),
                "nodes_by_type": {t: len(ids) for t, ids in self._nodes_by_type.items()},
                "edges_by_type": {t: len(ids) for t, ids in self._edges_by_type.items()},
            }

    def validate(self) -> dict[str, Any]:
        """Validate hypergraph integrity and detect orphaned references.

        Checks for:
        - Edges referencing non-existent nodes (orphaned incidences)
        - Edges referencing non-existent edges via edge_ref_id
        - Node-to-edges index consistency
        - Edge-to-edges index consistency
        - Type index consistency

        Returns:
            Dict with 'valid' (bool), 'errors' (list of error descriptions),
            and 'orphaned_edges' (list of edge IDs with missing node or edge references)
        """
        with self._lock:
            errors: list[str] = []
            orphaned_edges: list[str] = []

            # Check for edges referencing non-existent nodes or edges
            for edge_id, edge in self._edges.items():
                missing_nodes = []
                missing_edge_refs = []
                for inc in edge.incidences:
                    if inc.node_id is not None and inc.node_id not in self._nodes:
                        missing_nodes.append(inc.node_id)
                    if inc.edge_ref_id is not None and inc.edge_ref_id not in self._edges:
                        missing_edge_refs.append(inc.edge_ref_id)
                if missing_nodes:
                    orphaned_edges.append(edge_id)
                    errors.append(
                        f"Edge '{edge_id}' references non-existent nodes: {missing_nodes}"
                    )
                if missing_edge_refs:
                    if edge_id not in orphaned_edges:
                        orphaned_edges.append(edge_id)
                    errors.append(
                        f"Edge '{edge_id}' references non-existent edges: {missing_edge_refs}"
                    )

            # Verify node-to-edges index consistency
            for node_id, edge_ids in self._node_to_edges.items():
                if node_id not in self._nodes:
                    errors.append(f"Node-to-edges index contains non-existent node: '{node_id}'")
                for edge_id in edge_ids:
                    if edge_id not in self._edges:
                        errors.append(
                            f"Node-to-edges index for '{node_id}' references "
                            f"non-existent edge: '{edge_id}'"
                        )

            # Verify edge-to-edges index consistency
            for ref_edge_id, referencing_edge_ids in self._edge_to_edges.items():
                if ref_edge_id not in self._edges:
                    errors.append(
                        f"Edge-to-edges index contains non-existent "
                        f"referenced edge: '{ref_edge_id}'"
                    )
                for edge_id in referencing_edge_ids:
                    if edge_id not in self._edges:
                        errors.append(
                            f"Edge-to-edges index for '{ref_edge_id}' references "
                            f"non-existent edge: '{edge_id}'"
                        )

            # Verify type indexes
            for node_type, node_ids in self._nodes_by_type.items():
                for node_id in node_ids:
                    if node_id not in self._nodes:
                        errors.append(
                            f"Nodes-by-type index for '{node_type}' contains "
                            f"non-existent node: '{node_id}'"
                        )

            for edge_type, edge_ids in self._edges_by_type.items():
                for edge_id in edge_ids:
                    if edge_id not in self._edges:
                        errors.append(
                            f"Edges-by-type index for '{edge_type}' contains "
                            f"non-existent edge: '{edge_id}'"
                        )

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "orphaned_edges": orphaned_edges,
            }

    # ========== Serialization ==========

    def to_dict(self) -> dict[str, Any]:
        """Export to simple dict (for debugging/internal use)."""
        with self._lock:
            return {
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.type,
                        "properties": n.properties,
                    }
                    for n in self._nodes.values()
                ],
                "edges": [
                    {
                        "id": e.id,
                        "type": e.type,
                        "incidences": [
                            {
                                **( {"node_id": inc.node_id} if inc.node_id is not None else {}),
                                **(
                                    {"edge_ref_id": inc.edge_ref_id}
                                    if inc.edge_ref_id is not None
                                    else {}
                                ),
                                "direction": inc.direction,
                                "properties": inc.properties,
                            }
                            for inc in e.incidences
                        ],
                        "properties": e.properties,
                        "source": e.source,
                        "confidence": e.confidence,
                    }
                    for e in self._edges.values()
                ],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypergraphCore":
        """Import from simple dict."""
        store = cls()
        for node_data in data.get("nodes", []):
            store.add_node(
                Node(
                    id=node_data["id"],
                    type=node_data["type"],
                    properties=node_data.get("properties", {}),
                )
            )
        for edge_data in data.get("edges", []):
            store.add_edge(
                Hyperedge(
                    id=edge_data["id"],
                    type=edge_data["type"],
                    incidences=[
                        Incidence(
                            node_id=inc.get("node_id"),
                            edge_ref_id=inc.get("edge_ref_id"),
                            direction=inc.get("direction"),
                            properties=inc.get("properties", {}),
                        )
                        for inc in edge_data["incidences"]
                    ],
                    properties=edge_data.get("properties", {}),
                    source=edge_data.get("source", "unknown"),
                    confidence=edge_data.get("confidence", 1.0),
                )
            )
        return store

    def to_hif(self) -> dict[str, Any]:
        """Export to HIF-compliant JSON structure.

        HIF spec: https://github.com/HIF-org/HIF-standard

        Key HIF requirements:
        - Incidences are a flat array at root level (not nested in edges)
        - Nodes use "node" field for identifier (not "id")
        - Edges use "edge" field for identifier (not "id")
        - "network-type" field indicates directed/undirected/asc
        """
        with self._lock:
            # Flatten all incidences into root-level array
            # Edge-ref incidences are not part of HIF standard; only export node-based ones
            hif_incidences: list[dict[str, Any]] = []
            skipped_edge_refs = 0
            for edge in self._edges.values():
                for inc in edge.incidences:
                    if inc.node_id is None:
                        skipped_edge_refs += 1
                        continue  # Skip edge-ref incidences (not representable in HIF)
                    hif_inc: dict[str, Any] = {
                        "node": inc.node_id,
                        "edge": edge.id,
                    }
                    if inc.direction:
                        hif_inc["direction"] = inc.direction
                    if inc.properties:
                        hif_inc["attrs"] = inc.properties
                    hif_incidences.append(hif_inc)

            # Build nodes array with HIF field naming
            hif_nodes: list[dict[str, Any]] = []
            for node in self._nodes.values():
                hif_node: dict[str, Any] = {"node": node.id}
                node_attrs = dict(node.properties)
                node_attrs["_type"] = node.type
                hif_node["attrs"] = node_attrs
                hif_nodes.append(hif_node)

            # Build edges array with HIF field naming
            hif_edges: list[dict[str, Any]] = []
            for edge in self._edges.values():
                hif_edge: dict[str, Any] = {"edge": edge.id}
                edge_attrs: dict[str, Any] = {"_type": edge.type}
                if edge.source != "unknown":
                    edge_attrs["_source"] = edge.source
                if edge.confidence != 1.0:
                    edge_attrs["_confidence"] = edge.confidence
                edge_attrs.update(edge.properties)
                hif_edge["attrs"] = edge_attrs
                hif_edges.append(hif_edge)

            # Determine network type based on edge directionality
            has_direction = any(edge.is_directed for edge in self._edges.values())
            network_type = "directed" if has_direction else "undirected"

            metadata: dict[str, Any] = {
                "generator": "hypabase",
                "version": "1.0",
            }
            if skipped_edge_refs > 0:
                metadata["_hypabase_edge_refs_omitted"] = skipped_edge_refs

            return {
                "network-type": network_type,
                "metadata": metadata,
                "incidences": hif_incidences,
                "nodes": hif_nodes,
                "edges": hif_edges,
            }

    @classmethod
    def from_hif(
        cls,
        data: dict[str, Any],
        strict: bool = False,
    ) -> "HypergraphCore":
        """Import from HIF-compliant JSON structure.

        HIF spec: https://github.com/HIF-org/HIF-standard

        Handles:
        - Root-level incidences array (HIF standard)
        - "node" and "edge" field names (HIF standard)
        - Auto-creates nodes/edges if only incidences provided
        - Supports integer IDs (converts to string)

        Args:
            data: HIF-formatted dictionary
            strict: If True, raise ValueError when auto-creating missing
                   nodes/edges (for data validation). Default False for
                   permissive HIF import.

        Returns:
            Loaded HypergraphCore

        Raises:
            ValueError: In strict mode, if nodes or edges need to be auto-created
        """
        store = cls()
        auto_created_nodes: list[str] = []
        auto_created_edges: list[str] = []

        # Collect edge data from edges array (may be incomplete without incidences)
        edge_data: dict[str, dict[str, Any]] = {}
        for hif_edge in data.get("edges", []):
            edge_id = str(hif_edge["edge"])
            attrs = dict(hif_edge.get("attrs", {}))
            edge_data[edge_id] = {
                "type": attrs.pop("_type", "unknown"),
                "source": attrs.pop("_source", "unknown"),
                "confidence": attrs.pop("_confidence", 1.0),
                "properties": attrs,
                "incidences": [],
            }

        # Import nodes - use "node" field (HIF standard)
        for hif_node in data.get("nodes", []):
            node_id = str(hif_node["node"])
            attrs = dict(hif_node.get("attrs", {}))
            node_type = attrs.pop("_type", "unknown")
            store.add_node(
                Node(
                    id=node_id,
                    type=node_type,
                    properties=attrs,
                )
            )

        # Process root-level incidences array (HIF standard)
        for hif_inc in data.get("incidences", []):
            edge_id = str(hif_inc["edge"])
            node_id = str(hif_inc["node"])

            # Auto-create node if not in nodes array
            if store.get_node(node_id) is None:
                auto_created_nodes.append(node_id)
                store.add_node(Node(id=node_id, type="unknown"))

            # Auto-create edge entry if not in edges array
            if edge_id not in edge_data:
                auto_created_edges.append(edge_id)
                edge_data[edge_id] = {
                    "type": "unknown",
                    "source": "unknown",
                    "confidence": 1.0,
                    "properties": {},
                    "incidences": [],
                }

            # Add incidence to edge
            edge_data[edge_id]["incidences"].append(
                Incidence(
                    node_id=node_id,
                    direction=hif_inc.get("direction"),
                    properties=hif_inc.get("attrs", {}),
                )
            )

        # Create edges with collected incidences
        for edge_id, edata in edge_data.items():
            store.add_edge(
                Hyperedge(
                    id=edge_id,
                    type=edata["type"],
                    incidences=edata["incidences"],
                    properties=edata["properties"],
                    source=edata["source"],
                    confidence=edata["confidence"],
                )
            )

        # In strict mode, raise if any auto-creation occurred
        if strict and (auto_created_nodes or auto_created_edges):
            errors = []
            if auto_created_nodes:
                errors.append(f"Auto-created {len(auto_created_nodes)} nodes: {auto_created_nodes}")
            if auto_created_edges:
                errors.append(f"Auto-created {len(auto_created_edges)} edges: {auto_created_edges}")
            raise ValueError(f"HIF import validation failed (strict mode): {'; '.join(errors)}")

        return store


# Backward compatibility alias
HypergraphStore = HypergraphCore
