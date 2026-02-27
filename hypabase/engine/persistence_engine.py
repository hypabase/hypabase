"""Abstract persistence backend for Hypabase.

Defines the contract for incremental write-through persistence.
HypergraphCore remains the in-memory engine for reads and traversal;
implementations of this class handle durable storage only.

See SQLiteStorage for the reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hypabase.engine.core import Hyperedge, HypergraphCore, Node


class PersistenceEngine(ABC):
    """Abstract persistence backend for Hypabase.

    Defines the contract for incremental write-through persistence.
    HypergraphCore remains the in-memory engine for reads and traversal;
    implementations of this class handle durable storage.

    Commit behavior: methods that write data (write_node, write_edge, etc.)
    auto-commit when called outside a transaction. Inside a begin/commit
    block, writes are buffered and committed together.

    To implement a custom backend, subclass this and implement all abstract
    methods. See SQLiteStorage for the reference implementation.
    """

    # -- Lifecycle --

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection/resources."""
        ...

    # -- Namespace management --

    @abstractmethod
    def load(self) -> dict[str, HypergraphCore]:
        """Load all namespaces from the backend.

        Returns:
            Dict mapping namespace name to its HypergraphCore instance.
        """
        ...

    @abstractmethod
    def load_namespace(self, namespace: str) -> HypergraphCore:
        """Load a single namespace from the backend.

        Args:
            namespace: The namespace to load.

        Returns:
            A HypergraphCore populated with the namespace's data.
        """
        ...

    @abstractmethod
    def list_namespaces(self) -> list[str]:
        """List all namespaces that have data in the backend.

        Returns:
            Sorted list of namespace names.
        """
        ...

    @abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Delete all data for a namespace.

        Args:
            namespace: The namespace to delete.
        """
        ...

    # -- Incremental graph writes --

    @abstractmethod
    def write_node(self, namespace: str, node: Node) -> None:
        """Persist a single node (insert or update).

        Auto-commits if called outside a transaction.

        Args:
            namespace: Target namespace.
            node: The node to persist.
        """
        ...

    @abstractmethod
    def write_edge(self, namespace: str, edge: Hyperedge) -> None:
        """Persist a single edge with all its incidences and vertex-set index.

        Replaces any existing incidences/index for this edge. Auto-commits
        if called outside a transaction.

        Args:
            namespace: Target namespace.
            edge: The edge to persist.
        """
        ...

    @abstractmethod
    def remove_node(self, namespace: str, node_id: str) -> None:
        """Delete a single node from the backend.

        Auto-commits if called outside a transaction. Idempotent (no error
        if the node doesn't exist).

        Args:
            namespace: Target namespace.
            node_id: The node ID to remove.
        """
        ...

    @abstractmethod
    def remove_edge(self, namespace: str, edge_id: str) -> None:
        """Delete a single edge and its incidences/index from the backend.

        Auto-commits if called outside a transaction. Idempotent (no error
        if the edge doesn't exist).

        Args:
            namespace: Target namespace.
            edge_id: The edge ID to remove.
        """
        ...

    @abstractmethod
    def update_edge(self, namespace: str, edge: Hyperedge) -> None:
        """Update scalar fields on an existing edge without rewriting incidences.

        Use this for expire, confidence update, etc. Auto-commits if called
        outside a transaction.

        Args:
            namespace: Target namespace.
            edge: The edge with updated scalar fields.
        """
        ...

    # -- Transactions --

    @abstractmethod
    def begin(self) -> None:
        """Begin a transaction (or increment nesting depth).

        Nested calls increment a depth counter; only the outermost
        begin acquires the actual backend transaction.
        """
        ...

    @abstractmethod
    def commit(self) -> None:
        """Commit the transaction (or decrement nesting depth).

        Only the outermost commit actually flushes to disk.
        """
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the transaction, discarding all uncommitted writes.

        Resets nesting depth to 0.
        """
        ...

    # -- Bulk export (backward compat) --

    @abstractmethod
    def save_namespace(self, namespace: str, store: HypergraphCore) -> None:
        """Full overwrite of a namespace (snapshot persistence).

        Deletes all existing data for the namespace and re-inserts
        everything from the HypergraphCore.

        Args:
            namespace: Target namespace.
            store: The in-memory store to persist.
        """
        ...

    @abstractmethod
    def save(self, stores: dict[str, HypergraphCore]) -> None:
        """Full overwrite of all namespaces.

        Args:
            stores: Dict mapping namespace name to HypergraphCore.
        """
        ...

    # -- Embeddings --

    @abstractmethod
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
        """Save or update an embedding.

        Args:
            id: Unique embedding ID.
            namespace: Target namespace.
            kind: Embedding kind (e.g., "node", "edge").
            ref_id: Reference ID (node or edge ID).
            text: The text that was embedded.
            embedding: The embedding vector as bytes.
            dimension: Vector dimension.
            model: Model name used for embedding.
        """
        ...

    @abstractmethod
    def load_embeddings(
        self,
        namespace: str,
        kind: str | None = None,
    ) -> list[dict]:
        """Load embeddings for a namespace, optionally filtered by kind.

        Args:
            namespace: Target namespace.
            kind: If provided, filter to this kind.

        Returns:
            List of embedding dicts.
        """
        ...

    @abstractmethod
    def delete_embeddings(
        self,
        namespace: str,
        kind: str | None = None,
        ref_id: str | None = None,
    ) -> int:
        """Delete embeddings. Returns number of rows deleted.

        Args:
            namespace: Target namespace.
            kind: If provided, filter to this kind.
            ref_id: If provided, filter to this ref_id.

        Returns:
            Number of embeddings deleted.
        """
        ...

    @abstractmethod
    def search_vec(
        self,
        namespace: str,
        query_embedding: bytes,
        *,
        limit: int = 10,
        kind: str | None = None,
        type_filter: str | None = None,
    ) -> list[dict]:
        """KNN search over embeddings.

        Args:
            namespace: Target namespace.
            query_embedding: Query vector as bytes.
            limit: Maximum results.
            kind: If provided, filter to this kind.
            type_filter: If provided with kind, filter by entity/edge type.

        Returns:
            List of result dicts with scores.
        """
        ...

    # -- Vec extension (concrete defaults) --

    @property
    def vec_dimension(self) -> int | None:
        """Current vector dimension, or None if not initialized."""
        return None

    # -- Access tracking --

    @abstractmethod
    def record_access(self, namespace: str, kind: str, ref_id: str) -> None:
        """Record an access event for an item.

        Args:
            namespace: Target namespace.
            kind: Item kind (e.g., "node", "edge").
            ref_id: Item ID.
        """
        ...

    @abstractmethod
    def record_access_batch(self, namespace: str, kind: str, ref_ids: list[str]) -> None:
        """Record access events for multiple items in one call.

        Args:
            namespace: Target namespace.
            kind: Item kind.
            ref_ids: List of item IDs.
        """
        ...

    @abstractmethod
    def get_access_stats(self, namespace: str, kind: str, ref_id: str) -> dict | None:
        """Get access stats for a specific item.

        Args:
            namespace: Target namespace.
            kind: Item kind.
            ref_id: Item ID.

        Returns:
            Dict with last_accessed and access_count, or None.
        """
        ...

    @abstractmethod
    def get_batch_access_stats(self, namespace: str, kind: str, ref_ids: list[str]) -> dict[str, dict]:
        """Get access stats for multiple items.

        Args:
            namespace: Target namespace.
            kind: Item kind.
            ref_ids: List of item IDs.

        Returns:
            Dict mapping ref_id to stats dict.
        """
        ...

    @abstractmethod
    def get_all_access_stats(self, namespace: str) -> list[dict]:
        """Get all access stats for a namespace.

        Args:
            namespace: Target namespace.

        Returns:
            List of stats dicts.
        """
        ...
