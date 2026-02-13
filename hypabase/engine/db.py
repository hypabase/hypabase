"""Redis-like hypergraph database with namespacing.

Provides namespace isolation for managing multiple hypergraphs
within a single HypergraphDB instance.

Thread Safety:
    This module is thread-safe. All operations on HypergraphDB are protected by
    internal locks, allowing safe concurrent access from multiple threads without
    external synchronization.

    For concurrent access to different namespaces, use the namespace() method
    to get direct store references:

        store = db.namespace("financial/entities")
        store.add_node(node)  # Thread-safe

    The legacy select() + store pattern also works but is less efficient for
    concurrent scenarios.

Example:
    >>> from benchmarks.bird_v2.hypergraph import HypergraphDB, Node
    >>>
    >>> db = HypergraphDB()
    >>>
    >>> # Recommended: Direct namespace access (thread-safe)
    >>> store = db.namespace("financial/entities")
    >>> store.add_node(Node("customers", "table", {}))
    >>>
    >>> # Legacy: select() + store (still works)
    >>> db.select("financial/themes")
    >>> db.store.add_node(Node("revenue", "concept", {}))
    >>>
    >>> # Namespaces are isolated
    >>> db.select("financial/entities")
    >>> assert db.store.get_node("revenue") is None
"""

import copy
import threading
from typing import Any, Literal

from .core import HypergraphCore
from .persistence import load_db, save_db


class HypergraphDB:
    """Redis-like hypergraph database with namespacing.

    Provides namespace-based isolation for managing multiple hypergraphs.
    Each namespace is an independent HypergraphCore.

    Namespaces support hierarchical naming (e.g., "financial/entities")
    and are auto-created on first access (Redis-like behavior).
    """

    def __init__(self, default_namespace: str = "default"):
        """Initialize the database.

        Args:
            default_namespace: Initial namespace to use
        """
        self._namespaces: dict[str, HypergraphCore] = {}
        self._current_namespace: str = default_namespace
        # Auto-create the default namespace
        self._namespaces[default_namespace] = HypergraphCore()
        # DB-level lock for namespace operations
        self._db_lock = threading.RLock()

    @property
    def store(self) -> HypergraphCore:
        """Get the current namespace's HypergraphCore."""
        with self._db_lock:
            return self._namespaces[self._current_namespace]

    @property
    def current_namespace(self) -> str:
        """Get the current namespace name."""
        with self._db_lock:
            return self._current_namespace

    def namespace(self, name: str) -> HypergraphCore:
        """Get a namespace's store directly without changing current state.

        This is the recommended API for concurrent access. Each thread can
        work with different namespaces without interfering with each other.

        Args:
            name: Namespace name (supports hierarchical names like "a/b/c")

        Returns:
            The HypergraphCore for the namespace (created if needed)
        """
        with self._db_lock:
            if name not in self._namespaces:
                self._namespaces[name] = HypergraphCore()
            return self._namespaces[name]

    def select(self, namespace: str) -> "HypergraphDB":
        """Switch to a namespace, creating it if needed.

        Note: For concurrent access, prefer namespace() which doesn't mutate
        the current namespace state.

        Args:
            namespace: Namespace name (supports hierarchical names like "a/b/c")

        Returns:
            Self for method chaining
        """
        with self._db_lock:
            if namespace not in self._namespaces:
                self._namespaces[namespace] = HypergraphCore()
            self._current_namespace = namespace
            return self

    def list_namespaces(self) -> list[str]:
        """List all namespace names."""
        with self._db_lock:
            return sorted(self._namespaces.keys())

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists."""
        with self._db_lock:
            return namespace in self._namespaces

    def delete_namespace(self, namespace: str) -> bool:
        """Delete a namespace and its data.

        Args:
            namespace: Namespace to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If trying to delete the current namespace
        """
        with self._db_lock:
            if namespace == self._current_namespace:
                raise ValueError(f"Cannot delete current namespace: {namespace}")
            if namespace not in self._namespaces:
                return False
            del self._namespaces[namespace]
            return True

    def rename_namespace(self, old_name: str, new_name: str) -> bool:
        """Rename a namespace.

        Args:
            old_name: Current namespace name
            new_name: New namespace name

        Returns:
            True if renamed, False if old_name not found

        Raises:
            ValueError: If new_name already exists
        """
        with self._db_lock:
            if old_name not in self._namespaces:
                return False
            if new_name in self._namespaces:
                raise ValueError(f"Namespace already exists: {new_name}")

            self._namespaces[new_name] = self._namespaces.pop(old_name)

            # Update current namespace if it was renamed
            if self._current_namespace == old_name:
                self._current_namespace = new_name

            return True

    def copy_namespace(self, source: str, target: str) -> "HypergraphDB":
        """Copy a namespace to a new name.

        Args:
            source: Source namespace name
            target: Target namespace name

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source doesn't exist or target already exists
        """
        with self._db_lock:
            if source not in self._namespaces:
                raise ValueError(f"Source namespace not found: {source}")
            if target in self._namespaces:
                raise ValueError(f"Target namespace already exists: {target}")

            # Deep copy using copy.deepcopy for better performance
            self._namespaces[target] = copy.deepcopy(self._namespaces[source])

            return self

    def get_namespace(self, namespace: str) -> HypergraphCore | None:
        """Get a specific namespace's store without switching to it.

        Args:
            namespace: Namespace name

        Returns:
            HypergraphCore or None if not found
        """
        with self._db_lock:
            return self._namespaces.get(namespace)

    def stats(self) -> dict[str, Any]:
        """Get aggregated statistics across all namespaces."""
        with self._db_lock:
            total_nodes = 0
            total_edges = 0
            namespace_stats: dict[str, dict[str, int]] = {}

            for name, store in self._namespaces.items():
                store_stats = store.stats()
                total_nodes += store_stats["num_nodes"]
                total_edges += store_stats["num_edges"]
                namespace_stats[name] = {
                    "num_nodes": store_stats["num_nodes"],
                    "num_edges": store_stats["num_edges"],
                }

            return {
                "num_namespaces": len(self._namespaces),
                "current_namespace": self._current_namespace,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "namespaces": namespace_stats,
            }

    def clear_namespace(self, namespace: str | None = None) -> bool:
        """Clear all data from a namespace.

        Args:
            namespace: Namespace to clear (defaults to current)

        Returns:
            True if cleared, False if namespace not found
        """
        with self._db_lock:
            ns = namespace or self._current_namespace
            if ns not in self._namespaces:
                return False
            self._namespaces[ns] = HypergraphCore()
            return True

    # ========== Persistence ==========

    def save(self, path: str, format: Literal["json", "hif"] = "json") -> None:
        """Save all namespaces to a directory.

        Args:
            path: Output directory path
            format: "json" for internal dict format, "hif" for HIF standard
        """
        with self._db_lock:
            save_db(self._namespaces, path, format)

    @classmethod
    def load(
        cls,
        path: str,
        format: Literal["json", "hif"] | None = None,
    ) -> "HypergraphDB":
        """Load a database from a directory.

        Args:
            path: Directory path containing manifest.json
            format: Override format from manifest (optional)

        Returns:
            Loaded HypergraphDB instance
        """
        namespaces = load_db(path, format)

        db = cls.__new__(cls)
        db._namespaces = namespaces
        db._db_lock = threading.RLock()

        # Set current namespace to default or first available
        if "default" in namespaces:
            db._current_namespace = "default"
        elif namespaces:
            db._current_namespace = sorted(namespaces.keys())[0]
        else:
            # Empty DB - create default
            db._current_namespace = "default"
            db._namespaces["default"] = HypergraphCore()

        return db
