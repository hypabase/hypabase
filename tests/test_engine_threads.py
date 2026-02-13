"""Thread safety tests for HypergraphStore and HypergraphDB.

These tests verify that concurrent operations from multiple threads
do not cause data corruption or race conditions.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hypabase.engine import (
    Hyperedge,
    HypergraphDB,
    HypergraphStore,
    Incidence,
    Node,
)


class TestHypergraphStoreThreadSafety:
    """Thread safety tests for HypergraphStore."""

    def test_concurrent_add_node_no_corruption(self):
        """Multiple threads adding nodes concurrently should not corrupt data."""
        store = HypergraphStore()
        num_threads = 10
        nodes_per_thread = 100
        errors: list[Exception] = []

        def add_nodes(prefix: str):
            try:
                for i in range(nodes_per_thread):
                    store.add_node(Node(f"{prefix}_{i}", "test"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_nodes, args=(f"T{t}",)) for t in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert not errors, f"Errors during concurrent add: {errors}"

        # All nodes should be present
        expected_count = num_threads * nodes_per_thread
        actual_count = len(store.get_all_nodes())
        assert actual_count == expected_count, (
            f"Expected {expected_count} nodes, got {actual_count}"
        )

        # Verify each node exists
        for t in range(num_threads):
            for i in range(nodes_per_thread):
                node = store.get_node(f"T{t}_{i}")
                assert node is not None, f"Missing node T{t}_{i}"

    def test_concurrent_add_delete_no_crash(self):
        """Interleaved add and delete operations should not crash."""
        store = HypergraphStore()
        errors: list[Exception] = []

        # Pre-populate some nodes
        for i in range(100):
            store.add_node(Node(f"initial_{i}", "test"))

        def add_nodes():
            try:
                for i in range(200):
                    store.add_node(Node(f"add_{i}", "test"))
            except Exception as e:
                errors.append(e)

        def delete_nodes():
            try:
                for i in range(100):
                    store.delete_node(f"initial_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_nodes),
            threading.Thread(target=delete_nodes),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert not errors, f"Errors during concurrent operations: {errors}"

        # Validation should pass
        validation = store.validate()
        assert validation["valid"], f"Store validation failed: {validation['errors']}"

    def test_concurrent_read_write(self):
        """Reads should not see partial writes."""
        store = HypergraphStore()
        errors: list[Exception] = []
        inconsistencies: list[str] = []

        # Pre-populate
        for i in range(50):
            store.add_node(Node(f"node_{i}", "initial"))

        def writer():
            try:
                for i in range(100):
                    # Update node type
                    store.add_node(Node(f"node_{i % 50}", f"type_{i}"))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    # Read stats and verify internal consistency
                    stats = store.stats()
                    all_nodes = store.get_all_nodes()
                    if stats["num_nodes"] != len(all_nodes):
                        inconsistencies.append(
                            f"Stats ({stats['num_nodes']}) != actual ({len(all_nodes)})"
                        )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent read/write: {errors}"
        assert not inconsistencies, f"Inconsistencies detected: {inconsistencies}"

    def test_batch_atomicity(self):
        """Batch operations should be atomic."""
        store = HypergraphStore()
        errors: list[Exception] = []
        observed_states: list[int] = []

        def batch_writer():
            try:
                for i in range(50):
                    with store.batch():
                        store.add_node(Node(f"batch_{i}_a", "batch"))
                        store.add_node(Node(f"batch_{i}_b", "batch"))
                        store.add_node(Node(f"batch_{i}_c", "batch"))
            except Exception as e:
                errors.append(e)

        def observer():
            try:
                for _ in range(200):
                    # Count batch nodes - should always be multiple of 3
                    batch_nodes = store.get_nodes_by_type("batch")
                    count = len(batch_nodes)
                    observed_states.append(count)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=batch_writer),
            threading.Thread(target=observer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during batch operations: {errors}"

        # All observed states should be multiples of 3
        for state in observed_states:
            assert state % 3 == 0, f"Observed non-atomic state: {state} nodes"

    def test_upsert_edge_by_node_set_concurrent(self):
        """Concurrent upsert_edge_by_node_set with merge function should work."""
        store = HypergraphStore()
        errors: list[Exception] = []

        # Add nodes first
        for i in range(10):
            store.add_node(Node(f"n{i}", "test"))

        def merge_counts(existing: Hyperedge, new: Hyperedge) -> Hyperedge:
            """Merge by incrementing count."""
            count = existing.properties.get("count", 0) + new.properties.get("count", 0)
            return Hyperedge(
                id=existing.id,
                type=existing.type,
                incidences=existing.incidences,
                properties={"count": count},
            )

        def upsert_edges(thread_id: int):
            try:
                for _ in range(50):
                    store.upsert_edge_by_node_set(
                        node_ids={"n0", "n1"},
                        edge_type="counter",
                        properties={"count": 1},
                        merge_fn=merge_counts,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=upsert_edges, args=(t,)) for t in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent upsert: {errors}"

        # Verify the edge exists with correct count
        edge = store.get_edge_by_node_set({"n0", "n1"}, "counter")
        assert edge is not None
        assert edge.properties["count"] == 5 * 50  # 5 threads * 50 operations each

    def test_delete_node_cascade_concurrent(self):
        """Concurrent delete_node_cascade should not corrupt state."""
        store = HypergraphStore()
        errors: list[Exception] = []

        # Create a connected graph
        for i in range(20):
            store.add_node(Node(f"node_{i}", "test"))

        for i in range(19):
            store.add_edge(
                Hyperedge(
                    f"edge_{i}",
                    "link",
                    [Incidence(f"node_{i}"), Incidence(f"node_{i + 1}")],
                )
            )

        def delete_cascade(node_id: str):
            try:
                store.delete_node_cascade(node_id)
            except Exception as e:
                errors.append(e)

        # Try to delete multiple nodes concurrently
        threads = [
            threading.Thread(target=delete_cascade, args=(f"node_{i}",)) for i in range(0, 10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent cascade delete: {errors}"

        # Validation should pass
        validation = store.validate()
        assert validation["valid"], f"Store validation failed: {validation['errors']}"


class TestHypergraphDBThreadSafety:
    """Thread safety tests for HypergraphDB."""

    def test_namespace_method_returns_store_directly(self):
        """namespace() should return store without mutating current namespace."""
        db = HypergraphDB()

        # Get a namespace store directly
        store = db.namespace("test/namespace")

        # Current namespace should still be default
        assert db.current_namespace == "default"

        # The returned store should be usable
        store.add_node(Node("node1", "test"))
        assert store.get_node("node1") is not None

        # And should be the same store via select
        db.select("test/namespace")
        assert db.store.get_node("node1") is not None

    def test_concurrent_namespace_access(self):
        """Multiple threads accessing different namespaces should work."""
        db = HypergraphDB()
        errors: list[Exception] = []

        def work_in_namespace(ns_name: str, node_count: int):
            try:
                store = db.namespace(ns_name)
                for i in range(node_count):
                    store.add_node(Node(f"{ns_name}_node_{i}", "test"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=work_in_namespace, args=(f"ns_{t}", 100)) for t in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent namespace access: {errors}"

        # Verify all namespaces have correct node counts
        for t in range(10):
            store = db.namespace(f"ns_{t}")
            nodes = store.get_all_nodes()
            assert len(nodes) == 100, f"Namespace ns_{t} has {len(nodes)} nodes, expected 100"

    def test_concurrent_select_store_pattern(self):
        """Legacy select() + store pattern should not crash under concurrency."""
        db = HypergraphDB()
        errors: list[Exception] = []

        def use_select_pattern(thread_id: int):
            try:
                for i in range(50):
                    ns = f"thread_{thread_id}"
                    db.select(ns)
                    db.store.add_node(Node(f"node_{i}", "test"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=use_select_pattern, args=(t,)) for t in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # May have race conditions in which namespace gets nodes,
        # but should not crash
        assert not errors, f"Errors during select/store pattern: {errors}"

    def test_stats_consistency_during_modifications(self):
        """Stats should be internally consistent during modifications."""
        db = HypergraphDB()
        errors: list[Exception] = []
        inconsistencies: list[str] = []

        def modifier():
            try:
                for i in range(100):
                    store = db.namespace(f"ns_{i % 5}")
                    store.add_node(Node(f"node_{i}", "test"))
            except Exception as e:
                errors.append(e)

        def stats_checker():
            try:
                for _ in range(100):
                    stats = db.stats()
                    # Calculate sum of namespace nodes
                    calculated_total = sum(
                        ns_stats["num_nodes"] for ns_stats in stats["namespaces"].values()
                    )
                    if calculated_total != stats["total_nodes"]:
                        inconsistencies.append(
                            f"Sum ({calculated_total}) != total ({stats['total_nodes']})"
                        )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=modifier),
            threading.Thread(target=stats_checker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during stats check: {errors}"
        assert not inconsistencies, f"Inconsistencies detected: {inconsistencies}"


class TestAsyncCompatibility:
    """Tests for using HypergraphStore from async code."""

    @pytest.mark.asyncio
    async def test_store_operations_from_async(self):
        """Store operations should work when called from async via run_in_executor."""
        store = HypergraphStore()
        loop = asyncio.get_event_loop()

        async def add_node_async(node_id: str):
            await loop.run_in_executor(None, lambda: store.add_node(Node(node_id, "async_test")))

        # Run multiple async operations concurrently
        await asyncio.gather(*[add_node_async(f"async_node_{i}") for i in range(100)])

        # Verify all nodes were added
        nodes = store.get_all_nodes()
        assert len(nodes) == 100

    @pytest.mark.asyncio
    async def test_batch_from_async(self):
        """Batch operations should work from async."""
        store = HypergraphStore()
        loop = asyncio.get_event_loop()

        async def batch_add_async(prefix: str):
            def do_batch():
                with store.batch():
                    store.add_node(Node(f"{prefix}_a", "batch"))
                    store.add_node(Node(f"{prefix}_b", "batch"))
                    store.add_node(Node(f"{prefix}_c", "batch"))

            await loop.run_in_executor(None, do_batch)

        # Run multiple batch operations concurrently
        await asyncio.gather(*[batch_add_async(f"batch_{i}") for i in range(50)])

        # Verify all nodes were added (50 batches * 3 nodes each)
        nodes = store.get_nodes_by_type("batch")
        assert len(nodes) == 150


class TestDeepCopyThreadSafety:
    """Tests for __deepcopy__ under concurrent modification."""

    def test_deepcopy_during_concurrent_writes(self):
        """Deepcopy should produce consistent snapshot during concurrent writes."""
        import copy

        store = HypergraphStore()
        for i in range(100):
            store.add_node(Node(f"initial_{i}", "test"))

        errors: list[Exception] = []
        copies: list[HypergraphStore] = []
        copies_lock = threading.Lock()

        def writer():
            try:
                for i in range(200):
                    store.add_node(Node(f"new_{i}", "test"))
            except Exception as e:
                errors.append(e)

        def copier():
            try:
                for _ in range(10):
                    copied = copy.deepcopy(store)
                    with copies_lock:
                        copies.append(copied)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=copier),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent deepcopy: {errors}"
        # Each copy should be internally consistent
        for copied in copies:
            validation = copied.validate()
            assert validation["valid"], f"Copy validation failed: {validation['errors']}"

    def test_deepcopy_creates_independent_lock(self):
        """Copied store should have independent lock."""
        import copy

        store = HypergraphStore()
        store.add_node(Node("test", "test"))

        copied = copy.deepcopy(store)

        # Locks should be different objects
        assert store._lock is not copied._lock

        # Both should be usable from different threads
        results: list[bool] = []
        results_lock = threading.Lock()

        def use_store(s: HypergraphStore, node_id: str):
            s.add_node(Node(node_id, "test"))
            with results_lock:
                results.append(True)

        t1 = threading.Thread(target=use_store, args=(store, "s1"))
        t2 = threading.Thread(target=use_store, args=(copied, "c1"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        assert store.get_node("s1") is not None
        assert copied.get_node("c1") is not None


class TestBatchExceptionHandling:
    """Tests for batch() behavior during exceptions."""

    def test_batch_exception_releases_lock(self):
        """Lock should be released even when exception occurs in batch."""
        store = HypergraphStore()

        try:
            with store.batch():
                store.add_node(Node("a", "test"))
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Lock should be released - this should not deadlock
        store.add_node(Node("b", "test"))
        assert store.get_node("a") is not None  # Partial state persists
        assert store.get_node("b") is not None

    def test_batch_partial_state_persists_on_exception(self):
        """Partial changes should persist when exception occurs mid-batch."""
        store = HypergraphStore()

        try:
            with store.batch():
                store.add_node(Node("node1", "test"))
                store.add_node(Node("node2", "test"))
                raise RuntimeError("Failure after node2")
                store.add_node(Node("node3", "test"))  # Never reached
        except RuntimeError:
            pass

        # First two nodes should exist, third should not
        assert store.get_node("node1") is not None
        assert store.get_node("node2") is not None
        assert store.get_node("node3") is None


class TestUpsertEdgeExceptionSafety:
    """Tests for upsert_edge behavior when merge_fn raises."""

    def test_upsert_edge_merge_fn_exception_preserves_state(self):
        """If merge_fn raises, original edge and indexes should be intact."""
        store = HypergraphStore()
        store.add_node(Node("a", "test"))
        store.add_node(Node("b", "test"))

        original_edge = Hyperedge(
            "edge1",
            "link",
            [Incidence("a"), Incidence("b")],
            properties={"value": 1},
        )
        store.add_edge(original_edge)

        def failing_merge(existing: Hyperedge, new: Hyperedge) -> Hyperedge:
            raise ValueError("Merge failed")

        new_edge = Hyperedge(
            "edge1",
            "link",
            [Incidence("a"), Incidence("b")],
            properties={"value": 2},
        )

        with pytest.raises(ValueError, match="Merge failed"):
            store.upsert_edge(new_edge, merge_fn=failing_merge)

        # Original edge should still be intact with all indexes
        edge = store.get_edge("edge1")
        assert edge is not None
        assert edge.properties["value"] == 1

        # Indexes should be consistent
        assert store.get_edge_by_node_set({"a", "b"}) is not None
        assert len(store.get_edges_by_type("link")) == 1

        validation = store.validate()
        assert validation["valid"], f"Validation failed: {validation['errors']}"


class TestConcurrentNamespaceCreation:
    """Tests for concurrent namespace operations."""

    def test_concurrent_same_namespace_creation(self):
        """Multiple threads creating same namespace should get same store."""
        db = HypergraphDB()
        stores: list[HypergraphStore] = []
        lock = threading.Lock()

        def get_namespace():
            store = db.namespace("concurrent_test")
            with lock:
                stores.append(store)

        threads = [threading.Thread(target=get_namespace) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should reference the same store object
        assert all(s is stores[0] for s in stores)


class TestThreadPoolExecutor:
    """Tests using ThreadPoolExecutor for concurrent operations."""

    def test_thread_pool_node_operations(self):
        """Concurrent operations via ThreadPoolExecutor should work."""
        store = HypergraphStore()

        def add_node(node_id: str) -> bool:
            store.add_node(Node(node_id, "pool_test"))
            return True

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_node, f"pool_node_{i}") for i in range(500)]
            results = [f.result() for f in futures]

        assert all(results)
        assert len(store.get_all_nodes()) == 500

    def test_thread_pool_mixed_operations(self):
        """Mixed read/write operations via ThreadPoolExecutor should work."""
        store = HypergraphStore()

        # Pre-populate
        for i in range(100):
            store.add_node(Node(f"initial_{i}", "initial"))

        def read_op() -> int:
            return len(store.get_all_nodes())

        def write_op(node_id: str) -> bool:
            store.add_node(Node(node_id, "new"))
            return True

        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit mixed operations
            write_futures = [executor.submit(write_op, f"new_node_{i}") for i in range(100)]
            read_futures = [executor.submit(read_op) for _ in range(100)]

            # Wait for all
            write_results = [f.result() for f in write_futures]
            read_results = [f.result() for f in read_futures]

        assert all(write_results)
        # Read results should be between 100 and 200
        for count in read_results:
            assert 100 <= count <= 200

        # Final count should be 200
        assert len(store.get_all_nodes()) == 200
