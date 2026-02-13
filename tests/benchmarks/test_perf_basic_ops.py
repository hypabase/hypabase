"""Performance benchmarks for basic hypergraph operations."""

import time

from hypabase.engine import Hyperedge, Incidence, Node


class TestNodeOperationsPerformance:
    """Benchmarks for node operations."""

    def test_add_node_performance(self, graph_10k):
        """Adding a node should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            graph_10k.add_node(Node(f"new_node_{i}", "test"))
        elapsed = time.perf_counter() - start

        # 1000 adds should take < 100ms (0.1ms per add)
        assert elapsed < 0.1, f"Adding 1000 nodes took {elapsed:.3f}s"

    def test_get_node_performance(self, graph_10k):
        """Getting a node by ID should be O(1)."""
        # Warm up
        _ = graph_10k.get_node("node_5000")

        start = time.perf_counter()
        for i in range(10000):
            _ = graph_10k.get_node(f"node_{i % 10000}")
        elapsed = time.perf_counter() - start

        # 10K lookups should take < 50ms (5us per lookup)
        assert elapsed < 0.05, f"10K node lookups took {elapsed:.3f}s"

    def test_get_nodes_by_type_performance(self, graph_10k):
        """Getting nodes by type should be fast."""
        start = time.perf_counter()
        for _ in range(100):
            _ = graph_10k.get_nodes_by_type("table")
        elapsed = time.perf_counter() - start

        # 100 type lookups should take < 100ms
        assert elapsed < 0.1, f"100 type lookups took {elapsed:.3f}s"

    def test_node_degree_performance(self, graph_10k):
        """node_degree should be O(1) for raw count."""
        start = time.perf_counter()
        for i in range(10000):
            _ = graph_10k.node_degree(f"node_{i % 10000}")
        elapsed = time.perf_counter() - start

        # 10K degree checks should take < 100ms
        assert elapsed < 0.1, f"10K degree checks took {elapsed:.3f}s"


class TestEdgeOperationsPerformance:
    """Benchmarks for edge operations."""

    def test_add_edge_performance(self, graph_10k):
        """Adding an edge should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            graph_10k.add_edge(
                Hyperedge(
                    f"new_edge_{i}",
                    "test",
                    [Incidence(f"node_{i % 1000}"), Incidence(f"node_{(i + 1) % 1000}")],
                )
            )
        elapsed = time.perf_counter() - start

        # 1000 adds should take < 100ms
        assert elapsed < 0.1, f"Adding 1000 edges took {elapsed:.3f}s"

    def test_get_edge_performance(self, graph_10k):
        """Getting an edge by ID should be O(1)."""
        start = time.perf_counter()
        for i in range(10000):
            _ = graph_10k.get_edge(f"edge_{i % 50000}")
        elapsed = time.perf_counter() - start

        # 10K lookups should take < 50ms
        assert elapsed < 0.05, f"10K edge lookups took {elapsed:.3f}s"

    def test_get_edges_containing_any_performance(self, graph_10k):
        """Finding edges containing nodes (union) should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            _ = graph_10k.get_edges_containing({f"node_{i % 10000}"}, match_all=False)
        elapsed = time.perf_counter() - start

        # 1000 lookups should take < 500ms
        assert elapsed < 0.5, f"1K edge containment lookups took {elapsed:.3f}s"

    def test_get_edges_containing_all_performance(self, graph_10k):
        """Finding edges containing all nodes (intersection) should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            _ = graph_10k.get_edges_containing(
                {f"node_{i % 10000}", f"node_{(i + 1) % 10000}"},
                match_all=True,
            )
        elapsed = time.perf_counter() - start

        # 1000 lookups should take < 500ms
        assert elapsed < 0.5, f"1K intersection lookups took {elapsed:.3f}s"

    def test_edge_by_node_set_performance(self, graph_10k):
        """O(1) vertex-set lookup should be very fast."""
        # Get some actual node sets from edges
        edges = graph_10k.get_all_edges()[:1000]
        node_sets = [e.node_set for e in edges]

        start = time.perf_counter()
        for node_set in node_sets:
            _ = graph_10k.get_edge_by_node_set(node_set)
        elapsed = time.perf_counter() - start

        # 1000 O(1) lookups should take < 10ms
        assert elapsed < 0.01, f"1K vertex-set lookups took {elapsed:.3f}s"


class TestNeighborOperationsPerformance:
    """Benchmarks for neighbor operations."""

    def test_get_neighbor_nodes_performance(self, graph_10k):
        """Getting neighbors should be reasonably fast."""
        start = time.perf_counter()
        for i in range(1000):
            _ = graph_10k.get_neighbor_nodes(f"node_{i % 10000}")
        elapsed = time.perf_counter() - start

        # 1000 neighbor lookups should take < 500ms
        assert elapsed < 0.5, f"1K neighbor lookups took {elapsed:.3f}s"

    def test_get_neighbor_nodes_with_filter_performance(self, graph_10k):
        """Getting neighbors with type filter."""
        start = time.perf_counter()
        for i in range(1000):
            _ = graph_10k.get_neighbor_nodes(
                f"node_{i % 10000}",
                edge_types=["foreign_key"],
            )
        elapsed = time.perf_counter() - start

        # 1000 filtered lookups should take < 500ms
        assert elapsed < 0.5, f"1K filtered neighbor lookups took {elapsed:.3f}s"


class TestUpsertPerformance:
    """Benchmarks for upsert operations."""

    def test_upsert_node_performance(self, graph_10k):
        """Upserting nodes should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            # Half inserts, half updates
            graph_10k.upsert_node(
                Node(
                    f"node_{i % 500}" if i < 500 else f"upsert_new_{i}",
                    "test",
                    {"updated": True},
                )
            )
        elapsed = time.perf_counter() - start

        # 1000 upserts should take < 100ms
        assert elapsed < 0.1, f"1K node upserts took {elapsed:.3f}s"

    def test_upsert_edge_performance(self, graph_10k):
        """Upserting edges should be fast."""
        start = time.perf_counter()
        for i in range(1000):
            graph_10k.upsert_edge(
                Hyperedge(
                    f"edge_{i % 500}" if i < 500 else f"upsert_edge_{i}",
                    "test",
                    [Incidence(f"node_{i % 1000}"), Incidence(f"node_{(i + 1) % 1000}")],
                )
            )
        elapsed = time.perf_counter() - start

        # 1000 upserts should take < 200ms
        assert elapsed < 0.2, f"1K edge upserts took {elapsed:.3f}s"


class TestBasicOpsAt1ms:
    """Verify basic ops complete in <1ms on 10K graph."""

    def test_single_node_lookup_under_1ms(self, graph_10k):
        """Single node lookup should be < 1ms."""
        iterations = 100
        total = 0.0
        for i in range(iterations):
            start = time.perf_counter()
            _ = graph_10k.get_node(f"node_{i % 10000}")
            total += time.perf_counter() - start

        avg_ms = (total / iterations) * 1000
        assert avg_ms < 1.0, f"Average node lookup: {avg_ms:.3f}ms"

    def test_single_edge_lookup_under_1ms(self, graph_10k):
        """Single edge lookup should be < 1ms."""
        iterations = 100
        total = 0.0
        for i in range(iterations):
            start = time.perf_counter()
            _ = graph_10k.get_edge(f"edge_{i % 50000}")
            total += time.perf_counter() - start

        avg_ms = (total / iterations) * 1000
        assert avg_ms < 1.0, f"Average edge lookup: {avg_ms:.3f}ms"

    def test_single_degree_under_1ms(self, graph_10k):
        """Single degree check should be < 1ms."""
        iterations = 100
        total = 0.0
        for i in range(iterations):
            start = time.perf_counter()
            _ = graph_10k.node_degree(f"node_{i % 10000}")
            total += time.perf_counter() - start

        avg_ms = (total / iterations) * 1000
        assert avg_ms < 1.0, f"Average degree check: {avg_ms:.3f}ms"
