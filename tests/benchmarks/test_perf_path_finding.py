"""Performance benchmarks for path finding operations."""

import time


class TestPathFindingPerformance:
    """Benchmarks for path finding with different parameters."""

    def test_path_finding_max_hops_1(self, graph_10k):
        """Path finding with max_hops=1 (direct neighbors)."""
        start = time.perf_counter()
        for i in range(100):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 100) % 10000}"},
                max_hops=1,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # 100 searches with max_hops=1 should be fast
        assert elapsed < 1.0, f"100 path searches (hops=1) took {elapsed:.3f}s"

    def test_path_finding_max_hops_2(self, graph_10k):
        """Path finding with max_hops=2."""
        start = time.perf_counter()
        for i in range(50):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=2,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # 50 searches with max_hops=2
        assert elapsed < 2.0, f"50 path searches (hops=2) took {elapsed:.3f}s"

    def test_path_finding_max_hops_3(self, graph_10k):
        """Path finding with max_hops=3."""
        start = time.perf_counter()
        for i in range(20):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 1000) % 10000}"},
                max_hops=3,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # 20 searches with max_hops=3 should complete in <100ms avg
        assert elapsed < 2.0, f"20 path searches (hops=3) took {elapsed:.3f}s"

    def test_path_finding_with_edge_type_filter(self, graph_10k):
        """Path finding with edge type filter should be faster."""
        # Without filter
        start = time.perf_counter()
        for i in range(20):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
            )
        elapsed_no_filter = time.perf_counter() - start

        # With filter
        start = time.perf_counter()
        for i in range(20):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
                edge_types=["foreign_key"],
            )
        elapsed_with_filter = time.perf_counter() - start

        # Both should complete in reasonable time
        assert elapsed_no_filter < 3.0, f"No filter: {elapsed_no_filter:.3f}s"
        assert elapsed_with_filter < 3.0, f"With filter: {elapsed_with_filter:.3f}s"


class TestSparseVsDensePathFinding:
    """Compare path finding on sparse vs dense graphs."""

    def test_sparse_graph_path_finding(self, sparse_graph_10k):
        """Path finding on sparse graph."""
        start = time.perf_counter()
        for i in range(50):
            _ = sparse_graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # Sparse graphs should be faster
        assert elapsed < 2.0, f"Sparse graph path finding: {elapsed:.3f}s"

    def test_dense_graph_path_finding(self, dense_graph_10k):
        """Path finding on dense graph (more challenging)."""
        start = time.perf_counter()
        for i in range(20):
            _ = dense_graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=2,  # Limit hops on dense graph
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # Dense graphs take longer but should still complete
        assert elapsed < 5.0, f"Dense graph path finding: {elapsed:.3f}s"


class TestDirectionModePerformance:
    """Benchmarks for different direction modes."""

    def test_undirected_mode(self, graph_10k):
        """Undirected mode path finding."""
        start = time.perf_counter()
        for i in range(30):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
                direction_mode="undirected",
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Undirected mode: {elapsed:.3f}s"

    def test_forward_mode(self, graph_10k):
        """Forward mode path finding."""
        start = time.perf_counter()
        for i in range(30):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
                direction_mode="forward",
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Forward mode: {elapsed:.3f}s"

    def test_backward_mode(self, graph_10k):
        """Backward mode path finding."""
        start = time.perf_counter()
        for i in range(30):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
                direction_mode="backward",
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Backward mode: {elapsed:.3f}s"


class TestIntersectionConstraintPerformance:
    """Benchmarks for different intersection constraints."""

    def test_min_intersection_1(self, graph_10k):
        """Path finding with min_intersection=1 (default)."""
        start = time.perf_counter()
        for i in range(30):
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                min_intersection=1,
                max_hops=3,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"IS=1: {elapsed:.3f}s"

    def test_min_intersection_2(self, hyperedge_heavy_10k):
        """Path finding with min_intersection=2 on hyperedge-heavy graph."""
        start = time.perf_counter()
        for i in range(30):
            _ = hyperedge_heavy_10k.find_paths(
                start_nodes={f"node_{i % 10000}"},
                end_nodes={f"node_{(i + 500) % 10000}"},
                min_intersection=2,
                max_hops=3,
                max_paths=5,
            )
        elapsed = time.perf_counter() - start

        # Higher IS should reduce search space
        assert elapsed < 3.0, f"IS=2: {elapsed:.3f}s"


class TestPathFindingUnder100ms:
    """Verify path finding completes in <100ms with max_hops=3 on 10K."""

    def test_single_path_search_under_100ms(self, graph_10k):
        """Single path search should be < 100ms."""
        iterations = 10
        total = 0.0

        for i in range(iterations):
            start = time.perf_counter()
            _ = graph_10k.find_paths(
                start_nodes={f"node_{i * 1000 % 10000}"},
                end_nodes={f"node_{(i * 1000 + 500) % 10000}"},
                max_hops=3,
                max_paths=5,
            )
            total += time.perf_counter() - start

        avg_ms = (total / iterations) * 1000
        assert avg_ms < 100, f"Average path search: {avg_ms:.1f}ms"
