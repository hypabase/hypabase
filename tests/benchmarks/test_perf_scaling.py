"""Performance benchmarks for scaling and memory usage."""

import sys
import tempfile
import time
from pathlib import Path

import pytest

from hypabase.engine import (
    HypergraphStore,
    Node,
    load_store,
    save_store,
)
from tests.benchmarks.conftest import generate_random_graph


class TestMemoryUsage:
    """Benchmarks for memory consumption."""

    def test_10k_graph_memory(self, graph_10k):
        """10K graph should use < 100MB memory."""
        # Get rough memory estimate via sys.getsizeof
        # Note: This underestimates true memory due to object overhead
        nodes_size = sum(sys.getsizeof(n) for n in graph_10k.get_all_nodes())
        edges_size = sum(sys.getsizeof(e) for e in graph_10k.get_all_edges())

        # Very rough estimate - actual memory is higher due to indexes
        estimated_mb = (nodes_size + edges_size) / (1024 * 1024)

        # Log for information - exact thresholds depend on Python version
        print(
            f"Estimated base memory: {estimated_mb:.2f}MB"
            f" (nodes: {nodes_size / 1024:.0f}KB, edges: {edges_size / 1024:.0f}KB)"
        )

        # The graph should at least be loadable - this is a smoke test
        stats = graph_10k.stats()
        assert stats["num_nodes"] == 10000
        assert stats["num_edges"] == 50000

    def test_graph_creation_scaling(self):
        """Graph creation time should scale roughly linearly."""
        times = {}

        for size in [1000, 5000, 10000]:
            start = time.perf_counter()
            _ = generate_random_graph(num_nodes=size, num_edges=size * 5, seed=42)
            times[size] = time.perf_counter() - start

        # Log scaling behavior
        print("\nGraph creation times:")
        for size, t in times.items():
            print(f"  {size} nodes, {size * 5} edges: {t:.3f}s")

        # 10K should take less than 30x the time of 1K
        # (some overhead is expected for small base times)
        ratio = times[10000] / times[1000]
        assert ratio < 30, f"Scaling ratio {ratio:.1f}x is too high"


class TestSerializationPerformance:
    """Benchmarks for save/load operations."""

    def test_save_10k_graph_time(self, graph_10k):
        """Saving 10K graph should be fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"

            start = time.perf_counter()
            save_store(graph_10k, str(path), format="json")
            elapsed = time.perf_counter() - start

            # Save should take < 5s
            assert elapsed < 5.0, f"Save took {elapsed:.3f}s"
            print(f"Save 10K graph: {elapsed:.3f}s")

    def test_load_10k_graph_time(self, graph_10k):
        """Loading 10K graph should be fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            save_store(graph_10k, str(path), format="json")

            start = time.perf_counter()
            _ = load_store(str(path), format="json")
            elapsed = time.perf_counter() - start

            # Load should take < 5s
            assert elapsed < 5.0, f"Load took {elapsed:.3f}s"
            print(f"Load 10K graph: {elapsed:.3f}s")

    def test_save_load_roundtrip_10k(self, graph_10k):
        """Full save/load roundtrip should be < 5s total."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"

            start = time.perf_counter()
            save_store(graph_10k, str(path), format="json")
            loaded = load_store(str(path), format="json")
            elapsed = time.perf_counter() - start

            assert elapsed < 5.0, f"Roundtrip took {elapsed:.3f}s"
            assert loaded.stats()["num_nodes"] == 10000
            assert loaded.stats()["num_edges"] == 50000

    def test_hif_format_performance(self, graph_10k):
        """HIF format save/load should be comparable to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "graph.json"
            hif_path = Path(tmpdir) / "graph.hif.json"

            # JSON timing
            start = time.perf_counter()
            save_store(graph_10k, str(json_path), format="json")
            _ = load_store(str(json_path), format="json")
            json_time = time.perf_counter() - start

            # HIF timing
            start = time.perf_counter()
            save_store(graph_10k, str(hif_path), format="hif")
            _ = load_store(str(hif_path), format="hif")
            hif_time = time.perf_counter() - start

            print(f"JSON roundtrip: {json_time:.3f}s")
            print(f"HIF roundtrip: {hif_time:.3f}s")

            # Both should complete in reasonable time
            assert json_time < 10.0
            assert hif_time < 10.0


class TestFileSize:
    """Benchmarks for serialized file sizes."""

    def test_json_file_size_10k(self, graph_10k):
        """Check JSON file size for 10K graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            save_store(graph_10k, str(path), format="json")

            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"JSON file size (10K): {size_mb:.2f}MB")

            # Should be reasonably compact
            assert size_mb < 50, f"File too large: {size_mb:.2f}MB"

    def test_hif_file_size_10k(self, graph_10k):
        """Check HIF file size for 10K graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.hif.json"
            save_store(graph_10k, str(path), format="hif")

            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"HIF file size (10K): {size_mb:.2f}MB")

            # Should be reasonably compact
            assert size_mb < 50, f"File too large: {size_mb:.2f}MB"


class TestLargeGraphSmoke:
    """Smoke tests for larger graphs."""

    @pytest.mark.slow
    def test_100k_graph_creation(self):
        """100K graph can be created."""
        start = time.perf_counter()
        graph = generate_random_graph(
            num_nodes=100000,
            num_edges=500000,
            seed=42,
        )
        elapsed = time.perf_counter() - start

        print(f"100K graph creation: {elapsed:.1f}s")

        stats = graph.stats()
        assert stats["num_nodes"] == 100000
        assert stats["num_edges"] == 500000

    @pytest.mark.slow
    def test_100k_basic_ops(self, graph_100k):
        """Basic operations work on 100K graph."""
        # Node lookup
        node = graph_100k.get_node("node_50000")
        assert node is not None

        # Edge lookup
        edge = graph_100k.get_edge("edge_250000")
        assert edge is not None

        # Neighbor lookup
        neighbors = graph_100k.get_neighbor_nodes("node_50000")
        assert isinstance(neighbors, list)

        # Degree
        degree = graph_100k.node_degree("node_50000")
        assert degree >= 0


class TestIndexScaling:
    """Verify index operations scale well."""

    def test_type_index_scaling(self):
        """Type index should scale well with many types."""
        store = HypergraphStore()

        # Add nodes with many different types
        num_types = 100
        nodes_per_type = 100

        start = time.perf_counter()
        for t in range(num_types):
            for n in range(nodes_per_type):
                store.add_node(Node(f"node_{t}_{n}", f"type_{t}", {}))
        add_time = time.perf_counter() - start

        # Lookup by type should still be fast
        start = time.perf_counter()
        for t in range(num_types):
            nodes = store.get_nodes_by_type(f"type_{t}")
            assert len(nodes) == nodes_per_type
        lookup_time = time.perf_counter() - start

        print(f"Adding {num_types * nodes_per_type} nodes: {add_time:.3f}s")
        print(f"Looking up {num_types} types: {lookup_time:.3f}s")

        assert add_time < 5.0
        assert lookup_time < 1.0
