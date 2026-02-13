"""Benchmark fixtures for hypergraph performance tests."""

import random

import pytest

from hypabase.engine import Hyperedge, HypergraphStore, Incidence, Node


def generate_random_graph(
    num_nodes: int,
    num_edges: int,
    avg_cardinality: float = 2.5,
    directed_ratio: float = 0.3,
    seed: int = 42,
) -> HypergraphStore:
    """Generate a random hypergraph for benchmarking.

    Args:
        num_nodes: Number of nodes to create
        num_edges: Number of hyperedges to create
        avg_cardinality: Average number of nodes per edge
        directed_ratio: Fraction of edges that are directed
        seed: Random seed for reproducibility

    Returns:
        HypergraphStore with random data
    """
    rng = random.Random(seed)
    store = HypergraphStore()

    # Create nodes
    node_types = ["table", "column", "concept", "metric"]
    node_ids = [f"node_{i}" for i in range(num_nodes)]

    for node_id in node_ids:
        store.add_node(
            Node(
                id=node_id,
                type=rng.choice(node_types),
                properties={"index": int(node_id.split("_")[1])},
            )
        )

    # Create edges
    edge_types = ["foreign_key", "concept_mapping", "synonym", "hierarchy"]

    for i in range(num_edges):
        # Determine cardinality (2 to avg*2, centered around avg)
        cardinality = max(2, int(rng.gauss(avg_cardinality, avg_cardinality / 2)))
        cardinality = min(cardinality, num_nodes)

        # Select random nodes
        selected_nodes = rng.sample(node_ids, cardinality)

        # Determine if directed
        is_directed = rng.random() < directed_ratio

        # Create incidences
        incidences = []
        for j, node_id in enumerate(selected_nodes):
            if is_directed:
                # First half are tails, second half are heads
                direction = "tail" if j < len(selected_nodes) // 2 else "head"
            else:
                direction = None
            incidences.append(Incidence(node_id, direction=direction))

        store.add_edge(
            Hyperedge(
                id=f"edge_{i}",
                type=rng.choice(edge_types),
                incidences=incidences,
                confidence=rng.random(),
            )
        )

    return store


@pytest.fixture
def graph_1k() -> HypergraphStore:
    """1K nodes, 5K edges - small benchmark graph."""
    return generate_random_graph(
        num_nodes=1000,
        num_edges=5000,
        avg_cardinality=2.5,
        seed=42,
    )


@pytest.fixture
def graph_10k() -> HypergraphStore:
    """10K nodes, 50K edges - medium benchmark graph."""
    return generate_random_graph(
        num_nodes=10000,
        num_edges=50000,
        avg_cardinality=2.5,
        seed=42,
    )


@pytest.fixture
def graph_100k() -> HypergraphStore:
    """100K nodes, 500K edges - large benchmark graph."""
    return generate_random_graph(
        num_nodes=100000,
        num_edges=500000,
        avg_cardinality=2.5,
        seed=42,
    )


@pytest.fixture
def sparse_graph_10k() -> HypergraphStore:
    """Sparse 10K graph - fewer edges, lower connectivity."""
    return generate_random_graph(
        num_nodes=10000,
        num_edges=10000,  # 1 edge per node on average
        avg_cardinality=2.0,
        seed=42,
    )


@pytest.fixture
def dense_graph_10k() -> HypergraphStore:
    """Dense 10K graph - more edges, higher connectivity."""
    return generate_random_graph(
        num_nodes=10000,
        num_edges=100000,  # 10 edges per node on average
        avg_cardinality=3.0,
        seed=42,
    )


@pytest.fixture
def hyperedge_heavy_10k() -> HypergraphStore:
    """10K graph with many large hyperedges."""
    return generate_random_graph(
        num_nodes=10000,
        num_edges=20000,
        avg_cardinality=5.0,  # Larger edges
        seed=42,
    )
