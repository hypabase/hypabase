# Traversal

Hypabase provides methods for navigating the hypergraph: finding neighbors, discovering paths, and querying incident edges.

## Neighbors

Find all nodes connected to a given node through any shared edge:

```
neighbors = hb.neighbors("patient_123")
# Returns list of Node objects connected to patient_123
```

### Filter by edge type

```
# Only neighbors connected via treatment edges
neighbors = hb.neighbors("patient_123", edge_types=["treatment"])
```

The result excludes the query node itself.

## Paths

Find paths between two nodes through hyperedges:

```
paths = hb.paths("dr_smith", "mercy_hospital")
# [["dr_smith", "patient_123", "mercy_hospital"], ...]
```

Each path is a list of node IDs from start to end.

### Limit hop count

```
# Only short paths (up to 3 hops)
paths = hb.paths("dr_smith", "mercy_hospital", max_hops=3)
```

The default `max_hops` is 5.

### Filter by edge type

```
# Only traverse treatment and diagnosis edges
paths = hb.paths(
    "dr_smith",
    "mercy_hospital",
    edge_types=["treatment", "diagnosis"],
)
```

## Advanced path finding

`find_paths()` provides intersection-constrained path finding — it returns paths as sequences of edges rather than node IDs, and supports set-based start/end nodes:

```
paths = hb.find_paths(
    start_nodes={"dr_smith", "dr_jones"},
    end_nodes={"mercy_hospital"},
    max_hops=3,
    max_paths=10,
    edge_types=["treatment"],
)
# Returns list of list[Edge]
```

Parameters:

- `start_nodes` — set of possible start node IDs
- `end_nodes` — set of possible end node IDs
- `max_hops` — longest path length allowed (default 3)
- `max_paths` — cap on paths returned (default 10)
- `min_intersection` — required node overlap between consecutive edges (default 1)
- `edge_types` — filter to specific edge types
- `direction_mode` — `"undirected"` (default), `"forward"`, or `"backward"`

## Edges of a node

Get all edges incident to a specific node:

```
edges = hb.edges_of_node("patient_123")
# All edges that include patient_123
```

Filter by edge type:

```
edges = hb.edges_of_node("patient_123", edge_types=["treatment"])
```

## Graph metrics

### Node degree

Number of edges incident to a node:

```
degree = hb.node_degree("patient_123")
```

Filter by edge type:

```
degree = hb.node_degree("patient_123", edge_types=["treatment"])
```

### Edge cardinality

Number of unique nodes in an edge:

```
cardinality = hb.edge_cardinality(edge_id)
# 5 for a 5-node hyperedge
```

### Hyperedge degree

Sum of vertex degrees of nodes in a given set:

```
degree = hb.hyperedge_degree({"dr_smith", "patient_123"})
```
