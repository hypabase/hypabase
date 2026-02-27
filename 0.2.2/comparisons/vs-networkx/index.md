# Hypabase vs NetworkX

## Different data models

[NetworkX](https://networkx.org/) is the standard Python library for graph analysis. It supports undirected graphs, directed graphs, and multigraphs — but all edges are binary (connecting exactly two nodes).

Hypabase is a hypergraph library where a single edge connects any number of nodes.

If your relationships are all pairwise, NetworkX is the established choice. If your relationships involve 3+ entities and you need them to stay atomic, that's what Hypabase is for.

## The n-ary problem in NetworkX

Consider: "Alice, Bob, and Carol co-authored a paper."

### NetworkX

NetworkX has no native way to represent a 3-way relationship. Common workarounds:

**Clique expansion** — create pairwise edges between all participants:

```
import networkx as nx

G = nx.Graph()
G.add_edge("alice", "bob", type="coauthor")
G.add_edge("alice", "carol", type="coauthor")
G.add_edge("bob", "carol", type="coauthor")
```

The grouping — the fact that all three were part of *one* collaboration — becomes implicit. You can't distinguish "all three worked together" from "three separate pairs happened to collaborate."

**Bipartite/intermediate node** — add a node to represent the relationship:

```
G.add_edge("alice", "paper_1")
G.add_edge("bob", "paper_1")
G.add_edge("carol", "paper_1")
```

This preserves the grouping but mixes entities and relationships into the same node space.

### Hypabase

```
hb.edge(
    ["alice", "bob", "carol"],
    type="coauthor",
    source="publication_db",
    confidence=1.0,
)
```

One edge, three nodes, with provenance.

## Comparison

|                         | NetworkX                                                         | Hypabase                                   |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------------ |
| **Edge model**          | Binary                                                           | N-ary (2+ nodes)                           |
| **N-ary relationships** | Workarounds (cliques, bipartite)                                 | Native hyperedges                          |
| **Algorithms**          | Extensive (centrality, community detection, flow, matching, ...) | Traversal, path finding, vertex-set lookup |
| **Visualization**       | matplotlib, pyvis, etc.                                          | None                                       |
| **Persistence**         | Manual serialization (GraphML, GML, pickle, ...)                 | Automatic SQLite                           |
| **Provenance**          | None                                                             | Built-in `source` and `confidence`         |
| **Documentation**       | Extensive (textbooks, courses, tutorials)                        | Docs site                                  |
| **Graph types**         | Undirected, directed, multigraph                                 | Undirected and directed hyperedges         |

## When to use NetworkX

- Your relationships are all pairwise
- You need graph algorithms (centrality, community detection, shortest paths, flow, matching)
- You need visualization
- You're following tutorials or academic papers that use NetworkX

## When to use Hypabase

- Your relationships involve 3+ entities and you want the grouping stored in the edge itself
- You need provenance tracking on relationships
- You want automatic persistence without managing serialization
- You need isolated namespaces for different domains or data sources
