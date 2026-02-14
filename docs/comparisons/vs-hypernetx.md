# Hypabase vs HyperNetX

## Both are Python hypergraph libraries

[HyperNetX](https://github.com/pnnl/HyperNetX) (HNX) is a Python library from Pacific Northwest National Laboratory for analyzing and visualizing hypergraphs. Hypabase is a Python hypergraph library focused on storage, querying, and provenance tracking. They solve different problems within the same domain.

## Different focus

HyperNetX is built for **analysis and visualization**. It provides hypergraph metrics, set operations (union, intersection, difference), and visualization tools. It's designed for researchers who want to study hypergraph structure.

Hypabase is built for **storage and querying with provenance**. It provides CRUD operations, SQLite persistence, provenance tracking, and an MCP server. It's designed for applications that need to build, persist, and query hypergraphs.

## Comparison

| | HyperNetX | Hypabase |
|---|---|---|
| **Focus** | Analysis and visualization | Storage, querying, and provenance |
| **Persistence** | None (in-memory only) | SQLite (automatic) |
| **Provenance** | None | Built-in `source` and `confidence` |
| **Visualization** | Built-in | None |
| **Algorithms** | Metrics, set operations, analysis | Traversal, path finding, vertex-set lookup |
| **Namespace isolation** | None | `.database("name")` scoping |
| **Data backend** | Pandas DataFrames | Python dicts + SQLite |
| **MCP server** | None | 14 tools + 2 resources |
| **HIF support** | [Core contributor](https://github.com/pnnl/HyperNetX) to the format | Import/export supported |
| **API style** | Analysis-oriented | CRUD-oriented |

## When to use HyperNetX

- You need to visualize hypergraphs
- You need hypergraph analysis algorithms (metrics, set operations, arithmetic)
- You're doing academic research on hypergraph structure
- You need to compute hypergraph properties like s-connectedness or centrality measures

## When to use Hypabase

- You need to persist hypergraphs to disk and query them later
- You need provenance tracking on every edge
- You're building an application that creates and queries hypergraphs over time
- You need isolated namespaces for different domains or data sources
- You're integrating with AI agents via MCP

## Using them together

HyperNetX and Hypabase can complement each other. Use Hypabase as the persistent storage and query layer, and export to HyperNetX (via HIF) when you need analysis or visualization.

```python
import json
from hypabase import Hypabase

hb = Hypabase("my.db")

# Build your hypergraph with Hypabase
hb.edge(["a", "b", "c"], type="collaboration", source="hr_system", confidence=0.9)
hb.edge(["b", "c", "d"], type="collaboration", source="hr_system", confidence=0.85)

# Export to HIF for HyperNetX analysis
hif_data = hb.to_hif()

# Load in HyperNetX
import hypernetx as hnx
# ... construct HNX hypergraph from HIF data for visualization and analysis
```
