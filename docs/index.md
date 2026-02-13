# Hypabase

Hypabase is a Python library for storing and querying relationships between entities. A single edge connects two or more nodes, every edge tracks where it came from (`source` and `confidence`), and the whole graph lives in a local SQLite file with no server or configuration.

Use it to build knowledge graphs, retrieval-augmented generation pipelines, and structured agent memory. Recent research explores hypergraph representations for these tasks:

- [HyperGraphRAG](https://arxiv.org/abs/2503.21322) — n-ary knowledge retrieval across medicine, agriculture, CS, and law
- [Cog-RAG](https://arxiv.org/abs/2511.13201) — dual-hypergraph retrieval with theme-level and entity-level recall
- [Hypergraph Memory for Multi-step RAG](https://arxiv.org/abs/2512.23959) — hypergraph-based memory for long-context relational modeling

## Install

```bash
uv add hypabase
```

## Quick example

```python
from hypabase import Hypabase

hb = Hypabase("my.db")  # local SQLite, zero config

# One edge connecting five entities
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)

# Query edges involving a node
hb.edges(containing=["patient_123"])

# Find paths between entities
hb.paths("dr_smith", "mercy_hospital")
```

See [Getting Started](getting-started.md) for the full walkthrough.

## Features

- **N-ary hyperedges** — an edge connects 2+ nodes in a single relationship
- **O(1) vertex-set lookup** — find edges by their exact node set
- **Provenance** — every edge carries `source` and `confidence`
- **Provenance queries** — filter by `source` and `min_confidence`, summarize with `sources()`
- **SQLite persistence** — local-first, zero-config
- **CLI** — `hypabase init`, `hypabase node`, `hypabase edge`, `hypabase query`
- **Python SDK** — keyword args, method names read like English

## Limitations

- No semantic similarity or fuzzy search — pair with a vector database for that ([hybrid pattern](examples/hybrid-vector.md))
- No declarative query language (e.g., Cypher, SPARQL) — use the Python SDK, CLI, or MCP tools
- No built-in visualization
- Early project — small community

## Next steps

- [Getting Started](getting-started.md) — install and build your first graph
- [Concepts](concepts.md) — hypergraphs, provenance, and vertex-set indexing
- [API Reference](reference/client.md) — full SDK documentation
- [llms.txt](llms.txt) — LLM-friendly summary of the docs
- [llms-full.txt](llms-full.txt) — full docs in plain text for LLM context
