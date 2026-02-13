# Hypabase

[![CI](https://github.com/hypabase/hypabase/actions/workflows/ci.yml/badge.svg)](https://github.com/hypabase/hypabase/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hypabase)](https://pypi.org/project/hypabase/)
[![Python](https://img.shields.io/pypi/pyversions/hypabase)](https://pypi.org/project/hypabase/)
[![License](https://img.shields.io/github/license/hypabase/hypabase)](LICENSE)

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

## Provenance

Every edge carries `source` and `confidence`:

```python
hb.edge(
    ["patient_123", "aspirin", "ibuprofen"],
    type="drug_interaction",
    source="clinical_decision_support_v3",
    confidence=0.92,
)

# Bulk provenance
with hb.context(source="schema_analysis", confidence=0.9):
    hb.edge(["a", "b"], type="fk")
    hb.edge(["b", "c"], type="fk")

# Query by provenance
hb.edges(source="clinical_decision_support_v3")
hb.edges(min_confidence=0.9)

# Overview of all sources
hb.sources()
```

## Namespace isolation

Isolate data into separate namespaces within a single database file:

```python
hb = Hypabase("knowledge.db")

drugs = hb.database("drugs")
sessions = hb.database("sessions")

drugs.node("aspirin", type="drug")
sessions.node("s1", type="session")

drugs.nodes()     # → [aspirin]
sessions.nodes()  # → [s1]
```

## Features

- **N-ary hyperedges** — an edge connects 2+ nodes in a single relationship
- **O(1) vertex-set lookup** — find edges by their exact node set
- **Provenance** — every edge carries `source` and `confidence`
- **Provenance queries** — filter by `source` and `min_confidence`, summarize with `sources()`
- **SQLite persistence** — local-first, zero-config
- **Namespace isolation** — `.database("name")` for scoped views in a single file
- **MCP server** — 14 tools + 2 resources for AI agent integration
- **CLI** — `hypabase init`, `hypabase node`, `hypabase edge`, `hypabase query`

## Documentation

[docs.hypabase.app](https://docs.hypabase.app)

## License

Apache 2.0
