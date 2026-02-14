# Hypabase

[![CI](https://github.com/hypabase/hypabase/actions/workflows/ci.yml/badge.svg)](https://github.com/hypabase/hypabase/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hypabase)](https://pypi.org/project/hypabase/)
[![Python](https://img.shields.io/pypi/pyversions/hypabase)](https://pypi.org/project/hypabase/)
[![License](https://img.shields.io/github/license/hypabase/hypabase)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/hypabase)](https://pypi.org/project/hypabase/)

A Python hypergraph library with provenance and SQLite persistence.

## Install

```bash
uv add hypabase
```

## Quick example

```python
from hypabase import Hypabase

hb = Hypabase("my.db")

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

## Features

- **Hyperedges** — an edge connects 2+ nodes in a single relationship
- **Provenance** — every edge carries `source` and `confidence`
- **SQLite persistence** — data persists to a local file automatically
- **O(1) vertex-set lookup** — find edges by their exact node set
- **Namespace isolation** — `.database("name")` for scoped views in a single file
- **Provenance queries** — filter by `source` and `min_confidence`, summarize with `sources()`
- **MCP server** — 14 tools + 2 resources for AI agent integration
- **CLI** — `hypabase init`, `hypabase node`, `hypabase edge`, `hypabase query`

## Provenance

Every edge carries `source` and `confidence`:

```python
hb.edge(
    ["patient_123", "aspirin", "ibuprofen"],
    type="drug_interaction",
    source="clinical_decision_support_v3",
    confidence=0.92,
)

# Bulk provenance via context manager
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

Isolate data into separate namespaces within a single file:

```python
hb = Hypabase("knowledge.db")

drugs = hb.database("drugs")
sessions = hb.database("sessions")

drugs.node("aspirin", type="drug")
sessions.node("s1", type="session")

drugs.nodes()     # -> [aspirin]
sessions.nodes()  # -> [s1]
```

## What is a hypergraph?

In a regular graph, an edge connects exactly two nodes. In a hypergraph, a single edge — called a **hyperedge** — can connect any number of nodes at once.

Consider a medical event: *Dr. Smith prescribes aspirin to Patient 123 for a headache at Mercy Hospital.* In a traditional graph, you'd split this into binary edges — doctor-patient, doctor-drug, patient-hospital — and the fact that they belong to one event becomes an inference, not a structure. A hypergraph stores this natively: one edge connecting all five entities.

This matters because real-world relationships often involve more than two things. A paper has three or four authors, not one. A transaction involves a buyer, a seller, a product, and a payment method. A chemical reaction has reagents and products on both sides. Forcing these into pairs means the grouping becomes implicit.

### Why provenance?

When relationships come from different sources — manual entry, LLM extraction, sensor data, clinical records — you need to know where each one came from and how much you trust it. Hypabase tracks this with two fields on every edge: `source` (a string identifying the origin) and `confidence` (a float from 0 to 1). You can filter queries by these fields and get a summary of all sources in your graph with `hb.sources()`.

### Where hypergraphs show up

- **Knowledge graphs** — representing complex real-world relationships without decomposition
- **Agent memory** — structured, queryable memory for AI agents that persists across sessions
- **Biomedical data** — drug interactions, clinical events, molecular pathways
- **RAG pipelines** — storing extracted relationships for retrieval-augmented generation
- **Supply chains, collaboration networks, and anywhere relationships involve more than two things**

The broader idea has roots in AI research going back to OpenCog's [AtomSpace](https://wiki.opencog.org/w/AtomSpace), which uses hypergraph-like structures to represent knowledge for AGI. More recent work applies hypergraphs specifically to retrieval and reasoning:

- [HyperGraphRAG](https://arxiv.org/abs/2503.21322) — n-ary knowledge retrieval across medicine, agriculture, CS, and law
- [Cog-RAG](https://arxiv.org/abs/2511.13201) — dual-hypergraph retrieval with theme-level and entity-level recall
- [Hypergraph Memory for Multi-step RAG](https://arxiv.org/abs/2512.23959) — hypergraph-based memory for long-context relational modeling

## MCP server

Hypabase includes an MCP server with 14 tools and 2 resources so AI agents can use it as structured memory. Works with Claude Code, Claude Desktop, Cursor, Windsurf, and any MCP-compatible client.

```bash
uv add hypabase[mcp]
hypabase mcp
```

## CLI

```bash
uv add hypabase[cli]
hypabase init
hypabase node dr_smith --type doctor
hypabase edge dr_smith patient_123 aspirin --type treatment --source clinical_records
hypabase query --containing dr_smith
hypabase stats
```

## Documentation

[docs.hypabase.app](https://docs.hypabase.app)

## License

Apache 2.0
