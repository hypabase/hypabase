---
description: "A Python hypergraph library with provenance and SQLite persistence."
---

# Hypabase

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

See [Getting Started](getting-started.md) for the full walkthrough.

## Features

- **N-ary hyperedges** — an edge connects 2+ nodes in a single relationship
- **O(1) vertex-set lookup** — find edges by their exact node set
- **Provenance** — every edge carries `source` and `confidence`
- **Provenance queries** — filter by `source` and `min_confidence`, summarize with `sources()`
- **SQLite persistence** — local-first, zero-config
- **CLI** — `hypabase init`, `hypabase node`, `hypabase edge`, `hypabase query`
- **Python SDK** — keyword args, method names read like English

## Next steps

- [Getting Started](getting-started.md) — install and build your first graph
- [Concepts](concepts.md) — hypergraphs, provenance, and vertex-set indexing
- [API Reference](reference/client.md) — full SDK documentation
- [llms.txt](llms.txt) — LLM-friendly summary of the docs
- [llms-full.txt](llms-full.txt) — full docs in plain text for LLM context
