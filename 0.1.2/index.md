# Hypabase

A Python hypergraph library with provenance and SQLite persistence.

## Install

```
uv add hypabase
```

## Quick example

```
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

See [Getting Started](https://docs.hypabase.app/latest/getting-started/index.md) for the full walkthrough.

## Features

- **N-ary hyperedges** — an edge connects 2+ nodes in a single relationship
- **O(1) vertex-set lookup** — find edges by their exact node set
- **Provenance** — every edge carries `source` and `confidence`
- **Provenance queries** — filter by `source` and `min_confidence`, summarize with `sources()`
- **SQLite persistence** — local-first, zero-config
- **CLI** — `hypabase init`, `hypabase node`, `hypabase edge`, `hypabase query`
- **Python SDK** — keyword args, method names read like English

## Next steps

- [Getting Started](https://docs.hypabase.app/latest/getting-started/index.md) — install and build your first graph
- [Concepts](https://docs.hypabase.app/latest/concepts/index.md) — hypergraphs, provenance, and vertex-set indexing
- [API Reference](https://docs.hypabase.app/latest/reference/client/index.md) — full SDK documentation
- [llms.txt](https://docs.hypabase.app/latest/llms.txt) — LLM-friendly summary of the docs
- [llms-full.txt](https://docs.hypabase.app/latest/llms-full.txt) — full docs in plain text for LLM context
