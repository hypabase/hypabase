---
description: "Install Hypabase and build your first hypergraph with SQLite persistence, namespace isolation, and provenance tracking."
---

# Getting Started

## Installation

=== "uv"

    ```bash
    uv add hypabase
    ```

=== "pip"

    ```bash
    pip install hypabase
    ```

For CLI support:

```bash
uv add "hypabase[cli]"
```

## Your first hypergraph

```python
from hypabase import Hypabase

# File-backed database (persists to SQLite)
hb = Hypabase("my.db")

# Or in-memory for experiments
hb = Hypabase()
```

### Create nodes

```python
hb.node("dr_smith", type="doctor")
hb.node("patient_123", type="patient")
hb.node("aspirin", type="medication")
hb.node("headache", type="condition")
hb.node("mercy_hospital", type="hospital")
```

### Create a hyperedge

A single edge connects all five entities atomically:

```python
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)
```

!!! note
    Nodes referenced in an edge are auto-created if they don't exist. You can skip explicit `node()` calls if you don't need to set node types or properties upfront.

### Query edges

```python
# All edges involving a patient
edges = hb.edges(containing=["patient_123"])

# Edges connecting both patient and medication
edges = hb.edges(containing=["patient_123", "aspirin"], match_all=True)

# Filter by type
edges = hb.edges(type="treatment")

# Filter by provenance
edges = hb.edges(source="clinical_records")
edges = hb.edges(min_confidence=0.9)
```

### Find paths

```python
paths = hb.paths("dr_smith", "mercy_hospital")
# [["dr_smith", ..., "mercy_hospital"]]
```

### Check stats

```python
stats = hb.stats()
print(f"Nodes: {stats.node_count}, Edges: {stats.edge_count}")
```

## Using provenance

Every edge carries `source` and `confidence`. Set them per-edge or in bulk with a context manager:

```python
# Per-edge
hb.edge(
    ["patient_123", "aspirin", "ibuprofen"],
    type="drug_interaction",
    source="clinical_decision_support_v3",
    confidence=0.92,
)

# Bulk — all edges inside inherit source and confidence
with hb.context(source="schema_analysis", confidence=0.9):
    hb.edge(["a", "b"], type="fk")
    hb.edge(["b", "c"], type="fk")

# Query by provenance
hb.edges(source="clinical_decision_support_v3")
hb.edges(min_confidence=0.9)

# Overview of all sources
hb.sources()
# [{"source": "clinical_decision_support_v3", "edge_count": 1, "avg_confidence": 0.92}, ...]
```

## File persistence

```python
# Data persists across sessions
with Hypabase("project.db") as hb:
    hb.node("alice", type="user")
    hb.edge(["alice", "task_1"], type="assigned")
# Automatically saved and closed

# Reopen later
with Hypabase("project.db") as hb:
    edges = hb.edges(containing=["alice"])  # data is still there
```

## Namespace isolation

Separate data into independent namespaces within a single database file:

```python
hb = Hypabase("project.db")

# Scoped views — each namespace has its own nodes and edges
drugs = hb.database("drugs")
sessions = hb.database("sessions")

drugs.node("aspirin", type="medication")
sessions.node("session_1", type="session")

# List all namespaces
hb.databases()  # ["default", "drugs", "sessions"]
```

## CLI quickstart

```bash
# Initialize a database
hypabase init

# Add nodes and edges
hypabase node dr_smith --type doctor
hypabase edge dr_smith patient_123 aspirin --type treatment --source clinical --confidence 0.95

# Query
hypabase query --containing patient_123
hypabase stats
```

## Next steps

- [Concepts](concepts.md) — learn about hypergraphs, provenance, and vertex-set indexing
- [Traversal guide](guides/traversal.md) — neighbors, shortest paths, and multi-hop queries
- [Provenance guide](guides/provenance.md) — context managers, overrides, and source queries
- [CLI Quickstart](guides/cli.md) — build a knowledge graph from the terminal
- [Examples](examples/medical-kg.md) — real-world use cases with working code
- [Comparisons](comparisons/vs-neo4j.md) — how Hypabase compares to Neo4j, vector DBs, and Mem0
