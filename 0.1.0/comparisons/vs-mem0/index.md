# Hypabase vs Mem0

## Different memory models

Mem0 stores flat facts — individual key-value memories like "Alice prefers dark mode" or "User works at Acme Corp." Each fact stands alone.

Hypabase stores structured relationships — edges connecting two or more entities with provenance. "Alice works on the quarterly report with the spreadsheet tool" is one relationship, not three separate facts.

## Architectural differences

### Mem0

Mem0 stores each fact as an independent memory entry:

```
mem0.add("Alice is assigned to write the quarterly report", user_id="alice")
mem0.add("Alice uses the spreadsheet tool", user_id="alice")
mem0.add("The quarterly report is due Q3", user_id="alice")
```

### Hypabase

Hypabase stores facts as connected edges with explicit relationships between entities:

```
with hb.context(source="session_1", confidence=0.9):
    hb.edge(
        ["user_alice", "task_write_report", "doc_quarterly"],
        type="assigned",
    )
    hb.edge(
        ["user_alice", "task_write_report", "tool_spreadsheet"],
        type="uses_tool",
    )
```

The relationships are explicit. Query them:

```
# What tools are used for the report task?
report_edges = hb.edges(containing=["task_write_report"], type="uses_tool")

# How are the report and spreadsheet connected?
paths = hb.paths("doc_quarterly", "tool_spreadsheet")
# doc_quarterly → user_alice → tool_spreadsheet
```

## Comparison

|                           | Mem0                              | Hypabase                                  |
| ------------------------- | --------------------------------- | ----------------------------------------- |
| **Memory model**          | Flat facts (key-value)            | Structured relationships (hyperedges)     |
| **Relationships**         | Not stored                        | First-class edges connecting N entities   |
| **Multi-entity facts**    | Fragmented into separate memories | Single atomic edge                        |
| **Provenance**            | None                              | Built-in `source` and `confidence`        |
| **Cross-session queries** | Search by user/text               | Query by entity, type, source, confidence |
| **Path finding**          | Not possible                      | `hb.paths(start, end)`                    |
| **Storage**               | Cloud API                         | Local SQLite (zero-config)                |
| **Retrieval**             | Text similarity search            | Exact structured queries                  |

## When to use Mem0 instead

- You only need user preference storage without relationships
- You want managed cloud storage with no local infrastructure
- Your facts stand alone with no connections between them
- You need text-based semantic search over memories

## When to use Hypabase instead

- Your agent needs to remember relationships between entities (people, tasks, tools, documents)
- You need to traverse connections between memories
- You need provenance — which session or interaction created each memory
- You need confidence scores to distinguish certain from inferred memories
- You want local-first storage without cloud dependencies

## Session-aware memory

Hypabase tracks which session created each memory using provenance context blocks. See the [Agent Memory example](https://hypabase.app/docs/latest/examples/agent-memory/index.md) for a complete multi-session walkthrough.
