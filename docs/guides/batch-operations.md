# Batch Operations

## Batch persistence

By default, Hypabase auto-saves to SQLite after every mutation. For bulk inserts, use `batch()` to defer persistence until the block exits:

```python
with hb.batch():
    for i in range(1000):
        hb.node(f"entity_{i}", type="item")
        hb.edge([f"entity_{i}", "catalog"], type="belongs_to")
# Single save at the end, not 2000 saves
```

!!! note
    `batch()` provides batched persistence, not transaction rollback. If an exception occurs mid-batch, partial in-memory changes remain and persist when the batch exits.

### Nested batches

Batches can nest. Only the outermost batch triggers a save:

```python
with hb.batch():
    hb.node("a", type="x")
    with hb.batch():
        hb.node("b", type="x")
        hb.node("c", type="x")
    # No save yet — inner batch exited but outer batch is still open
    hb.node("d", type="x")
# Save happens here — outermost batch exits
```

## Upsert by vertex set

`upsert_edge_by_vertex_set()` finds an existing edge by its exact set of nodes, or creates a new one. This is useful for idempotent ingestion:

```python
# First call creates the edge
edge = hb.upsert_edge_by_vertex_set(
    {"dr_smith", "patient_123", "aspirin"},
    edge_type="treatment",
    properties={"date": "2025-01-15"},
    source="clinical_records",
    confidence=0.95,
)

# Second call finds the existing edge (same vertex set)
edge = hb.upsert_edge_by_vertex_set(
    {"dr_smith", "patient_123", "aspirin"},
    edge_type="treatment",
    properties={"date": "2025-01-16"},  # updates properties
)
```

### Custom merge function

Pass a `merge_fn` to control how the upsert merges properties:

```python
def merge_latest(existing_props, new_props):
    return {**existing_props, **new_props}

hb.upsert_edge_by_vertex_set(
    {"a", "b"},
    edge_type="link",
    properties={"count": 2},
    merge_fn=merge_latest,
)
```

## Cascade delete

Delete a node and all its incident edges in one call:

```python
node_deleted, edges_deleted = hb.delete_node_cascade("patient_123")
# node_deleted: True if the node existed
# edges_deleted: number of edges removed
```

Compare with `delete_node()`, which only removes the node itself:

```python
hb.delete_node("patient_123")  # Removes the node, edges remain (with dangling references)
```

## Bulk ingestion pattern

Combine `batch()` and `context()` for efficient bulk loading:

```python
with hb.batch():
    with hb.context(source="data_import_v2", confidence=0.9):
        for record in records:
            hb.edge(
                record["entities"],
                type=record["relation_type"],
                properties=record.get("metadata", {}),
            )
```

This gives you:

- Single disk write at the end (`batch`)
- Consistent provenance across all edges (`context`)
- Auto-created nodes for any new entity IDs
