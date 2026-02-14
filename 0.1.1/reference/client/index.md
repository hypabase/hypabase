# Client API

## hypabase.client.Hypabase

Hypergraph client.

The primary interface for creating, querying, and traversing hypergraphs. Supports in-memory and local SQLite backends.

Constructor patterns

- `Hypabase()` — in-memory, ephemeral (SQLite `:memory:`)
- `Hypabase("file.db")` — local persistent SQLite file
- `Hypabase("https://...")` — cloud backend (Phase 3, raises NotImplementedError)

Example

```
hb = Hypabase()                          # in-memory
hb = Hypabase("myproject.db")             # local SQLite file

# Namespace isolation
drugs = hb.database("drugs")
sessions = hb.database("sessions")
```

### current_database

```
current_database: str
```

Current namespace name.

### close

```
close() -> None
```

Close the database connection.

Saves pending changes and releases the SQLite connection. No-op for in-memory instances.

### save

```
save() -> None
```

Persist current state to SQLite.

No-op for in-memory instances. Normally called automatically after each mutation; use this for explicit manual saves.

### database

```
database(name: str) -> Hypabase
```

Return a scoped view into a named namespace.

The returned instance shares the same SQLite connection and stores dict, but reads/writes only the given namespace's data.

Parameters:

| Name   | Type  | Description     | Default    |
| ------ | ----- | --------------- | ---------- |
| `name` | `str` | Namespace name. | *required* |

Returns:

| Type       | Description                                      |
| ---------- | ------------------------------------------------ |
| `Hypabase` | A new Hypabase instance scoped to the namespace. |

### databases

```
databases() -> list[str]
```

List all namespaces.

Returns:

| Type        | Description                     |
| ----------- | ------------------------------- |
| `list[str]` | Sorted list of namespace names. |

### delete_database

```
delete_database(name: str) -> bool
```

Delete a namespace and all its data.

Parameters:

| Name   | Type  | Description          | Default    |
| ------ | ----- | -------------------- | ---------- |
| `name` | `str` | Namespace to delete. | *required* |

Returns:

| Type   | Description                                     |
| ------ | ----------------------------------------------- |
| `bool` | True if the namespace existed, False otherwise. |

### context

```
context(*, source: str, confidence: float = 1.0) -> Generator[None, None, None]
```

Set default provenance for all edges created within the block.

Edges created inside the context inherit `source` and `confidence` unless overridden per-edge. Contexts can be nested; the innermost wins.

Parameters:

| Name         | Type    | Description                                           | Default    |
| ------------ | ------- | ----------------------------------------------------- | ---------- |
| `source`     | `str`   | Provenance source string (e.g., "gpt-4o_extraction"). | *required* |
| `confidence` | `float` | Default confidence score, 0.0-1.0.                    | `1.0`      |

Example

```
with hb.context(source="clinical_records", confidence=0.95):
    hb.edge(["a", "b"], type="link")  # inherits provenance
```

### node

```
node(id: str, *, type: str = 'unknown', **properties: Any) -> Node
```

Create or update a node.

If a node with the given ID exists, its type and properties are updated. Otherwise a new node is created.

Parameters:

| Name           | Type  | Description                                      | Default     |
| -------------- | ----- | ------------------------------------------------ | ----------- |
| `id`           | `str` | Unique node identifier.                          | *required*  |
| `type`         | `str` | Node classification (e.g., "doctor", "patient"). | `'unknown'` |
| `**properties` | `Any` | Arbitrary key-value metadata stored on the node. | `{}`        |

Returns:

| Type   | Description                  |
| ------ | ---------------------------- |
| `Node` | The created or updated Node. |

Raises:

| Type         | Description               |
| ------------ | ------------------------- |
| `ValueError` | If id is an empty string. |

### get_node

```
get_node(id: str) -> Node | None
```

Get a node by ID.

Parameters:

| Name | Type  | Description             | Default    |
| ---- | ----- | ----------------------- | ---------- |
| `id` | `str` | The node ID to look up. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| \`Node | None\`      |

### nodes

```
nodes(*, type: str | None = None) -> list[Node]
```

Query nodes, optionally filtered by type.

Parameters:

| Name   | Type  | Description | Default                                      |
| ------ | ----- | ----------- | -------------------------------------------- |
| `type` | \`str | None\`      | If provided, return only nodes of this type. |

Returns:

| Type         | Description             |
| ------------ | ----------------------- |
| `list[Node]` | List of matching nodes. |

### find_nodes

```
find_nodes(**properties: Any) -> list[Node]
```

Find nodes matching all specified properties.

Parameters:

| Name           | Type  | Description                                      | Default |
| -------------- | ----- | ------------------------------------------------ | ------- |
| `**properties` | `Any` | Key-value pairs that must match node properties. | `{}`    |

Returns:

| Type         | Description             |
| ------------ | ----------------------- |
| `list[Node]` | List of matching nodes. |

Example

```
hb.find_nodes(role="admin", active=True)
```

### has_node

```
has_node(id: str) -> bool
```

Check if a node exists.

Parameters:

| Name | Type  | Description           | Default    |
| ---- | ----- | --------------------- | ---------- |
| `id` | `str` | The node ID to check. | *required* |

Returns:

| Type   | Description                               |
| ------ | ----------------------------------------- |
| `bool` | True if the node exists, False otherwise. |

### delete_node

```
delete_node(id: str, *, cascade: bool = False) -> bool
```

Delete a node by ID.

Parameters:

| Name      | Type   | Description                              | Default    |
| --------- | ------ | ---------------------------------------- | ---------- |
| `id`      | `str`  | The node ID to delete.                   | *required* |
| `cascade` | `bool` | If True, also delete all incident edges. | `False`    |

Returns:

| Type   | Description                                                |
| ------ | ---------------------------------------------------------- |
| `bool` | True if the node existed and was deleted, False otherwise. |

### delete_node_cascade

```
delete_node_cascade(node_id: str) -> tuple[bool, int]
```

Delete a node and all its incident edges.

.. deprecated:: 0.2.0 Use `delete_node(id, cascade=True)` instead.

Parameters:

| Name      | Type  | Description            | Default    |
| --------- | ----- | ---------------------- | ---------- |
| `node_id` | `str` | The node ID to delete. | *required* |

Returns:

| Type               | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `tuple[bool, int]` | Tuple of (node_was_deleted, number_of_edges_deleted). |

### edge

```
edge(nodes: list[str], *, type: str, directed: bool = False, source: str | None = None, confidence: float | None = None, properties: dict[str, Any] | None = None, id: str | None = None) -> Edge
```

Create a hyperedge linking two or more nodes in one relationship.

Nodes are auto-created if they don't exist. Provenance values fall back to the active `context()` block if not set explicitly.

Parameters:

| Name         | Type             | Description                                    | Default                                                 |
| ------------ | ---------------- | ---------------------------------------------- | ------------------------------------------------------- |
| `nodes`      | `list[str]`      | Node IDs to connect. Must contain at least 2.  | *required*                                              |
| `type`       | `str`            | Edge type (e.g., "treatment", "concept_link"). | *required*                                              |
| `directed`   | `bool`           | If True, first node is tail, last is head.     | `False`                                                 |
| `source`     | \`str            | None\`                                         | Provenance source. Falls back to context or "unknown".  |
| `confidence` | \`float          | None\`                                         | Confidence score 0.0-1.0. Falls back to context or 1.0. |
| `properties` | \`dict[str, Any] | None\`                                         | Arbitrary key-value metadata.                           |
| `id`         | \`str            | None\`                                         | Optional edge ID. Auto-generated UUID if omitted.       |

Returns:

| Type   | Description       |
| ------ | ----------------- |
| `Edge` | The created Edge. |

Raises:

| Type         | Description                                    |
| ------------ | ---------------------------------------------- |
| `ValueError` | If fewer than 2 nodes or any node ID is empty. |

Example

```
hb.edge(
    ["dr_smith", "patient_123", "aspirin"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)
```

### get_edge

```
get_edge(id: str) -> Edge | None
```

Get an edge by ID.

Parameters:

| Name | Type  | Description             | Default    |
| ---- | ----- | ----------------------- | ---------- |
| `id` | `str` | The edge ID to look up. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| \`Edge | None\`      |

### edges

```
edges(*, containing: list[str] | None = None, type: str | None = None, match_all: bool = False, source: str | None = None, min_confidence: float | None = None) -> list[Edge]
```

Query edges by contained nodes, type, source, and/or confidence.

All filters are combined with AND logic.

Parameters:

| Name             | Type        | Description                                                                                  | Default                                        |
| ---------------- | ----------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `containing`     | \`list[str] | None\`                                                                                       | Node IDs that must appear in the edge.         |
| `type`           | \`str       | None\`                                                                                       | Filter to edges of this type.                  |
| `match_all`      | `bool`      | If True, edges must contain all nodes in containing. If False (default), any match suffices. | `False`                                        |
| `source`         | \`str       | None\`                                                                                       | Filter to edges from this provenance source.   |
| `min_confidence` | \`float     | None\`                                                                                       | Filter to edges with confidence >= this value. |

Returns:

| Type         | Description             |
| ------------ | ----------------------- |
| `list[Edge]` | List of matching edges. |

Example

```
hb.edges(containing=["patient_123"], min_confidence=0.9)
```

### find_edges

```
find_edges(**properties: Any) -> list[Edge]
```

Find edges matching all specified properties.

Parameters:

| Name           | Type  | Description                                      | Default |
| -------------- | ----- | ------------------------------------------------ | ------- |
| `**properties` | `Any` | Key-value pairs that must match edge properties. | `{}`    |

Returns:

| Type         | Description             |
| ------------ | ----------------------- |
| `list[Edge]` | List of matching edges. |

### has_edge_with_nodes

```
has_edge_with_nodes(node_ids: set[str], edge_type: str | None = None) -> bool
```

Check if an edge with the exact vertex set exists.

Parameters:

| Name        | Type       | Description            | Default                                |
| ----------- | ---------- | ---------------------- | -------------------------------------- |
| `node_ids`  | `set[str]` | Exact set of node IDs. | *required*                             |
| `edge_type` | \`str      | None\`                 | If provided, also filter by edge type. |

Returns:

| Type   | Description                   |
| ------ | ----------------------------- |
| `bool` | True if matching edge exists. |

### sources

```
sources() -> list[dict[str, Any]]
```

Summarize provenance sources across all edges.

Returns:

| Type                   | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `list[dict[str, Any]]` | List of dicts with keys "source", "edge_count", |
| `list[dict[str, Any]]` | and "avg_confidence" for each unique source.    |

Example

```
hb.sources()
# [{"source": "clinical_records", "edge_count": 2, "avg_confidence": 0.95}]
```

### edges_by_vertex_set

```
edges_by_vertex_set(nodes: list[str]) -> list[Edge]
```

O(1) lookup: find edges with exactly this set of nodes.

Uses the SHA-256 vertex-set hash index for constant-time lookup. Order of `nodes` does not matter.

Parameters:

| Name    | Type        | Description                         | Default    |
| ------- | ----------- | ----------------------------------- | ---------- |
| `nodes` | `list[str]` | The exact set of node IDs to match. | *required* |

Returns:

| Type         | Description                           |
| ------------ | ------------------------------------- |
| `list[Edge]` | Edges whose node set matches exactly. |

### delete_edge

```
delete_edge(id: str) -> bool
```

Delete an edge by ID.

Parameters:

| Name | Type  | Description            | Default    |
| ---- | ----- | ---------------------- | ---------- |
| `id` | `str` | The edge ID to delete. | *required* |

Returns:

| Type   | Description                                                |
| ------ | ---------------------------------------------------------- |
| `bool` | True if the edge existed and was deleted, False otherwise. |

### neighbors

```
neighbors(node_id: str, *, edge_types: list[str] | None = None) -> list[Node]
```

Find all nodes connected to the given node via shared edges.

The query node itself is excluded from the results.

Parameters:

| Name         | Type        | Description                    | Default                                          |
| ------------ | ----------- | ------------------------------ | ------------------------------------------------ |
| `node_id`    | `str`       | The node to find neighbors of. | *required*                                       |
| `edge_types` | \`list[str] | None\`                         | If provided, only traverse edges of these types. |

Returns:

| Type         | Description                |
| ------------ | -------------------------- |
| `list[Node]` | List of neighboring nodes. |

### paths

```
paths(start: str, end: str, *, max_hops: int = 5, edge_types: list[str] | None = None) -> list[list[str]]
```

Find paths between two nodes through hyperedges.

Uses breadth-first search. Each path is a list of node IDs from `start` to `end`.

Parameters:

| Name         | Type        | Description                         | Default                                          |
| ------------ | ----------- | ----------------------------------- | ------------------------------------------------ |
| `start`      | `str`       | Starting node ID.                   | *required*                                       |
| `end`        | `str`       | Target node ID.                     | *required*                                       |
| `max_hops`   | `int`       | Maximum number of hops (default 5). | `5`                                              |
| `edge_types` | \`list[str] | None\`                              | If provided, only traverse edges of these types. |

Returns:

| Type              | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `list[list[str]]` | List of paths, where each path is a list of node IDs. |

Example

```
paths = hb.paths("dr_smith", "mercy_hospital")
# [["dr_smith", "patient_123", "mercy_hospital"]]
```

### find_paths

```
find_paths(start_nodes: set[str], end_nodes: set[str], *, max_hops: int = 3, max_paths: int = 10, min_intersection: int = 1, edge_types: list[str] | None = None, direction_mode: str = 'undirected') -> list[list[Edge]]
```

Find paths between two groups of nodes through shared edges.

Returns paths as sequences of edges. Supports set-based start/end nodes and configurable overlap requirements.

Parameters:

| Name               | Type        | Description                                                 | Default                                          |
| ------------------ | ----------- | ----------------------------------------------------------- | ------------------------------------------------ |
| `start_nodes`      | `set[str]`  | Set of possible starting node IDs.                          | *required*                                       |
| `end_nodes`        | `set[str]`  | Set of possible ending node IDs.                            | *required*                                       |
| `max_hops`         | `int`       | Maximum path length in edges (default 3).                   | `3`                                              |
| `max_paths`        | `int`       | Maximum number of paths to return (default 10).             | `10`                                             |
| `min_intersection` | `int`       | Minimum node overlap between consecutive edges (default 1). | `1`                                              |
| `edge_types`       | \`list[str] | None\`                                                      | If provided, only traverse edges of these types. |
| `direction_mode`   | `str`       | "undirected" (default), "forward", or "backward".           | `'undirected'`                                   |

Returns:

| Type               | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `list[list[Edge]]` | List of paths, where each path is a list of Edge objects. |

### node_degree

```
node_degree(node_id: str, *, edge_types: list[str] | None = None) -> int
```

Count how many edges touch a node.

Parameters:

| Name         | Type        | Description          | Default                                       |
| ------------ | ----------- | -------------------- | --------------------------------------------- |
| `node_id`    | `str`       | The node to measure. | *required*                                    |
| `edge_types` | \`list[str] | None\`               | If provided, only count edges of these types. |

Returns:

| Type  | Description                          |
| ----- | ------------------------------------ |
| `int` | The degree (edge count) of the node. |

### edge_cardinality

```
edge_cardinality(edge_id: str) -> int
```

Count how many distinct nodes an edge contains.

Parameters:

| Name      | Type  | Description          | Default    |
| --------- | ----- | -------------------- | ---------- |
| `edge_id` | `str` | The edge to measure. | *required* |

Returns:

| Type  | Description                             |
| ----- | --------------------------------------- |
| `int` | Count of distinct node IDs in the edge. |

### hyperedge_degree

```
hyperedge_degree(node_set: set[str], *, edge_type: str | None = None) -> int
```

Add up the edge counts of every node in a set.

Parameters:

| Name        | Type       | Description                   | Default                                     |
| ----------- | ---------- | ----------------------------- | ------------------------------------------- |
| `node_set`  | `set[str]` | Set of node IDs to aggregate. | *required*                                  |
| `edge_type` | \`str      | None\`                        | If provided, only count edges of this type. |

Returns:

| Type  | Description                     |
| ----- | ------------------------------- |
| `int` | Sum of individual node degrees. |

### validate

```
validate() -> ValidationResult
```

Check the hypergraph for internal consistency.

Returns:

| Type               | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `ValidationResult` | A ValidationResult with valid, errors, and warnings fields. |

### to_hif

```
to_hif() -> dict
```

Export the graph to HIF (Hypergraph Interchange Format).

Returns:

| Type   | Description                                               |
| ------ | --------------------------------------------------------- |
| `dict` | A dict representing the hypergraph in HIF JSON structure. |

### from_hif

```
from_hif(hif_data: dict) -> Hypabase
```

Build a new Hypabase instance from HIF (Hypergraph Interchange Format) data.

Creates an in-memory instance populated from the HIF structure.

Parameters:

| Name       | Type   | Description                | Default    |
| ---------- | ------ | -------------------------- | ---------- |
| `hif_data` | `dict` | A dict in HIF JSON format. | *required* |

Returns:

| Type       | Description                                           |
| ---------- | ----------------------------------------------------- |
| `Hypabase` | A new Hypabase instance containing the imported data. |

### upsert_edge_by_vertex_set

```
upsert_edge_by_vertex_set(node_ids: set[str], edge_type: str, properties: dict[str, Any] | None = None, *, source: str | None = None, confidence: float | None = None, merge_fn: Any = None) -> Edge
```

Create or update an edge matched by its exact set of nodes.

Finds an existing edge with the same nodes, or creates a new one. Useful for idempotent ingestion.

Parameters:

| Name         | Type             | Description                                                                                          | Default                                                 |
| ------------ | ---------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `node_ids`   | `set[str]`       | Set of node IDs for the edge.                                                                        | *required*                                              |
| `edge_type`  | `str`            | Edge type string.                                                                                    | *required*                                              |
| `properties` | \`dict[str, Any] | None\`                                                                                               | Key-value metadata. Merged on update.                   |
| `source`     | \`str            | None\`                                                                                               | Provenance source. Falls back to context or "unknown".  |
| `confidence` | \`float          | None\`                                                                                               | Confidence score 0.0-1.0. Falls back to context or 1.0. |
| `merge_fn`   | `Any`            | Optional callable (existing_props, new_props) -> merged_props for custom property merging on update. | `None`                                                  |

Returns:

| Type   | Description                  |
| ------ | ---------------------------- |
| `Edge` | The created or updated Edge. |

### edges_of_node

```
edges_of_node(node_id: str, *, edge_types: list[str] | None = None) -> list[Edge]
```

Get all edges incident to a node.

Parameters:

| Name         | Type        | Description        | Default                                        |
| ------------ | ----------- | ------------------ | ---------------------------------------------- |
| `node_id`    | `str`       | The node to query. | *required*                                     |
| `edge_types` | \`list[str] | None\`             | If provided, only return edges of these types. |

Returns:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `list[Edge]` | List of edges containing this node. |

### batch

```
batch() -> Generator[None, None, None]
```

Group write operations and save them all at once.

Reduces disk I/O for bulk inserts. Batches can nest; only the outermost batch triggers a save.

Note

Provides batched persistence, **not** transaction rollback. If an exception occurs mid-batch, partial in-memory changes remain and are persisted when the batch exits.

Example

```
with hb.batch():
    for i in range(1000):
        hb.edge([f"entity_{i}", "catalog"], type="belongs_to")
# Single save at the end
```

### stats

```
stats() -> HypergraphStats
```

Get node and edge counts by type.

Returns:

| Type              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `HypergraphStats` | A HypergraphStats with node_count, edge_count, |
| `HypergraphStats` | nodes_by_type, and edges_by_type fields.       |
