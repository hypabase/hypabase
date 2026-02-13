# Concepts

## What is a hypergraph?

A **hypergraph** generalizes a graph by allowing edges to connect any number of nodes, not only two. In a regular graph, an edge connects exactly two nodes (a pair). In a hypergraph, a single **hyperedge** can connect 2, 3, 5, or more nodes at once.

This matters because real-world facts are often n-ary:

- "Dr. Smith treated Patient 123 with Aspirin for a Headache at Mercy Hospital" — 5 entities, one fact
- "The board approved a $5M budget for APAC expansion into Japan and Korea in Q3" — 6 entities, one decision
- "BERT builds on the Transformer architecture using pretraining" — 3 entities, one relationship

A hypergraph represents these directly. Each example above is a single hyperedge.

### Hyperedges vs binary edges

In a standard property graph (e.g., Neo4j), edges connect exactly two nodes. To model the board decision, you'd introduce an intermediate node:

```
(d:Decision)
(board)-[:DECIDED]->(d)
(d)-[:BUDGET]->(budget_5m)
(d)-[:REGION]->(apac)
(d)-[:COUNTRY]->(japan)
(d)-[:COUNTRY]->(korea)
(d)-[:TIMELINE]->(q3)
```

That's 6 binary edges and an intermediate node representing the decision.

In Hypabase, a single hyperedge connects all participants:

```
hb.edge(
    ["board", "budget_5m", "apac", "japan", "korea", "q3"],
    type="budget_approval",
)
```

## Nodes

A node represents an entity. Every node has:

- **`id`** — unique string identifier (e.g., `"dr_smith"`, `"patient_123"`)
- **`type`** — classification string (e.g., `"doctor"`, `"patient"`, `"medication"`)
- **`properties`** — arbitrary key-value metadata

Nodes are auto-created when referenced in an edge. If you create an edge referencing `"aspirin"` and no node with that ID exists, Hypabase creates it with `type="unknown"`. See [Getting Started](https://docs.hypabase.app/latest/getting-started/#create-nodes) for usage.

## Edges (hyperedges)

An edge represents a relationship between 2 or more nodes. Every edge has:

- **`id`** — unique identifier (auto-generated UUID if not specified)
- **`type`** — relationship type (e.g., `"treatment"`, `"concept_link"`)
- **`incidences`** — ordered list of node participations
- **`directed`** — whether the edge has direction (tail/head semantics)
- **`source`** — provenance source string
- **`confidence`** — confidence score (0.0 to 1.0)
- **`properties`** — arbitrary key-value metadata

See [Getting Started](https://docs.hypabase.app/latest/getting-started/#create-a-hyperedge) for usage.

### Node order

The `position` column in the incidence table preserves node order. The order you pass nodes is the order they're stored. This matters for directed edges and for domain-specific semantics where position carries meaning.

### Directed edges

When `directed=True`, the first node is the **tail** and the last node is the **head**:

```
hb.edge(
    ["cause", "intermediate", "effect"],
    type="causal_chain",
    directed=True,
)
```

## Provenance

Every edge carries two provenance fields:

- **`source`** — a string identifying where the relationship came from (e.g., `"clinical_records"`, `"gpt-4o_extraction"`, `"user_input"`)
- **`confidence`** — a float between 0.0 and 1.0 representing certainty

Provenance is not bolted-on metadata — it's part of the core data model. This enables:

- Filtering edges by source or confidence threshold
- Aggregating reliability across sources
- Tracking which AI model or human produced each fact
- Building audit trails

See the [Provenance guide](https://docs.hypabase.app/latest/guides/provenance/index.md) for context managers, overrides, and querying.

## Vertex-set lookup

Hypabase maintains a SHA-256 hash index over the node sets of all edges. This enables **O(1) exact vertex-set lookup** — given a set of node IDs, find all edges that connect exactly those nodes (order-independent):

```
edges = hb.edges_by_vertex_set(["dr_smith", "patient_123", "aspirin"])
```

The query answers: "does a relationship connect exactly these entities?"

## Storage

Hypabase uses SQLite with WAL mode and foreign keys enabled. The database has four tables:

| Table              | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `nodes`            | Entity storage (id, type, properties)                             |
| `edges`            | Relationship metadata (id, type, source, confidence, properties)  |
| `incidences`       | Junction table linking edges to nodes with position and direction |
| `vertex_set_index` | SHA-256 hash index for O(1) exact vertex-set lookup               |

The storage engine encapsulates all SQL. The client API never exposes raw queries.
