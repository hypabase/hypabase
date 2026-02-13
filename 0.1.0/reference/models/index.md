# Models

## hypabase.models.Node

Bases: `BaseModel`

An entity in the hypergraph.

Each node has an ID, a type for classification, and optional key-value properties. Nodes are auto-created when referenced in an edge.

## hypabase.models.Edge

Bases: `BaseModel`

A hyperedge: one relationship linking two or more nodes.

Each edge has a type, provenance (source and confidence), and can carry arbitrary properties. Node order within the edge is preserved.

### node_ids

```
node_ids: list[str]
```

Ordered list of node IDs (backward compat).

### node_set

```
node_set: set[str]
```

Deduplicated set of node IDs.

## hypabase.models.Incidence

Bases: `BaseModel`

How a node or edge participates in a hyperedge.

Each incidence links one node (or one edge reference) to an edge, with an optional direction. Exactly one of node_id or edge_ref_id must be set.

## hypabase.models.HypergraphStats

Bases: `BaseModel`

Summary counts for a hypergraph database.

Reports total node and edge counts, broken down by type.

## hypabase.models.ValidationResult

Bases: `BaseModel`

Result of a hypergraph consistency check.

Contains a pass/fail flag, a list of errors, and a list of warnings found during validation.
