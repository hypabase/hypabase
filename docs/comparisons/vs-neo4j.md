# Hypabase vs Neo4j

## The core difference

Neo4j is a property graph database. Every edge connects exactly two nodes. When your data has relationships between 3+ entities, Neo4j requires you to decompose them using intermediate nodes (the reification pattern).

Hypabase is a hypergraph library. A single edge connects any number of nodes directly.

## Modeling n-ary relationships

**The fact**: "Dr. Smith treated Patient 123 with Aspirin for Headache at Mercy Hospital"

### Neo4j

Neo4j edges connect exactly two nodes. To model a 5-entity relationship, you create an intermediate node:

```cypher
CREATE (t:Treatment)
CREATE (dr_smith)-[:TREATS]->(t)
CREATE (t)-[:PATIENT]->(patient_123)
CREATE (t)-[:MEDICATION]->(aspirin)
CREATE (t)-[:CONDITION]->(headache)
CREATE (t)-[:LOCATION]->(mercy_hospital)
```

### Hypabase

```python
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)
```

## Comparison

| | Neo4j | Hypabase |
|---|---|---|
| **Edge model** | Binary (2 nodes per edge) | N-ary (2+ nodes per edge) |
| **N-ary relationships** | Reification pattern | Native hyperedges |
| **Query language** | Cypher | Python SDK |
| **Provenance** | Custom properties | Built-in `source` and `confidence` |
| **Setup** | Server process or cloud | `uv add hypabase` |
| **Storage** | Custom binary format | SQLite |
| **Visualization** | Neo4j Browser, Bloom | None |
| **Drivers** | Python, Java, JS, .NET, Go | Python only |
| **Data size** | Disk-backed, scales to billions | In-memory, limited by RAM |
| **Concurrency** | Multi-user, ACID transactions | Single-process |

## When to use Neo4j

- You have pairwise relationships and Cypher's pattern matching fits your queries
- You need a declarative query language — Cypher is genuinely powerful for complex graph patterns
- You need concurrent multi-user access with ACID transactions
- Your data exceeds available memory
- You need built-in visualization
- You want managed cloud deployment (Neo4j Aura)
- You need drivers in multiple languages

## When to use Hypabase

- Your relationships connect 3+ entities and you want them to stay atomic
- You need provenance tracking as part of the data model, not an afterthought
- You want zero-config embedded storage with no server to manage
- Your data fits in memory and you want fast in-process access
- You're integrating with AI agents via MCP

## Code comparison: patient lookup

### Neo4j

```cypher
MATCH (p:Patient {id: 'patient_123'})-[:PATIENT]-(t:Treatment)
MATCH (t)-[:TREATS]-(d:Doctor)
MATCH (t)-[:MEDICATION]-(m:Medication)
MATCH (t)-[:CONDITION]-(c:Condition)
RETURN d, m, c
```

### Hypabase

```python
edges = hb.edges(containing=["patient_123"], type="treatment")
# Each edge contains all connected entities directly
```
