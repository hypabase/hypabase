# Hypabase vs Neo4j

## The core difference

Neo4j is a property graph database. Every edge connects exactly two nodes. When your data has relationships between 3+ entities, Neo4j forces you to decompose them.

Hypabase is a hypergraph database. A single edge connects any number of nodes.

## Modeling n-ary relationships

**The fact**: "Dr. Smith treated Patient 123 with Aspirin for Headache at Mercy Hospital"

### Neo4j

Neo4j edges connect exactly two nodes. To model a 5-entity relationship, you use an intermediate node (reification pattern):

```
CREATE (t:Treatment)
CREATE (dr_smith)-[:TREATS]->(t)
CREATE (t)-[:PATIENT]->(patient_123)
CREATE (t)-[:MEDICATION]->(aspirin)
CREATE (t)-[:CONDITION]->(headache)
CREATE (t)-[:LOCATION]->(mercy_hospital)
```

### Hypabase

Hypabase edges connect any number of nodes directly:

```
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)
```

## Comparison

|                         | Neo4j                                    | Hypabase                           |
| ----------------------- | ---------------------------------------- | ---------------------------------- |
| **Edge model**          | Binary (2 nodes per edge)                | N-ary (2+ nodes per edge)          |
| **N-ary relationships** | Reification pattern (intermediate nodes) | Native hyperedges                  |
| **Provenance**          | Custom properties (no standard)          | Built-in `source` and `confidence` |
| **Query language**      | Cypher                                   | Python SDK (no query language)     |
| **Setup**               | Server process, Docker, or Aura cloud    | `uv add hypabase` — zero config    |
| **Storage**             | Custom binary format                     | SQLite (local-first)               |
| **Vertex-set lookup**   | Multi-hop traversal                      | O(1) hash index                    |
| **Visualization**       | Neo4j Browser, Bloom                     | None (library, not platform)       |
| **Community**           | Large, established                       | New                                |

## When to use Neo4j instead

- You only have pairwise relationships
- You need Cypher's query expressiveness for complex graph patterns
- You need a managed cloud service (Neo4j Aura)
- You need built-in visualization (Neo4j Browser, Bloom)
- Your team already knows Neo4j and Cypher

## When to use Hypabase instead

- Your relationships connect 3+ entities
- You need provenance tracking (source, confidence) as part of the data model
- You want zero-config local-first storage
- You're building for AI agents or LLM pipelines (SDK-only API, no query language)
- You want `uv add` and go, not a server process

## Code comparison: patient lookup

### Neo4j

```
MATCH (p:Patient {id: 'patient_123'})-[:PATIENT]-(t:Treatment)
MATCH (t)-[:TREATS]-(d:Doctor)
MATCH (t)-[:MEDICATION]-(m:Medication)
MATCH (t)-[:CONDITION]-(c:Condition)
RETURN d, m, c
```

### Hypabase

```
edges = hb.edges(containing=["patient_123"], type="treatment")
# Each edge contains all connected entities directly
```
