# Building Knowledge Graphs

A knowledge graph is a structured collection of entities and the relationships between them. Hypabase is a natural fit for building knowledge graphs because hyperedges let you represent complex relationships without decomposing them into pairs, and provenance tracking tells you where each fact came from.

## Modeling entities and relationships

In Hypabase, entities are **nodes** and relationships are **edges** (hyperedges). Both carry a `type` for classification and optional `properties` for metadata.

```python
from hypabase import Hypabase

hb = Hypabase("knowledge.db")

# Create typed entities
hb.node("aspirin", type="drug", dosage_form="tablet")
hb.node("ibuprofen", type="drug", dosage_form="tablet")
hb.node("headache", type="condition")
hb.node("dr_smith", type="doctor", specialty="neurology")
hb.node("patient_123", type="patient")
```

Nodes are also auto-created when you reference them in an edge, so you can skip explicit node creation and let the graph grow organically:

```python
# These nodes are created automatically if they don't exist
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache"],
    type="treatment",
    source="clinical_records",
    confidence=0.95,
)
```

## Provenance: tracking where facts come from

Knowledge graphs often combine facts from many sources — manual entry, LLM extraction, APIs, databases, sensor data. Provenance lets you track the origin and reliability of each relationship.

```python
# Facts from clinical records (high confidence)
with hb.context(source="clinical_records", confidence=0.95):
    hb.edge(["aspirin", "headache"], type="treats")
    hb.edge(["ibuprofen", "headache"], type="treats")

# Facts extracted by an LLM (lower confidence)
with hb.context(source="llm_extraction_gpt4", confidence=0.7):
    hb.edge(["aspirin", "ibuprofen"], type="drug_interaction")

# Facts from a structured database (high confidence)
with hb.context(source="drugbank_api", confidence=0.99):
    hb.edge(["aspirin", "ibuprofen", "gi_bleeding"], type="combined_risk")
```

Later, you can filter by provenance:

```python
# Only facts from clinical records
hb.edges(source="clinical_records")

# Only high-confidence facts
hb.edges(min_confidence=0.9)

# See all sources and their stats
hb.sources()
# [{'source': 'clinical_records', 'edge_count': 2, 'avg_confidence': 0.95},
#  {'source': 'llm_extraction_gpt4', 'edge_count': 1, 'avg_confidence': 0.7},
#  {'source': 'drugbank_api', 'edge_count': 1, 'avg_confidence': 0.99}]
```

## Why hyperedges matter for knowledge graphs

Traditional knowledge graphs use triples: `(subject, predicate, object)`. This works for binary facts like "aspirin treats headache." But many facts involve more than two entities.

**A clinical event**: "Dr. Smith prescribed aspirin to Patient 123 for a headache at Mercy Hospital on 2024-01-15."

With triples, you'd split this into binary facts, and the fact that they belong to one event becomes an inference, not a structure. A hyperedge stores this natively:

```python
hb.edge(
    ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
    type="prescription",
    source="ehr_system",
    confidence=0.99,
    properties={"date": "2024-01-15"},
)
```

All five entities are connected by a single edge. Querying for any one of them returns the full context.

## Querying the knowledge graph

```python
# Find all treatments involving a patient
hb.edges(containing=["patient_123"], type="treatment")

# Find all edges connecting two specific entities
hb.edges(containing=["aspirin", "ibuprofen"], match_all=True)

# Find the exact edge connecting a specific set of entities
hb.edges_by_vertex_set(["aspirin", "ibuprofen", "gi_bleeding"])

# Find how two entities are connected
hb.paths("dr_smith", "mercy_hospital")

# Find all neighbors of an entity
hb.neighbors("aspirin")
```

## Organizing with namespaces

For larger knowledge graphs, use namespaces to isolate different domains or data sources within a single file:

```python
hb = Hypabase("knowledge.db")

drugs = hb.database("drugs")
clinical = hb.database("clinical")

drugs.edge(["aspirin", "ibuprofen"], type="interaction", source="drugbank")
clinical.edge(["dr_smith", "patient_123", "aspirin"], type="prescription", source="ehr")

# Each namespace is independent
drugs.stats()     # only drug relationships
clinical.stats()  # only clinical relationships
```

## Next steps

- [Medical Knowledge Graph example](../examples/medical-kg.md) — a complete worked example with clinical data
- [RAG Extraction Pipeline example](../examples/rag-extraction.md) — extracting relationships from documents into a knowledge graph
- [Provenance guide](provenance.md) — deeper dive into provenance tracking
- [Batch Operations guide](batch-operations.md) — efficient bulk ingestion for larger graphs
