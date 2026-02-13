# Provenance

Every edge in Hypabase carries two provenance fields: `source` and `confidence`. These are first-class parts of the data model, not bolted-on metadata.

## Setting provenance per-edge

```python
hb.edge(
    ["patient_123", "aspirin", "ibuprofen"],
    type="drug_interaction",
    source="clinical_decision_support_v3",
    confidence=0.92,
)
```

If omitted, `source` defaults to `"unknown"` and `confidence` defaults to `1.0`.

## Context manager for bulk provenance

Use `hb.context()` to set default provenance for a block of operations:

```python
with hb.context(source="clinical_records", confidence=0.95):
    hb.edge(
        ["dr_smith", "patient_a", "aspirin", "headache", "mercy_hospital"],
        type="treatment",
    )
    hb.edge(
        ["dr_jones", "patient_b", "ibuprofen", "fever"],
        type="treatment",
    )
# Both edges get source="clinical_records", confidence=0.95
```

### Override within a context

Per-edge values override the context defaults:

```python
with hb.context(source="extraction", confidence=0.8):
    hb.edge(["a", "b"], type="x")                    # confidence=0.8
    hb.edge(["c", "d"], type="y", confidence=0.99)    # confidence=0.99
```

### Nested contexts

Contexts can nest. The innermost context wins:

```python
with hb.context(source="system_a", confidence=0.9):
    hb.edge(["a", "b"], type="x")  # source="system_a"

    with hb.context(source="system_b", confidence=0.7):
        hb.edge(["c", "d"], type="y")  # source="system_b"

    hb.edge(["e", "f"], type="z")  # source="system_a" (restored)
```

## Querying by provenance

### Filter by source

```python
edges = hb.edges(source="clinical_records")
```

### Filter by confidence threshold

```python
high_confidence = hb.edges(min_confidence=0.9)
```

### Combine provenance with other filters

```python
edges = hb.edges(
    containing=["patient_123"],
    source="clinical_records",
    min_confidence=0.9,
)
```

## Aggregating sources

The `sources()` method provides an overview of all provenance sources:

```python
sources = hb.sources()
# [
#     {"source": "clinical_records", "edge_count": 2, "avg_confidence": 0.95},
#     {"source": "lab_results", "edge_count": 1, "avg_confidence": 0.88},
# ]
```

Each entry includes:

- `source` — the source string
- `edge_count` — number of edges from this source
- `avg_confidence` — mean confidence across all edges from this source

## Use cases

### Multi-source knowledge graphs

Track which AI model, document, or human produced each fact:

```python
with hb.context(source="gpt-4o_extraction", confidence=0.85):
    hb.edge(["transformer", "attention", "nlp"], type="concept_link")

with hb.context(source="manual_review", confidence=0.99):
    hb.edge(["transformer", "attention", "nlp"], type="concept_link_verified")
```

### Audit trails

Know exactly which source contributed each relationship:

```python
# What did the legal review say?
legal_edges = hb.edges(source="legal_review")

# What do we trust?
trusted = hb.edges(min_confidence=0.85)

# What's unreliable?
all_sources = hb.sources()
low_quality = [s for s in all_sources if s["avg_confidence"] < 0.7]
```

### Confidence-based retrieval

In RAG pipelines, retrieve only high-confidence relationships:

```python
edges = hb.edges(
    containing=["query_entity"],
    min_confidence=0.8,
)
# Only facts we're confident about end up in the LLM context
```
