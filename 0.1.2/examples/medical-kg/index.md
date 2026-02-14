# Medical Knowledge Graph

Build a clinical knowledge graph where treatment events are single edges.

A treatment event connects a doctor, patient, medication, condition, and location. This example builds a graph of such events and shows query patterns.

## Setup

```
from hypabase import Hypabase

hb = Hypabase("clinical.db")
```

## Build the graph

```
# Create typed nodes
hb.node("dr_smith", type="doctor")
hb.node("dr_jones", type="doctor")
hb.node("patient_a", type="patient")
hb.node("patient_b", type="patient")
hb.node("aspirin", type="medication")
hb.node("ibuprofen", type="medication")
hb.node("headache", type="condition")
hb.node("fever", type="condition")
hb.node("mercy_hospital", type="hospital")

# Record treatments with provenance
with hb.context(source="clinical_records", confidence=0.95):
    hb.edge(
        ["dr_smith", "patient_a", "aspirin", "headache", "mercy_hospital"],
        type="treatment",
    )
    hb.edge(
        ["dr_jones", "patient_b", "ibuprofen", "fever"],
        type="treatment",
    )

# Record diagnosis from a different source
with hb.context(source="lab_results", confidence=0.88):
    hb.edge(
        ["dr_smith", "patient_a", "headache"],
        type="diagnosis",
    )
```

## Query patterns

### Patient lookup

Find all edges involving a patient:

```
edges = hb.edges(containing=["patient_a"])
# Returns: treatment edge + diagnosis edge
```

### Provenance filtering

Retrieve only high-confidence relationships:

```
high_conf = hb.edges(min_confidence=0.9)
# Returns: both treatment edges (0.95), excludes diagnosis (0.88)
```

### Path finding

Discover how entities connect:

```
paths = hb.paths("dr_smith", "mercy_hospital")
# [["dr_smith", "patient_a", "mercy_hospital"], ...]
```

### N-ary preservation check

Verify that a single edge stores the 5-entity treatment:

```
treatments = hb.edges(type="treatment")
five_node = [e for e in treatments if len(e.node_ids) == 5]
assert len(five_node) == 1
assert set(five_node[0].node_ids) == {
    "dr_smith", "patient_a", "aspirin", "headache", "mercy_hospital"
}
```

### Source overview

Audit which sources contributed what:

```
sources = hb.sources()
# [
#     {"source": "clinical_records", "edge_count": 2, "avg_confidence": 0.95},
#     {"source": "lab_results", "edge_count": 1, "avg_confidence": 0.88},
# ]
```
