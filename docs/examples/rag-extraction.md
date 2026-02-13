# RAG Extraction Pipeline

Build a knowledge graph from document extractions, storing entities and relationships with per-source confidence scores.

## Setup

```python
from hypabase import Hypabase

hb = Hypabase("knowledge.db")
```

## Extract and store

Simulate extracting facts from three documents with different confidence levels:

```python
# High-quality academic paper
with hb.context(source="doc_arxiv_2401", confidence=0.92):
    hb.edge(["transformer", "attention", "nlp"], type="concept_link")
    hb.edge(["bert", "transformer", "pretraining"], type="builds_on")

# Blog post — lower confidence
with hb.context(source="doc_blog_post", confidence=0.75):
    hb.edge(["transformer", "gpu", "training"], type="requires")
    hb.edge(["attention", "memory", "scaling"], type="tradeoff")

# Textbook with moderate confidence
with hb.context(source="doc_textbook_ch5", confidence=0.5):
    hb.edge(["rnn", "lstm", "attention"], type="evolution")
```

Each extraction batch gets its own source and confidence. The provenance context manager handles this cleanly.

## Query patterns

### Entity retrieval

Find all relationships involving a concept:

```python
edges = hb.edges(containing=["transformer"])
# Returns 3 edges: concept_link, builds_on, requires
```

### Source filtering

Retrieve facts from a specific document:

```python
edges = hb.edges(source="doc_arxiv_2401")
# Returns 2 edges from the arxiv paper
```

### Confidence-based retrieval

Only include high-quality extractions in your RAG context:

```python
high_quality = hb.edges(min_confidence=0.8)
# Returns 2 edges (arxiv paper), excludes blog post and textbook
```

### Multi-hop discovery

Find paths between concepts across documents:

```python
paths = hb.paths("bert", "nlp")
# bert → transformer → nlp (across two extraction sources)
```

### N-ary fact preservation

A single edge stores the 3-way concept link:

```python
concept_links = hb.edges(type="concept_link")
assert len(concept_links[0].node_ids) == 3
# ["transformer", "attention", "nlp"] — not three separate pairs
```

## Integration with LLM extraction

A typical pipeline:

```python
import json

def extract_and_store(document_id, text, hb):
    """Extract facts from text using an LLM and store in Hypabase."""
    # Your LLM extraction logic here
    # Returns: [{"entities": [...], "type": "...", "confidence": ...}, ...]
    extractions = llm_extract(text)

    with hb.context(source=document_id, confidence=0.85):
        with hb.batch():  # Single save for all extractions
            for fact in extractions:
                hb.edge(
                    fact["entities"],
                    type=fact["type"],
                    confidence=fact.get("confidence"),  # Override if LLM provides per-fact score
                )
```

## RAG retrieval function

```python
def retrieve_context(query_entities, hb, min_confidence=0.7):
    """Retrieve structured relationships for RAG context."""
    edges = hb.edges(
        containing=query_entities,
        min_confidence=min_confidence,
    )
    # Format for LLM context
    facts = []
    for e in edges:
        facts.append(
            f"{e.type}: {' + '.join(e.node_ids)} "
            f"(source={e.source}, confidence={e.confidence})"
        )
    return "\n".join(facts)
```

This gives your LLM structured, provenance-tracked relationships as context.
