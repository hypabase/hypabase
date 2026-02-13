# Hybrid Vector Pattern

Combine Hypabase (structured relationships) with a vector database (semantic similarity).

## When to use this pattern

- You need both semantic search ("find documents about GDPR") and structured queries ("which entities connect to regulation_gdpr?")
- Your RAG pipeline needs to retrieve related entities, not only similar text chunks
- You want provenance-tracked relationships alongside vector similarity scores

## Architecture

```
Query → Vector DB (semantic retrieval) → entity IDs
     → Hypabase (structured relationships) → connected entities
     → Combine both → LLM context
```

The vector database finds *what's relevant*. Hypabase finds *what's connected*.

## Example: Legal document analysis

### Step 1: Store extractions in both systems

```python
from hypabase import Hypabase

hb = Hypabase("legal_kg.db")

# After LLM extracts entities and relationships from documents:
with hb.context(source="doc_gdpr_analysis", confidence=0.9):
    hb.edge(
        ["regulation_gdpr", "company_techcorp", "violation_data_breach"],
        type="enforcement_action",
    )
    hb.edge(
        ["regulation_gdpr", "right_to_erasure", "article_17"],
        type="defines",
    )
    hb.edge(
        ["company_techcorp", "fine_20m", "year_2024"],
        type="penalty",
    )

# Meanwhile, store document chunks in your vector DB:
# vector_db.upsert(chunks, embeddings)
```

### Step 2: Hybrid retrieval

```python
def hybrid_retrieve(query_text, hb, vector_db, min_confidence=0.7):
    """Combine vector search with structured graph queries."""

    # 1. Vector search for semantic retrieval
    similar_docs = vector_db.search(query_text, top_k=10)
    doc_entity_ids = extract_entity_ids(similar_docs)

    # 2. Hypabase for structured multi-entity queries
    edges = hb.edges(
        containing=doc_entity_ids,
        min_confidence=min_confidence,
    )

    # 3. Expand context with graph neighbors
    all_entities = set()
    for e in edges:
        all_entities.update(e.node_ids)

    neighbor_edges = []
    for entity in all_entities:
        neighbor_edges.extend(
            hb.edges_of_node(entity, edge_types=["defines", "enforcement_action"])
        )

    return {
        "vector_results": similar_docs,
        "graph_relationships": edges,
        "expanded_context": neighbor_edges,
    }
```

### Step 3: Build LLM context

```python
def build_context(retrieval_results):
    """Format hybrid results for LLM consumption."""
    parts = []

    # Structured relationships
    parts.append("Known relationships:")
    for e in retrieval_results["graph_relationships"]:
        parts.append(
            f"  {e.type}: {' + '.join(e.node_ids)} "
            f"(confidence={e.confidence})"
        )

    # Relevant text passages
    parts.append("\nRelevant passages:")
    for doc in retrieval_results["vector_results"]:
        parts.append(f"  {doc['text'][:200]}...")

    return "\n".join(parts)
```

## What each system provides

**Vector database:** semantic similarity search, fuzzy natural language queries, embedding-based ranking.

**Hypabase:** structured relationship queries, multi-hop traversal, provenance filtering, n-ary facts.

The hybrid pattern combines both — semantic retrieval to find relevant entities, then structured queries to expand context with connected relationships and provenance.

## Compatible vector databases

Any vector database works with this pattern:

- **ChromaDB** — local-first, Python-native (good match for Hypabase's local-first model)
- **Qdrant** — high-performance, supports filtering
- **Weaviate** — hybrid search built-in
- **Pinecone** — managed cloud service
- **pgvector** — PostgreSQL extension
