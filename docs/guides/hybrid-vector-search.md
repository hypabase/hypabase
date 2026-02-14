# Hybrid Vector Search

Vector databases and hypergraph libraries solve different problems. Vector search finds things that are semantically similar. Hypabase finds things that are structurally connected. Combining them gives you both.

## Why combine them

Vector databases (Pinecone, Qdrant, Weaviate, ChromaDB, pgvector) store embeddings and answer "what resembles X?" They're good at fuzzy semantic retrieval — finding documents about a topic even when the exact words differ.

Hypabase stores explicit relationships and answers "what's connected to X, through which relationships, with what provenance?" It's good at structured traversal — finding the entities connected to a given entity and the chain of relationships between them.

Neither one replaces the other. A vector search can find relevant entities, and a hypergraph traversal can find what those entities are connected to. Together, they give an LLM richer context than either could alone.

## The pattern

```
Query → Vector DB → semantically relevant entity IDs
     → Hypabase  → structurally connected entities + provenance
     → Merge     → enriched context for the LLM
```

1. **Vector retrieval** — embed the query and find the top-k similar entities or chunks
2. **Hypergraph expansion** — take those entity IDs and query Hypabase for their neighbors, shared edges, and paths
3. **Provenance filtering** — use `source` and `min_confidence` to keep only trusted relationships
4. **Combine** — merge both result sets into the LLM prompt

This works because vector search casts a wide net (fuzzy, semantic), and the hypergraph narrows it down to structured, provenance-tracked facts.

## Comparison

| Capability | Vector DB | Hypabase |
|-----------|-----------|----------|
| Semantic similarity search | Yes | No |
| Structured relationships | No | Yes |
| Multi-hop traversal | No | Yes |
| N-ary facts (3+ entities) | No | Yes |
| Provenance tracking | No | Yes |
| Fuzzy natural language queries | Yes | No |
| Confidence-based filtering | No | Yes |

## When to use this

- RAG pipelines where you need both semantic retrieval and structured relationship context
- Knowledge systems where entities have both text descriptions (vectorizable) and explicit connections (graph-queryable)
- Any application where "related to" (semantic) and "connected to" (structural) are both useful signals

## Implementation

See the [Hybrid Vector Pattern example](../examples/hybrid-vector.md) for a complete worked implementation with code.
