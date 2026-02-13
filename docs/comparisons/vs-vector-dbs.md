# Hypabase vs Vector Databases

## Different tools, different jobs

Vector databases (Pinecone, Qdrant, Weaviate, ChromaDB, pgvector) store embeddings and find matches. They answer "what resembles X?"

Hypabase stores relationships and finds connections. It answers "what's connected to X, through which relationships, with what provenance?"

Vector databases and Hypabase complement each other.

## What vector databases do

Vector databases store embeddings and retrieve by similarity. They excel at semantic search ("find documents about GDPR"), fuzzy natural language queries, and ranking by embedding distance.

## What Hypabase does

Hypabase stores explicit relationships between entities and retrieves by structure. It provides multi-entity edges, multi-hop traversal, provenance tracking (`source` and `confidence`), and exact vertex-set lookup.

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

## The hybrid pattern

For RAG and knowledge systems, a strong architecture combines both:

1. **Vector DB** for initial semantic retrieval — find relevant documents/chunks
2. **Hypabase** for structured relationship queries — find connected entities with provenance
3. **Combine** both contexts for the LLM

See the [Hybrid Vector Pattern](../examples/hybrid-vector.md) for a complete implementation with code.

## Related research

HyperGraphRAG (NeurIPS 2025) studied n-ary retrieval vs binary graph retrieval across medicine, agriculture, computer science, and law.
