# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-13

### Added

- **Hypergraph core** — native n-ary hyperedges connecting 2+ nodes atomically
- **Provenance-native** — every edge carries `source` and `confidence` (0.0-1.0)
- **Provenance context** — `hb.context(source=..., confidence=...)` for bulk provenance
- **O(1) vertex-set lookup** — SHA-256 hash index for instant exact-match queries
- **SQLite persistence** — WAL mode, foreign keys, zero-config local-first storage
- **Namespace isolation** — `hb.database("name")` for scoped views in a single file
- **Traversal** — `neighbors()`, `paths()`, `find_paths()` with edge-type filters
- **Graph metrics** — `node_degree()`, `edge_cardinality()`, `hyperedge_degree()`
- **Batch operations** — `hb.batch()` defers auto-persist for bulk inserts
- **HIF import/export** — Hypergraph Interchange Format support
- **Validation** — `hb.validate()` for internal consistency checks
- **Upsert by vertex set** — idempotent edge creation for repeated ingestion
- **MCP server** — 14 tools + 2 resources via FastMCP (stdio transport)
- **CLI** — `hypabase init`, `node`, `edge`, `query`, `stats`, `validate`, `export-hif`, `import-hif`, `mcp`
- **435+ tests** across client API, engine core, MCP server, HIF, threading, and use cases

### Known Limitations

- Local SQLite only (cloud backends planned for Phase 3)
- `batch()` provides batched persistence, not transaction rollback
- `delete_node_cascade()` is deprecated; use `delete_node(id, cascade=True)`
