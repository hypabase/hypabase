# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-02-27

### Changed

- Rewrote OpenClaw skill listing — benefit-first description, better keywords for discoverability
- Restructured SKILL.md — setup and quick start first, reference tables at the bottom

## [0.2.1] - 2026-02-27

### Fixed

- `__version__` now correctly reports `0.2.1` (was stuck at `0.1.2` in the 0.2.0 package)
- OpenClaw skill uses `uvx` for zero-install MCP server setup (ecosystem standard pattern)
- SKILL.md setup section replaced with `mcpServers` config snippet

## [0.2.0] - 2026-02-27

### Added

- **Memory module** (`hypabase/memory/`) — structured AI agent memory using kāraka semantic roles, neuroscience-informed decay, entity resolution, spreading activation recall, and contradiction detection
- **Memory MCP server** (`hypabase-memory`) — 4 tools: remember, recall, consolidate, forget. PENMAN notation input, provenance-tracked output.
- **OpenClaw skill** (`skills/openclaw/`) — SKILL.md + claw.json for ClawHub discovery
- **Embedding providers** — FastEmbed (default), OpenAI, SentenceTransformers via `HYPABASE_EMBEDDER` env var
- **Vector search** — sqlite-vec backed KNN for semantic entity resolution and recall
- **Temporal queries** — `valid_at` / `expired_at` timestamps, point-in-time queries, soft deletes
- **Access tracking** — `access_log` table for memory strength scoring (recency, frequency, salience)
- **CLI `mcp` command** — `hypabase mcp` starts the Memory MCP server

### Changed

- **MCP server replaced** — old 14-tool general-purpose server replaced with focused 4-tool memory server
- **Python 3.11+ required** — dropped Python 3.10 (onnxruntime 1.24.2 has no 3.10 wheel)
- **All deps are core** — `uv add hypabase` gives CLI, MCP, and embeddings. No optional extras for core functionality.

### Fixed

- Package docs updated to match actual MCP server API (4 tools, PENMAN notation)
- `concepts.md` table count corrected (7 tables + 1 virtual table)
- CLI reference updated (removed `[cli]` extra, added `mcp` command)

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
