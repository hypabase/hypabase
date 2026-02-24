# Scaling Roadmap

Hypabase is a Python hypergraph library backed by SQLite. This document describes the architecture path from its current single-process design to a disk-first, backend-agnostic engine capable of scaling to billions of edges.

Each phase is additive. Later phases don't rewrite earlier ones — they add new `StorageEngine` implementations behind the same trait.

---

## Current Architecture

```
Python (hypabase)
├── client.py                Hypabase class — public API
├── models.py                Pydantic models (Node, Edge, HypergraphStats)
├── engine/
│   ├── core.py              HypergraphCore — in-memory indexes (7 Python dicts)
│   ├── storage.py           SQLiteStorage — all SQL, schema, and queries
│   ├── embeddings.py        EmbeddingProvider ABC + OpenAI/SentenceTransformer impls
│   ├── vector.py            Pure-Python vector math (cosine similarity, pack/unpack)
│   └── persistence_engine.py  PersistenceEngine abstract interface
├── memory/
│   ├── agent.py             Memory — remember/recall/forget/consolidate/connections
│   ├── extraction.py        Regex-based entity extraction
│   ├── resolution.py        EntityResolver — normalization, alias, embedding match
│   ├── strength.py          Memory strength scoring (recency, frequency, salience)
│   └── types.py             Literal types: KarakaRole, MemoryType, RecallStrategy
├── cli/
│   └── main.py              Click CLI commands (init, node, edge, query, stats)
└── mcp/
    └── server.py            MCP server (FastMCP, stdio transport)
```

**How it works today:** On startup, the entire graph loads from SQLite into Python dicts. Reads are dict lookups. Writes mutate dicts and mirror to SQLite. The graph lives in RAM; SQLite is a persistence side effect.

**Current operating envelope:**

| Dimension | Comfortable | Ceiling |
|-----------|-------------|---------|
| Nodes + Edges | 50K–100K | ~500K |
| Memories | ~100 | ~1K |
| Recall latency | <2s at 100 memories | Degrades linearly |
| Writes/sec | 1–5K | SQLite single-writer limit |
| Memory footprint | O(graph size) | ~200 bytes/node, ~500 bytes/edge (estimated, not benchmarked) |

**What limits scale:**

1. **Memory-first design.** The entire graph must fit in RAM as Python objects.
2. **Python object overhead.** The core engine (`core.py`) uses `@dataclass` internally; the public API (`models.py`) uses Pydantic `BaseModel`. Both carry `dict[str, Any]` properties and GC headers. Pydantic models are 4–5× larger than raw data; core dataclasses roughly 2–3×.
3. **Global RLock.** All operations (reads and writes) serialize through one lock.
4. **Per-write SQLite commits.** Each `write_node`/`write_edge` auto-commits outside a `batch()` context.
5. **Spreading activation BFS.** Uses a `visited_nodes` set and `min_activation` threshold to prune, bounded by a `depth` limit (default 2). Worst case is O(V+E) within the depth bound. The concern is not algorithmic complexity but the per-visit constant: each node triggers a Python-level `edges_of_node()` call returning full model objects.
6. **Embedding dimension lock.** `SQLiteStorage` locks the embedding vector dimension on first use. All embeddings must share the same dimension. Switching embedding models (e.g., 384-dim to 1536-dim) requires deleting all existing embeddings and rebuilding.

---

## Phase 0: Python Optimizations

*No Rust. No new dependencies. Improvements within the current architecture.*

### 0a. RWLock on HypergraphCore

Replace the single `RLock` with a read-write lock. Recalls are pure reads — they don't mutate the graph. With a RWLock, unlimited concurrent reads run in parallel while writes still serialize.

**Impact:** N× concurrent read throughput (N = reader threads).

### 0b. Push filters to SQL in recall

The filter-only recall path (no entity provided) calls `hb.edges(active=True)`, which reads all edges from the in-memory `HypergraphCore` dicts and filters with `filter_temporal()` in a Python loop. The optimization: for filter-only recall, bypass the in-memory engine and route directly to `SQLiteStorage` SQL queries. Some filter dimensions have SQL indexes (`edges.type` for action, `expired_at`/`valid_at`/`created_at` for temporal filters) and can be pushed to WHERE clauses. However, `memory_type` and `mood` are stored in the `properties` JSON column and have no SQL index — these require `json_extract()` calls or generated columns.

**Impact:** Routing filter-only recall to SQL takes it from O(E) Python objects to O(matching) SQL rows. JSON property filters can still be applied in SQL via `json_extract()`, avoiding the Python object overhead even without an index.

### 0c. Budget-bounded spreading activation

The current BFS already prunes via `min_activation` threshold and a `visited_nodes` set, bounding traversal to O(V+E) within the depth limit. The proposal: replace the FIFO deque with a max-heap priority queue and an explicit node-visit budget. Best-first search explores highest-activation paths first; the budget cap guarantees a fixed upper bound on work regardless of graph density.

**Impact:** Recall becomes O(budget) instead of O(reachable subgraph within depth). The budget makes latency predictable under worst-case topologies (e.g., dense hub nodes). The single highest-ROI change for the memory module.

### 0d. Write coalescing

Wrap multi-write operations (like `remember()`, which creates nodes + edges + same_as links) in `hb.batch()`. Note: `batch()` operates at two levels. On the core engine (`engine/core.py`), `batch()` holds the `RLock` for isolation — it is not a SQLite transaction. On the client (`client.py`), `hb.batch()` does both: holds the core lock AND wraps storage writes in a `begin()`/`commit()` transaction with rollback on failure. Only the client-level `hb.batch()` eliminates per-operation SQLite commits.

Currently `remember()` makes N+2 separate auto-committed writes (N entity nodes + 1 edge + 1 embedding). Wrapping in `hb.batch()` coalesces these into a single commit.

**Impact:** `remember()` latency drops 4–8× (one commit instead of 4–8).

### 0e. Incremental consolidation index

Maintain a co-occurrence counter as a side effect of `remember()` instead of computing all pairwise counts in `consolidate()`. The current O(E × k²) full scan becomes an O(1) read of pre-computed counts.

**Impact:** `consolidate()` scales to 100K+ memories.

### 0f. Entity resolution cache scaling

`EntityResolver.warm_cache()` loads all nodes linearly and builds a name-to-ID lookup dict. At scale, this has two costs:

1. **Startup cost:** `warm_cache()` calls `hb.nodes()` to iterate all nodes — O(N) on every `Memory` initialization.
2. **Per-resolve embedding search:** When the cache misses and an embedder is available, `resolve()` performs a vector similarity search per entity. With many unique entities, this becomes a bottleneck during `remember()`.

**Mitigation:** Lazy cache population (resolve on demand, not at startup) and batched embedding lookups for multi-entity `remember()` calls.

### Envelope after Phase 0

| Dimension | Before | After |
|-----------|--------|-------|
| Memories | ~1K | ~50K |
| Recall latency (10K memories) | Seconds | <50ms |
| Concurrent reads | 1 (serialized) | N (parallel) |
| `remember()` latency | ~10–20ms | ~2–5ms |

*Envelope figures are projections based on overhead analysis, not measured benchmarks.*

---

## Phase 1: Rust Core Engine

*Port HypergraphCore and spreading activation to Rust via PyO3. Python API unchanged.*

### Why Rust

- **Memory:** A Rust `Hyperedge` struct is ~120 bytes. The Python equivalent is ~500+ bytes with Pydantic/GC overhead. 4–5× denser.
- **No GIL:** `parking_lot::RwLock` enables true parallel reads across cores. Python's RLock serializes everything.
- **Cache locality:** Rust structs are contiguous in memory. BFS over a `Vec<Hyperedge>` hits L1/L2 cache. Python dict lookups chase pointers across the heap.
- **Zero-cost abstractions:** Set intersection for N-ary overlap scoring compiles to tight loops, not Python object allocation.

### What gets ported

```
hypabase-core (Rust crate, exposed via PyO3)
├── model.rs           Node, Hyperedge, Incidence structs
├── graph.rs           7 index HashMaps, traversal, vertex-set lookup
├── activation.rs      BFS, N-ary overlap, role-weighted propagation
└── python/lib.rs      PyO3 bindings — same method signatures
```

### What stays in Python

- `client.py` — calls Rust core instead of Python core
- `memory/agent.py` — calls Rust activation instead of Python BFS
- `mcp/server.py` — unchanged
- `engine/storage.py` — SQLite persistence, unchanged for now

### Key crates

| Component | Crate | Role |
|-----------|-------|------|
| Python bindings | `pyo3` + `maturin` | Expose Rust types as Python classes |
| Fast hashing | `rustc-hash` (FxHashMap) | 2–5× faster than std HashMap for string keys |
| Serialization | `serde` + `serde_json` | Properties are JSON — serde handles natively |
| Concurrency | `parking_lot::RwLock` | Faster RwLock than std. Read-parallel, write-exclusive. |

### Migration path

The swap is one import:

```python
# Before
from hypabase.engine.core import HypergraphCore

# After
from hypabase_core import HypergraphCore  # Rust, same API
```

All existing tests pass against the Rust core because the API surface is identical.

### Serialization boundary

The import swap is the goal, but the Python↔Rust boundary has friction points:

- **Two model layers.** The core engine uses `@dataclass` types (`engine/core.py`). The public API uses Pydantic `BaseModel` (`models.py`). The Rust port replaces the dataclass layer; Pydantic models stay in Python. `client.py` already converts between them (`_core_node_to_model`, `_core_edge_to_model`).
- **`dict[str, Any]` properties.** Node and edge properties are arbitrary Python dicts. PyO3 converts these to/from `serde_json::Value`, but nested Python objects (e.g., `datetime`, custom types) require explicit handling or will fail at the boundary.
- **`frozenset` keys.** The vertex-set index uses `frozenset[str]` as dict keys. Rust has no direct equivalent; use `BTreeSet<String>` or a sorted-string hash key.
- **Pickle support.** `HypergraphCore` implements `__getstate__`/`__setstate__` for pickle/deepcopy. The Rust type needs PyO3 equivalents or an alternative serialization path.

### Envelope after Phase 1

| Dimension | Before | After |
|-----------|--------|-------|
| Nodes + Edges | ~500K | ~5M |
| Memories | ~50K | ~500K |
| Recall latency (10K memories) | <50ms | <1ms |
| Memory for 1M edges | ~600MB–1GB | ~150MB |
| Concurrent reads | N (Python RWLock) | N (no GIL, true parallel) |

*Projections, not benchmarks. Actual performance depends on graph topology, property sizes, and workload.*

---

## Phase 2: Disk-First Storage

*Data lives on disk. The engine queries storage on demand. RAM = cache, not the graph.*

This is the fundamental architecture shift. The in-memory index design (7 HashMaps holding the full graph) is replaced by a storage engine that serves indexed lookups from disk with a page cache for hot data.

### The StorageEngine trait

```rust
pub trait StorageEngine: Send + Sync {
    // Point lookups
    fn get_node(&self, ns: &str, id: &str) -> Option<Node>;
    fn get_edge(&self, ns: &str, id: &str) -> Option<Hyperedge>;

    // Index scans
    fn edges_of_node(&self, ns: &str, node_id: &str) -> Vec<Hyperedge>;
    fn edges_by_vertex_set(&self, ns: &str, hash: &str) -> Vec<Hyperedge>;
    fn edges_by_type(&self, ns: &str, edge_type: &str) -> Vec<Hyperedge>;
    fn nodes_by_type(&self, ns: &str, node_type: &str) -> Vec<Node>;
    fn edges_referencing(&self, ns: &str, edge_id: &str) -> Vec<Hyperedge>;

    // Filtered scans
    fn scan_edges(&self, ns: &str, filters: &EdgeFilters) -> Vec<Hyperedge>;

    // Writes
    fn put_node(&self, ns: &str, node: &Node);
    fn put_edge(&self, ns: &str, edge: &Hyperedge);
    fn delete_node(&self, ns: &str, id: &str) -> bool;
    fn delete_edge(&self, ns: &str, id: &str) -> bool;

    // Transactions
    fn begin(&self);
    fn commit(&self);
    fn rollback(&self);
}
```

The graph logic (`graph.rs`, `activation.rs`) calls this trait. It doesn't know what's behind it.

**Design note: `Vec` vs iterators.** The scan methods above return `Vec`, materializing full result sets in memory. For a disk-first architecture this is a tradeoff: a query matching 100K edges allocates all of them before the caller can filter or limit. The production implementation should consider `impl Iterator<Item = Hyperedge>` for scan methods to enable lazy evaluation and early termination, or add `limit`/`offset` parameters. `Vec` is shown here for API clarity.

### KV key layout

All backends — embedded or distributed — use the same key-prefix scheme in an ordered key-value store:

```
n:{namespace}:{node_id}                   → Node (serialized)
e:{namespace}:{edge_id}                   → Edge (serialized)
i:{namespace}:{node_id}:{edge_id}         → ()   (node→edge index)
t:{namespace}:n:{type}:{node_id}          → ()   (node type index)
t:{namespace}:e:{type}:{edge_id}          → ()   (edge type index)
v:{namespace}:{vertex_set_hash}:{edge_id} → ()   (vertex-set index)
m:{namespace}:{ref_edge_id}:{edge_id}     → ()   (metagraph index)
c:{namespace}:{created_at_ms}:{edge_id}  → ()   (creation-time index)
x:{namespace}:{expired_at_ms}:{edge_id}  → ()   (expiry-time index)
a:{namespace}:{kind}:{ref_id}            → AccessStats (access log)
```

Point lookup: `get("n:default:Alice")` — O(1) in any backend.
Index scan: `scan_prefix("i:default:Alice:")` — all edges containing Alice.
Vertex-set lookup: `scan_prefix("v:default:{hash}:")` — exact node-set match.
Temporal query: range scan on `c:` prefix supports `since`/`before` filters. Active-edge queries combine `c:` and `x:` prefixes.

### Backend: SQLite (rusqlite)

First disk-first backend. Uses `rusqlite` to talk to SQLite directly from Rust — no Python sqlite3, no FFI overhead for each query.

Same SQLite schema as today. The goal is that existing `.db` files work unchanged, but the migration needs care:

- **Extension loading.** The current Python code loads sqlite-vec via `sqlite_vec.load(conn)` (a Python C extension shim). Rust must load the sqlite-vec shared library directly via `rusqlite`'s `load_extension()`, with different platform-specific paths and linking requirements.
- **PRAGMA settings.** The current code sets `journal_mode=WAL` and `foreign_keys=ON` at connection open. The Rust backend must reproduce these exactly; different WAL behavior or missing foreign key enforcement would silently change semantics.
- **Schema versioning.** The current schema is at version 5, tracked in the `meta` table. The Rust backend must read and respect this version, and any new schema changes need the same version-check + migration pattern.
- **Embedding dimension lock.** `SQLiteStorage` locks embedding dimension on first use via `_ensure_vec_table()`. The Rust backend must enforce the same constraint or document how to handle dimension mismatches.

**Why start here:** backward compatibility. With the above handled, users upgrade the package and their existing databases work unchanged.

### Backend: redb or fjall (pure Rust)

Optional second backend for users who want higher performance without external C dependencies.

| | redb | fjall |
|---|---|---|
| Design | B-tree (like LMDB) | LSM-tree (like RocksDB) |
| Reads | Very fast (memory-mapped) | Fast |
| Writes | Fast | Very fast (append-only) |
| Rust-native | Yes (pure Rust) | Yes (pure Rust) |
| Maturity | Stable, used in production | Newer, actively developed |

For read-heavy hypergraph workloads (many recalls, fewer remembers), redb's memory-mapped B-tree is a good fit. For write-heavy workloads, fjall's LSM-tree handles sustained ingestion better.

### Spreading activation on disk

Budget-bounded BFS with 50 node visits against a disk-backed store:

- Page cache warm: ~1–10μs per lookup → **0.05–0.5ms total**
- Page cache cold (SSD): ~50–100μs per lookup → **2.5–5ms total**

The budget bound is what makes disk-backed activation practical. You visit a fixed number of nodes regardless of graph size. A billion-edge graph does 50 lookups, same as a thousand-edge graph.

### Envelope after Phase 2

| Dimension | Before | After |
|-----------|--------|-------|
| Nodes + Edges | ~5M (RAM-bound) | **50M+** (disk-bound) |
| Memories | ~500K | **1M+** |
| Memory footprint | O(graph) | **O(cache size)** — configurable |
| Max database size | ~2GB (practical RAM limit) | **100GB+** (SQLite/redb file limit) |
| Existing .db files | — | Work unchanged (SQLite backend) |

*Projections. Actual limits depend on hardware, data shape, and cache configuration.*

---

## Phase 3: Vector Search Upgrade

*Replace sqlite-vec with a purpose-built ANN engine.*

sqlite-vec works for small-to-medium embedding counts but it's a virtual table bolted onto SQLite — not a dedicated vector index. At scale, a purpose-built ANN library is 10–100× faster.

### usearch

[USearch](https://github.com/unum-cloud/usearch) is a C++/Rust ANN library. Single-file index, SIMD-optimized, supports quantization (float16, int8) for memory savings. Used in production systems.

```rust
// Rust API
let index = usearch::Index::new(&usearch::IndexOptions {
    dimensions: 384,
    metric: usearch::MetricKind::Cos,
    quantization: usearch::ScalarKind::F32,
    ..Default::default()
})?;

index.add(id, &embedding)?;
let results = index.search(&query_embedding, 10)?;
```

### What changes

The `StorageEngine` trait gets vector methods:

```rust
fn save_embedding(&self, ns: &str, kind: &str, ref_id: &str,
                  text: &str, embedding: &[f32]) -> Result<()>;
fn search_vectors(&self, ns: &str, query: &[f32], limit: usize,
                  kind: Option<&str>) -> Vec<VectorResult>;
```

The SQLite backend continues using sqlite-vec. The redb/fjall backend uses usearch for its vector index. Users choose based on their needs.

### Envelope after Phase 3

| Dimension | Before | After |
|-----------|--------|-------|
| Embedding count | ~10K (sqlite-vec) | **1M+** (usearch) |
| Vector search latency | ~10ms | **<1ms** |
| Semantic entity expansion | Bottleneck at scale | Scales to millions of embeddings |

*Projections based on usearch benchmarks, not hypabase-specific measurements.*

---

## Phase 4: Networked Backends

*Same StorageEngine trait, backed by a network database. Multi-writer, multi-machine.*

### Backend: PostgreSQL

The pragmatic first step to multi-writer support. Battle-tested, widely deployed, well-understood operationally.

The KV key layout maps to PostgreSQL tables with indexes. The `StorageEngine` implementation uses a connection pool (`deadpool-postgres` or `bb8`) and translates trait methods to SQL queries.

**What it unlocks:**
- Multiple writers (concurrent `remember()` from multiple agents)
- Larger-than-disk datasets (PostgreSQL handles TB-scale)
- Standard operational tooling (backups, monitoring, replication)

### Backend: FoundationDB

For horizontal scaling beyond a single machine. FoundationDB is a distributed ordered key-value store with ACID transactions. Used by Apple (iCloud) and Snowflake at massive scale.

The KV key layout from Phase 2 maps directly to FoundationDB's ordered keyspace. The `scan_prefix` operation is a FoundationDB range read. Transactions are native.

A Rust client exists ([`foundationdb-rs`](https://github.com/foundationdb-rs/foundationdb-rs)).

**What it unlocks:**
- Horizontal scaling — add machines to add capacity
- Multi-region replication
- Effectively unlimited storage

### Envelope after Phase 4

| Dimension | Before | After |
|-----------|--------|-------|
| Nodes + Edges | 50M+ (single machine) | **Billions** (distributed) |
| Writers | 1 (embedded) | **Many** (concurrent) |
| Storage | Hundreds of GB (single disk) | **Unlimited** (distributed) |
| Deployment | Embedded library | Library or client-server |

*Projections. Distributed system performance depends heavily on network topology and consistency requirements.*

---

## Summary

```
Phase 0 ─── Python optimizations ──────── 50K memories, <50ms recall
  │
Phase 1 ─── Rust core (PyO3) ─────────── 500K memories, <1ms recall
  │
Phase 2 ─── Disk-first storage trait ──── 1M+ memories, 100GB+ databases
  │         ├── SQLite (backward compat)
  │         └── redb/fjall (pure Rust)
  │
Phase 3 ─── Vector search (usearch) ───── 1M+ embeddings, <1ms search
  │
Phase 4 ─── Networked backends ────────── Billions of edges, multi-writer
            ├── PostgreSQL
            └── FoundationDB
```

Each phase is independently valuable and ships as a release. Users on Phase 0 get a faster Python library. Users on Phase 1 get a Rust-accelerated Python library. Users on Phase 2+ get a disk-backed engine that scales beyond RAM. The Python API and MCP interface remain unchanged throughout.

### Design invariants across all phases

1. **Same Python API.** `hb.edge(...)`, `hb.recall(...)`, `hb.remember(...)` work the same regardless of backend.
2. **Same MCP tools.** The MCP server doesn't know or care what storage engine is underneath.
3. **Backward-compatible files.** The SQLite backend reads existing `.db` files, given correct extension loading, PRAGMA settings, and schema version handling (see Phase 2 migration notes).
4. **StorageEngine trait is the boundary.** All backends implement the same trait. Graph logic, activation, and memory are backend-agnostic.
5. **Budget-bounded activation.** Recall cost is O(budget), not O(graph), at every phase.
