# Scaling Roadmap

Hypabase is a Python hypergraph library backed by SQLite. This document describes the architecture path from its current single-process design to a disk-first, backend-agnostic engine capable of scaling to billions of edges.

Each phase is additive. Later phases don't rewrite earlier ones — they add new `StorageEngine` implementations behind the same trait.

---

## Current Architecture

```
Python (hypabase)
├── client.py         Hypabase class — public API
├── engine/
│   ├── core.py       HypergraphCore — in-memory indexes (6 Python dicts)
│   └── storage.py    SQLiteStorage — persistence layer
└── memory/
    └── agent.py      Memory module — remember/recall/forget
```

**How it works today:** On startup, the entire graph loads from SQLite into Python dicts. Reads are dict lookups. Writes mutate dicts and mirror to SQLite. The graph lives in RAM; SQLite is a persistence side effect.

**Current operating envelope:**

| Dimension | Comfortable | Ceiling |
|-----------|-------------|---------|
| Nodes + Edges | 50K–100K | ~500K |
| Memories | ~100 | ~1K |
| Recall latency | <2s at 100 memories | Degrades linearly |
| Writes/sec | 1–5K | SQLite single-writer limit |
| Memory footprint | O(graph size) | ~200 bytes/node, ~500 bytes/edge in Python |

**What limits scale:**

1. **Memory-first design.** The entire graph must fit in RAM as Python objects.
2. **Python object overhead.** Each node/edge is a dataclass + dict + GC header. 4–5× larger than the raw data.
3. **Global RLock.** All operations (reads and writes) serialize through one lock.
4. **Per-write SQLite commits.** Each `write_node`/`write_edge` auto-commits outside a `batch()` context.
5. **Spreading activation BFS.** Visits all reachable nodes up to depth — O(V×E) worst case.

---

## Phase 0: Python Optimizations

*No Rust. No new dependencies. Improvements within the current architecture.*

### 0a. RWLock on HypergraphCore

Replace the single `RLock` with a read-write lock. Recalls are pure reads — they don't mutate the graph. With a RWLock, unlimited concurrent reads run in parallel while writes still serialize.

**Impact:** N× concurrent read throughput (N = reader threads).

### 0b. Push filters to SQL in recall

The filter-only recall path (no entity provided) loads all active edges into Python, then filters in a loop. Every filter dimension (action, memory_type, mood, since, before) already has a SQL index. Push them down.

**Impact:** Filter-only recall goes from O(E) Python objects to O(matching) SQL rows. 10–200× fewer objects at scale.

### 0c. Budget-bounded spreading activation

Replace BFS deque (explores everything up to depth) with a max-heap and activation budget. Best-first search — highest-activation paths explored first, weak paths pruned by budget exhaustion.

**Impact:** Recall becomes O(budget) instead of O(graph). Constant-time regardless of graph size. The single highest-ROI change for the memory module.

### 0d. Write coalescing

Wrap multi-write operations (like `remember()`, which creates nodes + edges + same_as links) in implicit `batch()` transactions. Eliminates per-operation SQLite commits.

**Impact:** `remember()` latency drops 4–8× (one commit instead of 4–8).

### 0e. Incremental consolidation index

Maintain a co-occurrence counter as a side effect of `remember()` instead of computing all pairwise counts in `consolidate()`. The current O(E × k²) full scan becomes an O(1) read of pre-computed counts.

**Impact:** `consolidate()` scales to 100K+ memories.

### Envelope after Phase 0

| Dimension | Before | After |
|-----------|--------|-------|
| Memories | ~1K | ~50K |
| Recall latency (10K memories) | Seconds | <50ms |
| Concurrent reads | 1 (serialized) | N (parallel) |
| `remember()` latency | ~10–20ms | ~2–5ms |

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
├── graph.rs           6 index HashMaps, traversal, vertex-set lookup
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

### Envelope after Phase 1

| Dimension | Before | After |
|-----------|--------|-------|
| Nodes + Edges | ~500K | ~5M |
| Memories | ~50K | ~500K |
| Recall latency (10K memories) | <50ms | <1ms |
| Memory for 1M edges | ~600MB–1GB | ~150MB |
| Concurrent reads | N (Python RWLock) | N (no GIL, true parallel) |

---

## Phase 2: Disk-First Storage

*Data lives on disk. The engine queries storage on demand. RAM = cache, not the graph.*

This is the fundamental architecture shift. The in-memory index design (6 HashMaps holding the full graph) is replaced by a storage engine that serves indexed lookups from disk with a page cache for hot data.

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
a:{namespace}:{kind}:{ref_id}             → AccessStats (access log)
```

Point lookup: `get("n:default:Alice")` — O(1) in any backend.
Index scan: `scan_prefix("i:default:Alice:")` — all edges containing Alice.
Vertex-set lookup: `scan_prefix("v:default:{hash}:")` — exact node-set match.

### Backend: SQLite (rusqlite)

First disk-first backend. Uses `rusqlite` to talk to SQLite directly from Rust — no Python sqlite3, no FFI overhead for each query.

Same SQLite schema as today. Existing `.db` files work unchanged. This is a non-breaking migration.

**Why start here:** backward compatibility. Users upgrade the package and their existing databases just work.

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
3. **Backward-compatible files.** The SQLite backend always reads existing `.db` files.
4. **StorageEngine trait is the boundary.** All backends implement the same trait. Graph logic, activation, and memory are backend-agnostic.
5. **Budget-bounded activation.** Recall cost is O(budget), not O(graph), at every phase.
