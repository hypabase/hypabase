# Architecture

## What hypabase is

**hypabase** is a Python hypergraph engine. Edges connect 2+ nodes. Every edge carries provenance (source + confidence). Storage is SQLite. That's it.

It is infrastructure. Most people will never use it directly — they'll use a product built on it.

## What hypabase memory is

**hypabase memory** is structured, semantic memory for AI agents — built on the hypabase engine. This is the product. This is what goes on ClawHub. This is what agents install.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  PRODUCTS                                        │
│                                                  │
│  hypabase memory         (future products)       │
│  "memory for AI agents"                          │
│                                                  │
│  ┌────────────┐          ┌────────────┐          │
│  │ MCP server │          │ MCP server │          │
│  │ (7 tools)  │          │ (N tools)  │          │
│  └─────┬──────┘          └─────┬──────┘          │
│        │                       │                 │
├────────▼───────────────────────▼─────────────────┤
│  ENGINE                                          │
│                                                  │
│  hypabase                                        │
│  "Python hypergraph library"                     │
│                                                  │
│  Hypabase client → SQLiteStorage → SQLite        │
│                                    + sqlite-vec  │
│                                                  │
│  ┌─────┐                                         │
│  │ CLI │                                         │
│  └─────┘                                         │
└──────────────────────────────────────────────────┘
```

Products are opinionated interfaces for specific audiences. The engine is general-purpose infrastructure. Products depend on the engine; the engine knows nothing about products.

---

## MCP server — `hypabase-memory`

An agent connects to this server to remember and recall information. It runs as a sidecar, continuously, alongside the agent.

**Audience:** AI agents (OpenClaw, Claude with MCP, custom agents).

**Start it:**

```bash
uv add hypabase
hypabase-memory
```

No env vars. No config. SQLite auto-created. Embeddings included.

**Tools (7):**

| Tool | What it does |
|------|-------------|
| `remember` | Store a memory as an action with participants in roles |
| `recall` | Query memories by entity, action, role, type, mood |
| `forget` | Expire old or low-strength memories |
| `consolidate` | Compress repeated episodic memories into knowledge |
| `connections` | Explore an entity's neighborhood in the graph |
| `who_knows_what` | Summary stats of stored knowledge |
| `resolve_contradiction` | Resolve conflicts between memories |

That's it. 7 tools. The agent sees nothing else — no engine internals, no hypergraph concepts, no configuration tools.

**What the agent sees on connect:**

```
You have persistent memory.
- remember(action="...", entities=[...]) — store a memory
- recall(entity="...") — find memories
- forget(older_than_days=30) — clean up

Example: remember(action="uses", entities=[{"name": "Alice", "role": "agent"}, {"name": "Python", "role": "object"}])
```

The starting point is grammar-based: `action=` + `entities=`. Deeper dimensions are in the tool descriptions — roles, memory_type, mood, importance, negation. The agent discovers them when it needs precision, not at onboarding.

There is no engine MCP server. The engine's interface is Python. MCP exists to serve products, not to wrap the engine.

---

## Design principles

### 1. Products hide the engine

Memory users never see hypergraphs. They see: remember, recall, forget. The engine is an implementation detail — like PostgreSQL inside Supabase.

Product docs say "structured memory for AI agents." Engine docs say "Python hypergraph library." They never cross.

### 2. One install, everything works

```bash
uv add hypabase
```

This gives you the engine, the memory module, the MCP server, embeddings, and the CLI. No extras, no brackets, no "which optional features do I need?" decisions.

If a friction exists to serve our architecture rather than the user's task, remove it.

### 3. Semantic by default

Embeddings are core, not optional. Recall uses embedding similarity to match entities — agents don't need exact names to find memories. "deployment" finds memories about "Docker" and "staging." "Bob" finds memories about "Robert Smith."

The system is semantic. That's the product.

### 4. No LLM inside

The calling agent (an external LLM) does all natural language parsing. It decomposes its query into structured grammar before calling tools. Hypabase does graph work: store, traverse, score, return.

This keeps the library simple, deterministic, and debuggable. No API keys, no model dependencies, no hallucination risk inside the persistence layer.

### 5. Grammar in, graph out

Remember and recall use the same grammar — the karaka system:

```
Remember stores:              Recall queries:
──────────────────           ──────────────────
entities + roles        →    entity + role filter
action (verb)           →    action filter
memory_type             →    memory_type filter
mood                    →    mood filter
negated                 →    negated filter
importance              →    strength ranking
```

The agent decomposes its intent into grammar dimensions. The system stores and retrieves. Symmetry makes the API predictable.

### 6. Progressive disclosure

The onboarding message is 4 lines: remember, recall, forget, and one example. The full grammar — roles, memory_type, mood, importance, negation — lives in individual tool descriptions. The agent reads a tool's docstring when it calls that tool, not before. Complexity is discovered at the point of use.

### 7. One SQLite file, zero config

All state lives in one `.db` file. No external services, no Docker, no API keys.

```bash
hypabase-memory        # creates memory.db, starts serving
```

That's the setup. There is no step two.

### 8. Each product gets its own MCP server

Products don't share MCP servers. Memory has 7 tools. A future product would have its own N tools.

An agent connecting to `hypabase-memory` sees exactly 7 tools. No clutter, no confusion.

### 9. Strength through forgetting

Memories aren't permanent. They decay — episodic memories fast, semantic memories slow, procedural memories slowest. Strength is computed from recency, access frequency, importance, and confidence. Weak memories are expired. This prevents memory bloat and keeps recall fast and relevant.

---

## Package structure

```
hypabase/
├── hypabase/                   # Python package
│   ├── __init__.py             # Engine exports: Hypabase, Node, Edge
│   ├── client.py               # Hypabase class — the engine API
│   ├── models.py               # Pydantic models shared across layers
│   │
│   ├── engine/                 # Storage and computation layer
│   │   ├── core.py             # In-memory hypergraph (indexes, traversal)
│   │   ├── storage.py          # SQLiteStorage — all SQL lives here
│   │   ├── embeddings.py       # EmbeddingProvider ABC + implementations
│   │   ├── persistence_engine.py
│   │   └── vector.py           # Pure-Python cosine similarity, pack/unpack
│   │
│   ├── memory/                 # Memory product
│   │   ├── __init__.py         # Exports: Memory
│   │   ├── agent.py            # Memory class (remember/recall/forget/etc.)
│   │   ├── server.py           # Memory MCP server (7 tools, stdio transport)
│   │   ├── resolution.py       # EntityResolver (normalization, aliases, embeddings)
│   │   ├── strength.py         # Memory strength scoring (decay, frequency, salience)
│   │   └── types.py            # KarakaRole, MemoryType, Mood, role weights
│   │
│   └── cli/                    # CLI
│       └── main.py
│
├── skills/                     # OpenClaw skill definitions
│   └── openclaw/
│       ├── SKILL.md            # Memory skill documentation
│       └── claw.json           # ClawHub manifest
│
├── tests/
│   ├── test_client.py          # Engine tests
│   ├── test_memory.py          # Memory tests
│   └── test_memory_mcp.py      # Memory MCP tests
│
├── pyproject.toml
└── ARCHITECTURE.md             # This file
```

---

## Dependencies

```toml
dependencies = [
    "pydantic>=2.0",
    "sqlite-vec>=0.1.1",
    "mcp>=1.0",
    "click>=8.0",
    "fastembed>=0.7",
]
```

One install. Everything works. No optional extras for core functionality.

FastEmbed (ONNX-based, ~50MB) provides high-quality embeddings without pulling PyTorch. `sentence-transformers` remains available as an optional alternative under the `[memory]` extra.

---

## Entry points

```toml
[project.scripts]
hypabase = "hypabase.cli.main:cli"               # CLI
hypabase-memory = "hypabase.memory.server:run"    # Memory MCP server
```

---

## Getting started

### I'm building an AI agent and want persistent memory

```bash
uv add hypabase
hypabase-memory
```

Connect your agent (OpenClaw, Claude, etc.) to the MCP server. The agent sees 7 tools. It starts storing memories immediately. No configuration. No schema setup.

### I'm a Python developer and want a hypergraph library

```bash
uv add hypabase
```

```python
from hypabase import Hypabase

hb = Hypabase("my.db")
hb.edge(["Alice", "Bob", "Project X"], type="collaborates", source="meeting")
edges = hb.edges(containing=["Alice"])
```

### I want to use memory from Python (no MCP)

```bash
uv add hypabase
```

```python
from hypabase.memory import Memory

mem = Memory(path="agent.db")
mem.remember(
    action="assigned",
    entities=[
        {"name": "Alice", "type": "person", "role": "agent"},
        {"name": "API task", "type": "task", "role": "object"},
        {"name": "Bob", "type": "person", "role": "recipient"},
    ],
    memory_type="episodic",
    importance=0.7,
)
results = mem.recall(entity="Alice")
```

Preferences with comparison (the `source` role captures "over X"):

```python
mem.remember(
    action="prefers",
    entities=[
        {"name": "Alice", "role": "agent"},
        {"name": "Python", "role": "object"},
        {"name": "Java", "role": "source"},     # compared against
    ],
    memory_type="semantic",
)
```

### I want to use memory from OpenClaw

Install the hypabase-memory skill from ClawHub or add `skills/openclaw/SKILL.md` to your agent's skills directory. The skill teaches your agent the memory grammar.

---

## Future products

New products follow the same pattern:

```
hypabase/
├── memory/        ← agent memory (shipped)
├── ontology/      ← domain ontology builder (future)
├── trace/         ← agent execution tracing (future)
└── ...
```

Each product:
- Lives in its own sub-module under `hypabase/`
- Gets its own MCP server with focused tools
- Gets its own CLI entry point
- Depends on the engine internally, hides it externally

The engine stays stable and general. Products move fast and serve specific audiences.
