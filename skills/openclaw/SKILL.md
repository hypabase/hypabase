# Hypabase Memory Skill

A persistent memory system for AI agents that understands WHO did WHAT to WHOM — powered by a hypergraph engine with semantic roles, provenance tracking, and neuroscience-informed decay.

## Memory Protocol

### When to Remember
- After decisions, preferences, stated facts, assigned tasks, or events
- When the user tells you something about themselves or their project
- After learning a procedure or workflow

### How to Structure
Every memory is an **ACTION** (verb) with **PARTICIPANTS** in **ROLES** (kāraka):

| Role | Meaning | Example |
|------|---------|---------|
| `agent` | Who did it | Alice |
| `object` | What was acted upon | the proposal |
| `instrument` | By what means | Slack |
| `recipient` | For/to whom | Bob |
| `source` | From where | the old system |
| `locus` | Where/when | sprint review |

### Memory Types
| Type | Use for | Decay |
|------|---------|-------|
| `episodic` | Events, meetings, conversations | Fast |
| `semantic` | Facts, preferences, definitions | Slow |
| `procedural` | How-to, workflows, processes | Slowest |

### Importance Ratings
- `0.9` — Critical (user preferences, key decisions)
- `0.7` — Important (project facts, team structure)
- `0.5` — Relevant (meeting notes, task updates)
- `0.3` — Minor (casual mentions, context)

### Mood

Mood captures the modality of a memory — what kind of truth it represents.

| Mood | When to use | Example |
|------|-------------|---------|
| `actual` | Something that happened or is true (default) | "Alice deployed the API" |
| `planned` | Something that will or is intended to happen | "Alice will deploy on Friday" |
| `uncertain` | Something that might be true | "The API might have a memory leak" |
| `normative` | Something that should or shouldn't be true | "We should use PostgreSQL" |

When mood is omitted, it defaults to `actual`. Only set mood explicitly when the memory is planned, uncertain, or normative.

### Negation

Set `negated=true` when a memory expresses that something is NOT the case.

| Statement | negated | mood |
|-----------|---------|------|
| "Alice uses Python" | false | actual |
| "Alice does NOT use Java" | true | actual |
| "We should NOT use MongoDB" | true | normative |
| "Bob might not attend" | true | uncertain |

### Decomposition Rule

**One action per memory.** When a sentence contains multiple actions, decompose it into separate `remember()` calls. Shared entities will link them in the graph.

Example: "Alice told Bob to migrate the database"
- Memory 1: action="tell", agent=Alice, recipient=Bob, object="database migration"
- Memory 2: action="migrate", agent=Bob, object="database"

Both memories share the Bob and database entities, so they are naturally connected in the graph.

### Valence Templates

Use the verb class to determine which roles are needed:

| Verb class | Examples | Required roles |
|------------|----------|---------------|
| **Transfer** | assign, send, give, delegate | agent + object + recipient |
| **State** | is, has, exists, contains | agent + object |
| **Cognitive** | knows, believes, decided, learned | agent + object (+ source) |
| **Creation** | built, wrote, designed, implemented | agent + object (+ instrument) |
| **Preference** | prefers, likes, wants, avoids | agent + object |
| **Communication** | told, asked, reported, announced | agent + object + recipient |
| **Motion** | moved, deployed, migrated, shipped | agent + object + locus |
| **Usage** | uses, runs, employs, applies | agent + object (+ instrument) |
| **Evaluation** | approved, rejected, reviewed, rated | agent + object |
| **Temporal** | scheduled, started, finished, delayed | agent + object + locus |

## The Recall Mirror

Remember and recall use the **same grammar**. The agent decomposes its query into grammar dimensions, the system stores and retrieves.

```
Remember stores:              Recall queries:
──────────────────           ──────────────────
entities + roles        →    entity + role filter + semantic vertex set lookup
action (verb)           →    action filter
memory_type             →    memory_type filter + type-specific decay
mood                    →    mood filter
negated                 →    negated filter
importance              →    strength ranking (salience)
confidence              →    strength ranking (confidence)
text                    →    (stored for display, not queried directly)
created_at              →    since / before filters
```

## Tools

### remember
Store a memory: ACTION + ENTITIES in ROLES. At least 2 entities required (it's a hyperedge).

**Parameters:**
- `action` (required): The verb (e.g., "assigned", "prefers", "deployed").
- `entities` (required): Participants — list of dicts with `name` (required), `role` (kāraka), and optional `type` (default "entity"). Minimum 2.
- `text` (optional): Human-readable form. Stored for display only, not parsed.
- `memory_type` (optional): "episodic", "semantic", or "procedural".
- `importance` (optional): 0.0-1.0 importance rating.
- `mood` (optional): "actual", "planned", "uncertain", or "normative". Default: actual.
- `negated` (optional): true if this is a negation. Default: false.
- `source` (optional): Provenance source identifier. Default: "memory".
- `confidence` (optional): Confidence score 0.0-1.0. Default: 1.0.

**Returns:** Edge ID, node IDs, action, and any detected contradictions.

### recall
Recall memories using the same grammar you stored with. At least one dimension required.

**Parameters:**
- `entity` (optional): WHO/WHAT — entity name or list of names. Single: focused lookup. List: vertex set overlap (find memories involving ALL).
- `action` (optional): Filter by action type (the verb).
- `role` (optional): Filter by kāraka role (agent/object/instrument/recipient/source/locus).
- `memory_type` (optional): Filter by memory type (episodic/semantic/procedural).
- `mood` (optional): Filter by mood (actual/planned/uncertain/normative).
- `negated` (optional): Filter — true=only negated, false=only positive.
- `since` (optional): ISO date string — only memories created after this.
- `before` (optional): ISO date string — only memories created before this.
- `limit` (optional): Maximum results. Default: 10.
- `min_strength` (optional): Minimum memory strength threshold. Default: 0.0.

**Examples:**
- `recall(entity="Alice")` — everything about Alice
- `recall(entity="Alice", action="assign", role="agent")` — what Alice assigned
- `recall(entity="Bob", role="recipient")` — what was done TO Bob
- `recall(entity=["Alice", "API"])` — memories involving both
- `recall(mood="planned")` — all plans
- `recall(action="deploy", negated=true)` — what should NOT be deployed

**Returns:** List of matching memories with scores, strength, roles, mood, negated, and memory type.

### consolidate
Compress repeated episodic memories into semantic knowledge. Call periodically to keep memory efficient.

**Parameters:**
- `entity` (optional): Only consolidate memories involving this entity.

**Returns:** List of consolidated summaries.

### forget
Expire old or low-strength memories (soft delete).

**Parameters:**
- `older_than_days` (optional): Expire memories older than this many days.
- `min_strength` (optional): Expire memories below this strength threshold.
- `entity` (optional): Only forget memories involving this entity.
- `memory_type` (optional): Only forget memories of this type.
- `mood` (optional): Only forget memories of this mood.

**Returns:** Number of memories expired.

### connections
Explore an entity's neighborhood in the memory graph using multi-hop BFS.

**Parameters:**
- `entity` (required): The entity name to explore.
- `max_hops` (optional): Maximum traversal depth. Default: 2.
- `role` (optional): Filter by kāraka role.

**Returns:** Neighbors, connected edges, and path information.

### who_knows_what
Summary of what the memory system knows: entity counts, edge counts, memory type breakdown, mood breakdown, and top entities.

**Returns:** Statistics about stored knowledge.

### resolve_contradiction
Resolve a contradiction between two memories (called after `remember` returns contradictions).

**Parameters:**
- `new_edge_id` (required): The newer memory edge ID.
- `old_edge_id` (required): The older memory edge ID.
- `resolution` (required): "supersede" (expire old), "keep_both", or "keep_old" (expire new).

**Returns:** Resolution outcome.

## Configuration

Memory and embeddings are enabled by default — zero config required.

Environment variables for customization:
- `HYPABASE_EMBEDDER` — Embedder for semantic search (default: FastEmbed):
  - `fastembed` / `fast` / `default` — BAAI/bge-small-en-v1.5 via ONNX (default).
  - `openai` — Uses text-embedding-3-small via OpenAI API (requires `OPENAI_API_KEY`).
  - `sentence-transformers` / `st` / `local` — Uses all-MiniLM-L6-v2 (requires `sentence-transformers`).
  - `none` — Disable embeddings entirely.
- `HYPABASE_DB_PATH` — SQLite database path (default: `hypabase.db`).
