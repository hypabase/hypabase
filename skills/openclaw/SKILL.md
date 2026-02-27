---
metadata:
  clawdbot:
    emoji: "🧠"
    requires:
      env: []
      bins: ["hypabase-memory"]
    primaryEnv: "HYPABASE_DB_PATH"
    files: []
---

# Hypabase Memory

Persistent structured memory for AI agents — store and recall WHO did WHAT to WHOM using PENMAN notation, semantic roles, and provenance.

## When to Remember

Store a memory when the user:
- Makes a decision or states a preference
- Shares a fact about themselves, their team, or a project
- Assigns a task or delegates work
- Describes an event, meeting, or outcome
- Explains a procedure or workflow

## PENMAN Notation

Every memory is a verb with participants in role slots:

```
(verb :role "entity" :role "entity" ...)
```

Examples:

```
(prefers :subject Alice :object Python :memory_type semantic)

(assigned :subject Alice :object "billing task" :recipient Bob
 :instrument Jira :locus Monday :tense past :memory_type episodic)

(has :subject "quick sort" :attribute "time complexity"
 :value "O(n log n)" :memory_type semantic)

(uses :subject Django :object "Python 2" :negated true :memory_type semantic)
```

Multiple atoms in a single call:

```
(deployed :subject Alice :object API :locus Monday :tense past)
(reviewed :subject Bob :object API :locus Tuesday :tense past)
```

## Roles

Eight kāraka (semantic case) roles. Fill in what applies, skip what doesn't.

| PENMAN role | Recall role | Meaning | Example |
|-------------|-------------|---------|---------|
| `:subject` | `subject` | Who or what it's about | Alice |
| `:object` | `object` | What is acted on | the proposal |
| `:instrument` | `instrument` | Tool, method, or means used | Slack |
| `:recipient` | `recipient` | Who receives or benefits | Bob |
| `:origin` | `source` | Where it came from, previous state | the old system |
| `:locus` | `locus` | Where, when, or in what context | sprint review |
| `:attribute` | `attribute` | A named property or dimension | time complexity |
| `:value` | `value` | The specific value of that property | O(n log n) |

Note: PENMAN uses `:origin` but `recall(role=...)` uses the internal name `source`.

## Memory Types

| Type | Use for | Decay rate |
|------|---------|------------|
| `episodic` | Events, meetings, conversations | Fast |
| `semantic` | Facts, preferences, definitions | Slow |
| `procedural` | How-to, workflows, processes | Slowest |

Set via the `:memory_type` modifier in PENMAN.

## Mood

| Mood | When to use | Example |
|------|-------------|---------|
| `actual` | Something that happened or is true (default) | "Alice deployed the API" |
| `planned` | Something intended to happen | "Alice will deploy on Friday" |
| `uncertain` | Something that might be true | "The API might have a memory leak" |
| `normative` | Something that should or shouldn't be | "We should use PostgreSQL" |
| `conditional` | Something that depends on a condition | "If tests pass, deploy" |

Set via `:mood` modifier. Omit for `actual` (the default).

## Negation & Modifiers

| Modifier | Values | Default |
|----------|--------|---------|
| `:tense` | `past`, `present`, `future` | — |
| `:negated` | `true`, `false` | `false` |
| `:importance` | `0.0` to `1.0` | — |
| `:mood` | `actual`, `planned`, `uncertain`, `normative`, `conditional` | `actual` |
| `:memory_type` | `episodic`, `semantic`, `procedural` | — |

Context modifiers (can hold nested atoms):

| Modifier | Meaning |
|----------|---------|
| `:cause` | Why it happened |
| `:purpose` | What for |
| `:condition` | If/when/unless |

## Entity Naming Rules

Entity naming determines whether memories connect or fragment. This is critical.

- **Same string after lowercasing = same entity.** "Alice" and "alice" share one node.
- **Different strings = different entities.** "Bob" and "Robert" create separate nodes until `consolidate()` merges them.
- Pick one canonical name per entity and reuse it across all memories.
- Use full descriptive names: "machine learning" not "ML", "JavaScript" not "JS".
- Check the `resolved` field in `remember()` responses. If an entity you expected to be "existing" shows as "new", you have a naming inconsistency.
- When `recall()` returns nothing, the most common cause is naming variance. Try the exact name used in `remember()`, or query by action/memory_type instead.
- Call `consolidate()` periodically to merge similar names via semantic similarity.

## Decomposition Rule

**One action per memory.** When a sentence contains multiple actions, decompose into separate atoms. Shared entities link them in the graph.

"Alice told Bob to migrate the database":

```
(told :subject Alice :recipient Bob :object "database migration" :tense past)
(migrate :subject Bob :object database :mood planned)
```

Both memories share the Bob entity, so they connect naturally.

## Nesting

Any role slot can hold a nested atom instead of a string:

```
(believes :subject Alice :object (is :subject deadline :value Friday))

(caused :subject outage
 :cause (deployed :subject Bob :object "broken build"))
```

## Tools

### remember(penman, source?, confidence?)

Store memories as PENMAN atoms.

**Parameters:**
- `penman` (required): One or more PENMAN atoms.
- `source` (optional): Provenance source identifier. Default: `"memory"`.
- `confidence` (optional): Confidence score 0.0–1.0. Default: `1.0`.

**Returns:**
```json
{
  "stored": 1,
  "memories": [
    {
      "text": "Alice prefers Python",
      "action": "prefers",
      "roles": {"subject": "Alice", "object": "Python"},
      "type": "semantic",
      "resolved": {"Alice": "existing", "Python": "new"}
    }
  ],
  "activated": [
    {"text": "Alice uses Python daily", "shared": ["Alice", "Python"]}
  ]
}
```

The `resolved` field only appears when at least one entity is new. The `activated` field shows related memories triggered by associative activation.

### recall(entity?, action?, role?, memory_type?, mood?, negated?, since?, before?, limit?, min_strength?)

Query memories using the same grammar you stored with. At least one parameter required.

**Parameters:**
- `entity` (optional): Entity name or list of names. Single: focused lookup. List: finds memories involving ALL named entities.
- `action` (optional): Filter by verb.
- `role` (optional): Filter by kāraka role (`subject`/`object`/`instrument`/`recipient`/`source`/`locus`/`attribute`/`value`).
- `memory_type` (optional): `episodic` / `semantic` / `procedural`.
- `mood` (optional): `actual` / `planned` / `uncertain` / `normative` / `conditional`.
- `negated` (optional): `true` = only negated, `false` = only positive.
- `since` (optional): ISO date string — only memories after this date.
- `before` (optional): ISO date string — only memories before this date.
- `limit` (optional): Maximum results. Default: `10`.
- `min_strength` (optional): Minimum memory strength threshold. Default: `0.0`.

**Examples:**
- `recall(entity="Alice")` — everything about Alice
- `recall(entity="Alice", action="assign", role="subject")` — what Alice assigned
- `recall(entity="Bob", role="recipient")` — what was done TO Bob
- `recall(entity=["Alice", "API"])` — memories involving both
- `recall(mood="planned")` — all plans
- `recall(action="deploy", negated=true)` — what should NOT be deployed

**Returns:**
```json
{
  "count": 2,
  "memories": [
    {
      "text": "Alice prefers Python",
      "action": "prefers",
      "roles": {"subject": "Alice", "object": "Python"},
      "when": "2025-12-01T14:30:00",
      "reliability": "strong",
      "type": "semantic"
    }
  ]
}
```

Reliability labels: `strong` (>= 0.7), `moderate` (>= 0.4), `faint` (< 0.4).

### consolidate(entity?)

Merge similar entities and compress repeated memories. Phase 1 merges semantically similar entity nodes (cosine >= 0.95). Phase 2 groups edges sharing the same vertex set into summaries.

**Parameters:**
- `entity` (optional): Only consolidate memories involving this entity.

**Returns:** List of consolidated summaries.

Call periodically to keep memory efficient and to merge naming variants (e.g., "Bob" + "Robert").

### forget(older_than_days?, min_strength?, entity?)

Expire old or low-strength memories (soft delete).

**Parameters:**
- `older_than_days` (optional): Expire memories older than this many days.
- `min_strength` (optional): Expire memories below this strength threshold.
- `entity` (optional): Only forget memories involving this entity.

**Returns:** Number of memories expired.

## The Recall Mirror

Remember and recall use the same grammar. What you store is how you query.

```
Remember stores:              Recall queries:
──────────────────           ──────────────────
entities + roles        →    entity + role filter
action (verb)           →    action filter
memory_type             →    memory_type filter + type-specific decay
mood                    →    mood filter
negated                 →    negated filter
importance              →    strength ranking (salience)
confidence              →    strength ranking (confidence)
created_at              →    since / before filters
```

## Setup

Install hypabase:

```bash
pip install hypabase
# or
uv pip install hypabase
```

Environment variables:
- `HYPABASE_DB_PATH` — SQLite database path (default: `hypabase.db`)
- `HYPABASE_EMBEDDER` — Embedder for semantic search:
  - `fastembed` / `fast` / `default` — BAAI/bge-small-en-v1.5 via ONNX (default)
  - `openai` — text-embedding-3-small (requires `OPENAI_API_KEY`)
  - `sentence-transformers` / `st` / `local` — all-MiniLM-L6-v2
  - `none` — Disable embeddings
