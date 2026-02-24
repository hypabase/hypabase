# Hypabase Memory Grammar v1

A complete, minimal grammar for structured AI agent memory — derived from
Pāṇini's Aṣṭādhyāyī and neuroscience memory models.

## Core Formula

```
MEMORY = ACTION + ENTITIES(ROLES) + TYPE + MOOD + IMPORTANCE + [NEGATED]
```

Every memory is an **action** (verb) performed by **entities** in **roles**,
classified by **type**, **mood**, and **importance**, with optional **negation**.

---

## 1. Action (dhātu)

The verb describing what happened. Always use base form (`assign` not `assigned`).

The action becomes the hyperedge type. Examples: `assign`, `prefer`, `deploy`,
`decide`, `create`, `is`.

---

## 2. Kāraka Roles (6)

Six semantic roles classify how each entity participates in a memory.
Derived from Pāṇini's kāraka system — proven complete for over 2,500 years.

| Role | Sanskrit | Meaning | Example |
|------|----------|---------|---------|
| `agent` | kartā | Who did it / who is described | Alice |
| `object` | karma | What was acted upon | the proposal |
| `instrument` | karaṇa | By what means | Slack, Docker |
| `recipient` | sampradāna | For/to whom; beneficiary; goal/target of transfer | Bob; GraphQL (in "migrate to GraphQL") |
| `source` | apādāna | From where; point of origin or departure | the old system; REST (in "migrate from REST") |
| `locus` | adhikaraṇa | Where/when; spatial or temporal context | sprint review, 3pm, Room 5 |

> **Recipient vs Locus for destinations:** Per Pāṇini's sūtra 1.4.32, the
> goal/target of a transfer or movement action is `recipient` (sampradāna),
> not `locus`. "Migrate **to** GraphQL" — GraphQL is `recipient`. "Meet
> **at** the office" — the office is `locus`. The test: "to X" = recipient,
> "at/in/on X" = locus.

> **Note:** The kāraka role `source` (apādāna — origin within the event) is
> distinct from the provenance field `source` (where the memory itself came
> from, e.g. "meeting_notes" or "user_input"). Context disambiguates: roles
> are on entities, provenance is on the memory.

### Role assignment rules

- Every memory has at least an **agent** and/or **object**.
- An entity's role depends on the action, not the entity itself.
  Alice can be `agent` in one memory and `recipient` in another.
- When the actor is unknown (passive voice), omit `agent` — the action
  and remaining roles still form a valid memory.
- `locus` covers both spatial and temporal context (adhikaraṇa in Sanskrit
  encompasses both). Use edge properties to distinguish if needed.

### Completeness notes

These 6 roles cover experiencers ("Alice fears spiders" — Alice is `agent`),
inanimate actors ("rain destroyed crops" — rain is `agent`), and comparatives
("Alice outperforms Bob" — Alice=`agent`, Bob=`object`). No additional roles
are needed.

---

## 3. Valence Templates (dhātu-kāraka mapping)

The action determines which roles are expected. 10 verb classes cover all
common agent memory patterns. Required roles are listed first; `[optional]`
roles follow.

```
TRANSFER     assign, give, send, delegate, share, forward, pass
             → agent + object + recipient + [instrument] + [locus]

STATE        is, has, equals, means, contains, belongs_to
             → agent + object

COGNITIVE    knows, believes, thinks, decided, learned, understands
             → agent + object + [source]

CREATION     created, built, wrote, designed, implemented, configured
             → agent + object + [instrument] + [locus]

DESTRUCTION  deleted, removed, deprecated, cancelled, disabled
             → agent + object + [instrument] + [locus]

COMMUNICATION said, told, asked, announced, reported, requested
             → agent + object + [recipient] + [locus]

MOVEMENT     moved, deployed, migrated, transferred, went
             → agent + [object] + [source] + [recipient] + [locus]

PREFERENCE   prefers, likes, wants, needs, chooses, avoids
             → agent + object

RELATION     manages, reports_to, works_with, depends_on, owns
             → agent + object

USAGE        uses, runs, executes, operates, applies
             → agent + object + [instrument]
```

**Default for unknown actions:** `agent + object` (transitive default).

> **Note:** Valence templates are guidance, not enforcement. The system does
> not reject memories with missing "required" roles — agents may omit roles
> when the information is unknown or irrelevant. "Required" means "strongly
> expected for complete memory" not "will fail without."

---

## 4. Memory Types (3)

Neuroscience-informed categories with distinct decay rates.

| Type | Use for | Decay rate | Half-life |
|------|---------|------------|-----------|
| `episodic` | Events, meetings, conversations | 0.15/day | ~4.6 days |
| `semantic` | Facts, preferences, definitions | 0.02/day | ~34.7 days |
| `procedural` | How-to, workflows, processes | 0.01/day | ~69.3 days |

### When to use which

- **Episodic**: Tied to a specific time/place. "Alice and Bob met on Tuesday."
- **Semantic**: True in general, not tied to an event. "Alice is the CEO."
- **Procedural**: Instructions or processes. "To deploy: run tests, then push."

Episodic memories that are reinforced through repetition should be consolidated
into semantic memories (automatic via `consolidate()`).

---

## 5. Mood (4 modalities)

The epistemic status of the memory — is it fact, plan, possibility, or
recommendation? Derived from Pāṇini's lakāra (tense-mood) system.

| Mood | Meaning | Example |
|------|---------|---------|
| `actual` | It is/was true (default) | "Alice deployed the API" |
| `planned` | It will happen | "Alice will deploy on Friday" |
| `uncertain` | It might be true | "Alice might attend the meeting" |
| `normative` | It should be true | "We should use Kubernetes" |

### Why mood matters

Without mood, these are indistinguishable in the graph:
- "Alice is the CEO" (actual fact)
- "Alice will be the CEO" (plan)
- "Alice should be the CEO" (recommendation)

All three would be: `action=is, agent=Alice, object=CEO, type=semantic`.
Mood is the only structural way to tell them apart.

### Default

`actual`. Most memories are about things that happened or are true.

---

## 6. Negation

A boolean flag indicating the memory asserts the OPPOSITE of the action.

| Field | Type | Default |
|-------|------|---------|
| `negated` | bool | `false` |

### Why negation matters

Without it, "Alice does NOT use Java" stored as `action=use, agent=Alice,
object=Java` becomes a false positive — the graph says Alice uses Java.

When `negated=true`:
- Structural queries know to invert the meaning
- Recall results are flagged as negative assertions
- The graph does not contain false positives

### Examples

```
"Alice does not use Java"
→ action=use, agent=Alice, object=Java, negated=true

"Don't deploy to production"
→ action=deploy, object=production, negated=true, mood=normative

"We decided not to use GraphQL"
→ action=decide, agent=we, object=GraphQL, negated=true
```

---

## 7. Importance

A float score (0.0–1.0) indicating how important the memory is. Maps to
`salience` in the strength formula.

| Score | Meaning | Example |
|-------|---------|---------|
| 1.0 | Critical | User preferences, key decisions |
| 0.8 | Important | Project facts, team structure |
| 0.5 | Relevant | Meeting notes, task updates (default) |
| 0.3 | Minor | Casual mentions, background context |

Default: `0.5` (relevant — the neutral midpoint). Marking a memory as critical
(1.0) doubles its salience relative to default. Marking it as minor (0.3) reduces
it. Unrated memories are neither boosted nor penalized.

---

## 8. Confidence

Provenance reliability score (0.0–1.0). How reliable is the source of this
memory?

- `1.0` — User stated directly, verified fact
- `0.8` — Inferred from reliable context
- `0.5` — Second-hand, unverified
- `0.3` — Speculative, low reliability

Default: `1.0`.

---

## 9. Decomposition Rule (ekavākyatā)

**One action, one memory.** If a statement contains multiple actions,
decompose into separate memories — one per verb.

This follows Pāṇini's ekavākyatā (sentence unity): a vākya (sentence) has
exactly one primary verb. Each verb produces one hyperedge.

### Example

> "Alice told Bob that they should migrate the API from REST to GraphQL."

This contains two actions: **tell** and **migrate**. Decompose:

```
Memory 1 (the communication event):
  action: "tell", type: episodic, mood: actual
  Alice=agent, Bob=recipient, "API migration plan"=object,
  "architecture review"=locus

Memory 2 (the recommendation):
  action: "migrate", type: semantic, mood: normative
  "the team"=agent, "API"=object, "REST"=source, "GraphQL"=recipient,
  "new gateway"=instrument
```

### Why decompose

- Each memory has one clear action and one clear set of roles
- Queries work precisely: "What did Alice tell Bob?" hits Memory 1;
  "What should be migrated?" hits Memory 2
- No ambiguity about which entities fill which roles
- The hypergraph naturally links the two memories through shared entities

### When NOT to decompose

Simple statements with one verb need no decomposition:
- "Alice assigned the task to Bob" → one memory
- "The API uses JWT for authentication" → one memory

---

## 10. Entity Identity (same_as)

When two names refer to the same entity, a `same_as` edge is created
between them in the hypergraph. This is NOT an external lookup — it is
a graph edge traversed by spreading activation like any other.

### How identity is detected

The EntityResolver checks three levels (in order):

1. **Normalization** — "Alice Smith" and "alice smith" → same node
2. **Alias detection** — "Bob" and "Bob Jones" → same_as edge
   (prefix/suffix match with minimum 3 characters)
3. **Embedding similarity** — "Robert" and "Bob" → same_as edge
   (cosine similarity above threshold, when embedder is available)

### Why this matters

Without same_as edges, graph traversal from "Bob" misses memories
stored under "Bob Jones". With same_as edges, spreading activation
traverses the link and finds both:

```
"Bob" ──same_as──→ "Bob Jones"
  │                    │
  │ (recall "Bob")     │ (stored under "Bob Jones")
  │                    │
  ▼                    ▼
"Bob met Alice"    "Bob Jones deployed the API"
```

Both memories are discovered through a single query for "Bob".

### Cross-language and synonym entities

The same mechanism handles translations and synonyms:
- "dark mode" ←same_as→ "dark theme" (detected via embedding similarity)
- Entity resolution creates the edge; spreading activation uses it
- No separate synonym system needed

---

## Complete Example

Natural language:
> "Alice told Bob that they should migrate the API from REST to GraphQL
> using the new gateway, at the architecture review on Tuesday."

Decomposed into two memories (one per action):

**Memory 1** — the communication event:
```
action:      "tell"
entities:
  - name: "Alice",              role: "agent"
  - name: "Bob",                role: "recipient"
  - name: "API migration plan", role: "object"
  - name: "architecture review", role: "locus"
  - name: "Tuesday",            role: "locus"
type:        "episodic"
mood:        "actual"
negated:     false
importance:  0.5
confidence:  1.0
```

**Memory 2** — the recommendation:
```
action:      "migrate"
entities:
  - name: "the team",           role: "agent"
  - name: "API",                role: "object"
  - name: "REST",               role: "source"
  - name: "GraphQL",            role: "recipient"
  - name: "new gateway",        role: "instrument"
type:        "semantic"
mood:        "normative"     # "should migrate"
negated:     false
importance:  0.8
confidence:  1.0
```

Now queryable from any angle:
- "What did Alice tell Bob?" → Memory 1 (agent=Alice, recipient=Bob)
- "What should be migrated?" → Memory 2 (mood=normative, action=migrate)
- "What happened at the architecture review?" → Memory 1 (locus)
- "What involves GraphQL?" → Memory 2 (entity=GraphQL)
- "What does Bob know?" → Memory 1 found via recipient role

---

## Excluded Concepts (with reasoning)

| Concept | What it is | Why excluded |
|---------|-----------|--------------|
| **Prātipadika** (lemmatization) | Normalize verb forms: "assigned"→"assign" | English irregular verbs require NLP dependency. Instructions ("use base form") + semantic search cover 95%+. |
| **Yogyatā** (semantic fitness) | Validate role compatibility | Requires domain-specific model. Agents rarely assign incompatible roles. |
| **Samāsa** (compounds) | Decompose compound entities | Entities-as-strings handles "Project Alpha" naturally. No decomposition needed. |
| **Kṛdanta** (derivation) | Relate "teach"↔"teacher" | Semantic embeddings handle derived forms. |
| **Sandhi** (sound combination) | Phonological combination rules | Not applicable to digital text. |
| **Vacana** (singular/plural) | Normalize "meeting"↔"meetings" | EntityResolver embedding match handles most cases. |
| **Conditional decomposition** | Structural if-then modeling | `mood=uncertain` or `mood=planned` + text is sufficient for v1. |

Each excluded concept has a specific mechanism (instructions, semantic search,
EntityResolver, or edge properties) that handles it without grammar support.

---

## Strength Formula

How memory strength is calculated for retrieval ranking:

```
strength = recency × frequency × salience × confidence

recency   = exp(-decay × age_days)        # decay varies by memory type
frequency = 1 + log(1 + access_count)     # reinforced by recall
salience  = importance score (0-1)        # set at storage time
confidence = provenance reliability (0-1) # set at storage time
```

Decay rates by memory type:
- Episodic: 0.15/day (fast fade)
- Semantic: 0.02/day (slow fade)
- Procedural: 0.01/day (most durable)

---

## Retrieval Architecture

Three channels, combined by strength ranking:

1. **Semantic search** — embedding similarity finds conceptually related
   memories. Handles synonyms, paraphrases, and partial matches implicitly.

2. **Graph traversal** — exact entity matching via hypergraph indexes.
   Handles structural queries (who, what, relationships).

3. **Spreading activation** — BFS from seed entities with decaying
   activation. Discovers indirect connections the other channels miss.

Synonym handling is NOT a separate system — it emerges from:
- Semantic search (embedding similarity)
- EntityResolver (normalization + alias + embedding match)
- `same_as` edges in the graph (traversed by spreading activation)

---

## Design Principles

1. **The hypergraph is the universal structure.** Entities, relationships,
   aliases, and knowledge are all nodes and edges. No external lookup tables.
   Entity identity lives in the graph as `same_as` edges, not in a cache.

2. **One action, one memory.** Complex statements decompose into multiple
   hyperedges, one per verb. Shared entities naturally link them.

3. **The verb determines the roles.** Valence templates make role assignment
   deterministic, not guesswork.

4. **Retrieval is activation, not search.** Spreading activation through
   the graph discovers what keyword matching cannot — including aliases,
   synonyms, and indirect connections.

5. **Types guide behavior.** Memory type controls decay. Mood controls
   interpretation. Negation controls polarity. Each field has one job.

6. **Minimal and complete.** 6 roles, 3 types, 4 moods, 1 negation flag,
   10 valence templates, 1 decomposition rule, 1 identity mechanism.
   No more, no less.
