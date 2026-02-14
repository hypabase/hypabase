# Agent Memory

Use Hypabase as persistent, structured memory for AI agents across sessions, with session-tagged provenance.

## Multi-session persistence

Hypabase persists to SQLite. An agent can write memory in one session and read it in the next:

```
from hypabase import Hypabase

# --- Session 1: Agent records task context ---
with Hypabase("agent_memory.db") as hb:
    with hb.context(source="session_1", confidence=0.9):
        hb.node("user_alice", type="user")
        hb.node("task_write_report", type="task")
        hb.node("doc_quarterly", type="document")
        hb.edge(
            ["user_alice", "task_write_report", "doc_quarterly"],
            type="assigned",
        )
```

```
# --- Session 2: Agent reopens, queries, adds new context ---
with Hypabase("agent_memory.db") as hb:
    # Session 1 data is still there
    alice_edges = hb.edges(containing=["user_alice"])
    # Returns the "assigned" edge from session 1

    with hb.context(source="session_2", confidence=0.85):
        hb.node("tool_spreadsheet", type="tool")
        hb.edge(
            ["user_alice", "task_write_report", "tool_spreadsheet"],
            type="uses_tool",
        )
```

```
# --- Session 3: Agent queries across all sessions ---
with Hypabase("agent_memory.db") as hb:
    # All data from all sessions
    assert len(hb.nodes()) == 4
    assert len(hb.edges()) == 2

    # Cross-session path discovery
    paths = hb.paths("doc_quarterly", "tool_spreadsheet")
    # doc_quarterly → user_alice → tool_spreadsheet (across sessions)

    # Track which session contributed what
    sources = hb.sources()
    # [
    #     {"source": "session_1", "edge_count": 1, "avg_confidence": 0.9},
    #     {"source": "session_2", "edge_count": 1, "avg_confidence": 0.85},
    # ]
```

## Key patterns

### Session tracking via provenance

Use `source` to track which session or agent interaction created each memory:

```
with hb.context(source=f"session_{session_id}", confidence=0.9):
    # All memories in this block are tagged with the session
    hb.edge([user, task, resource], type="context")
```

### Confidence decay

Lower confidence for older or uncertain memories:

```
# Fresh interaction — high confidence
with hb.context(source="session_current", confidence=0.95):
    hb.edge(["user", "preference_dark_mode"], type="prefers")

# Inferred from past behavior — lower confidence
with hb.context(source="inference_engine", confidence=0.6):
    hb.edge(["user", "preference_vim"], type="likely_prefers")
```

### Context retrieval

When the agent needs to recall context about an entity:

```
def get_agent_context(hb, entity_id, min_confidence=0.7):
    """Retrieve all high-confidence memories about an entity."""
    edges = hb.edges(
        containing=[entity_id],
        min_confidence=min_confidence,
    )
    neighbors = hb.neighbors(entity_id)
    return {
        "relationships": edges,
        "connected_entities": neighbors,
    }
```

### Decision traces

Record why the agent made a decision:

```
with hb.context(source="planning_step_3", confidence=0.88):
    hb.edge(
        ["decision_use_react", "requirement_speed", "constraint_team_skill"],
        type="decision_trace",
        properties={"reasoning": "React chosen due to team familiarity"},
    )
```

Later, the agent (or a human) can audit the decision:

```
decisions = hb.edges(type="decision_trace")
for d in decisions:
    print(f"Decision involved: {d.node_ids}")
    print(f"Source: {d.source}, Confidence: {d.confidence}")
    print(f"Reasoning: {d.properties.get('reasoning')}")
```
