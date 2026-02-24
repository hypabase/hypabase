"""Hypabase Memory MCP server — 7 tools for AI agent persistent memory."""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from hypabase.client import Hypabase
from hypabase.memory import Memory

# All logging goes to stderr — stdout is reserved for JSON-RPC
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("hypabase.memory")

# ---------------------------------------------------------------------------
# Singletons — safe for single-process stdio MCP
# ---------------------------------------------------------------------------

_CLIENT: Hypabase | None = None
_MEMORY: Memory | None = None


def _init_embedder() -> Any:
    """Initialize an embedder based on HYPABASE_EMBEDDER env var.

    Defaults to FastEmbed when unset. Use ``HYPABASE_EMBEDDER=none`` to disable.
    """
    embedder_type = os.environ.get("HYPABASE_EMBEDDER", "").lower()
    if embedder_type in ("none", "off", "disabled"):
        return None
    if embedder_type == "openai":
        try:
            from hypabase.engine.embeddings import OpenAIProvider
        except ImportError:
            logger.warning("openai not installed, embedder disabled")
            return None
        try:
            return OpenAIProvider()
        except Exception as exc:
            logger.warning("Failed to initialize OpenAI embedder: %s", exc)
            return None
    elif embedder_type in ("sentence-transformers", "st", "local"):
        try:
            from hypabase.engine.embeddings import SentenceTransformerProvider
        except ImportError:
            logger.warning("sentence-transformers not installed, embedder disabled")
            return None
        try:
            return SentenceTransformerProvider()
        except Exception as exc:
            logger.warning("Failed to initialize SentenceTransformer embedder: %s", exc)
            return None
    elif embedder_type in ("", "fastembed", "fast", "default"):
        try:
            from hypabase.engine.embeddings import FastEmbedProvider
        except ImportError:
            logger.warning("fastembed not installed, embedder disabled")
            return None
        try:
            return FastEmbedProvider()
        except Exception as exc:
            logger.warning("Failed to initialize FastEmbed embedder: %s", exc)
            return None
    else:
        logger.warning("Unknown HYPABASE_EMBEDDER value: %r, embedder disabled", embedder_type)
        return None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    global _CLIENT, _MEMORY
    db_path = os.environ.get("HYPABASE_DB_PATH", "hypabase.db")
    logger.info("Opening Hypabase database: %s", db_path)
    embedder = _init_embedder()
    _CLIENT = Hypabase(db_path, embedder=embedder)
    _MEMORY = Memory(hb=_CLIENT, embedder=embedder)
    logger.info("Memory server ready")

    try:
        yield {}
    finally:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None
        _MEMORY = None


mcp = FastMCP(
    "Hypabase Memory",
    instructions=(
        "You have persistent memory.\n"
        "- remember(action=\"...\", entities=[...]) — store a memory\n"
        "- recall(entity=\"...\") — find memories\n"
        "- forget(older_than_days=30) — clean up\n\n"
        "Example: remember(action=\"uses\", entities=["
        "{\"name\": \"Alice\", \"role\": \"agent\"}, "
        "{\"name\": \"Python\", \"role\": \"object\"}])\n\n"
        "Each memory is an ACTION with PARTICIPANTS in ROLES "
        "(agent=who, object=what, instrument=how, recipient=for whom, "
        "source=from where, locus=where/when).\n"
        "Entity names are fuzzy-matched — the system resolves aliases "
        "and similar names.\n"
        "Resolve contradictions when remember() flags them.\n"
        "Consolidate periodically to compress episodic clusters."
    ),
    lifespan=app_lifespan,
)


def _get_memory() -> Memory:
    """Return the memory agent, raising if not initialized."""
    if _MEMORY is None:
        raise RuntimeError(
            "Memory module is not enabled. "
            "Server lifespan has not run yet."
        )
    return _MEMORY


def _safe_tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except (ValueError, TypeError) as exc:
            logger.warning("Tool %s: invalid input: %s", fn.__name__, exc)
            return {"error": True, "category": "validation", "type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            logger.exception("Tool %s failed", fn.__name__)
            return {"error": True, "category": "internal", "type": type(exc).__name__, "message": str(exc)}
    return wrapper


# ===================================================================
# Memory tools (7)
# ===================================================================


@mcp.tool()
@_safe_tool
def remember(
    action: str,
    entities: list[dict[str, str]],
    text: str | None = None,
    memory_type: str | None = None,
    importance: float | None = None,
    mood: str | None = None,
    negated: bool = False,
    source: str = "memory",
    confidence: float = 1.0,
) -> dict:
    """Store a memory: ACTION + ENTITIES in ROLES.

    Every memory is a verb (action) with participants (entities) in semantic
    roles (kāraka). At least 2 entities required — it's a hyperedge.

    Examples:
        remember(action="prefers", entities=[
            {"name": "Alice", "role": "agent"},
            {"name": "Python", "role": "object"},
            {"name": "Java", "role": "source"},
        ])

        remember(action="assigned", entities=[
            {"name": "Alice", "type": "person", "role": "agent"},
            {"name": "API task", "type": "task", "role": "object"},
            {"name": "Bob", "type": "person", "role": "recipient"},
        ], memory_type="episodic", importance=0.7)

    Args:
        action: The verb (e.g. "assigned", "prefers", "deployed").
        entities: Participants — list of dicts with 'name' (required),
            'role' (agent/object/instrument/recipient/source/locus),
            and optional 'type' (default "entity").
        text: Optional human-readable form. Stored for display only.
        memory_type: "episodic" (events), "semantic" (facts), "procedural" (how-to).
        importance: 0.0-1.0 importance rating.
        mood: "actual", "planned", "uncertain", or "normative". Default: actual.
        negated: True if the memory is a negation (e.g. "does NOT use Java").
        source: Provenance source identifier.
        confidence: Confidence score between 0.0 and 1.0.
    """
    mem = _get_memory()
    return mem.remember(
        action=action,
        entities=entities,
        text=text,
        memory_type=memory_type,
        importance=importance,
        mood=mood,
        negated=negated,
        source=source,
        confidence=confidence,
    )


@mcp.tool()
@_safe_tool
def recall(
    entity: str | list[str] | None = None,
    action: str | None = None,
    role: str | None = None,
    memory_type: str | None = None,
    mood: str | None = None,
    negated: bool | None = None,
    since: str | None = None,
    before: str | None = None,
    limit: int = 10,
    min_strength: float = 0.0,
) -> dict:
    """Recall memories using the same grammar you stored with.

    Use the dimensions you know:
    - entity: WHO/WHAT — "Alice", "API", or ["Alice", "API"] for both
    - action: verb — "assign", "decide", "deploy"
    - role: kāraka role — agent/object/instrument/recipient/source/locus

    Classification:
    - memory_type: "episodic" / "semantic" / "procedural"
    - mood: "actual" / "planned" / "uncertain" / "normative"
    - negated: true = negations only

    Temporal:
    - since / before: ISO date strings

    Examples:
    - recall(entity="Alice")                                 — everything about Alice
    - recall(entity="Alice", action="assign", role="agent")  — what Alice assigned
    - recall(entity="Bob", role="recipient")                 — what was done TO Bob
    - recall(entity=["Alice", "API"])                        — memories involving both
    - recall(mood="planned")                                 — all plans
    - recall(action="deploy", negated=true)                  — what should NOT be deployed

    Args:
        entity: Entity name or list of names for lookup.
        action: Filter by action type (the verb).
        role: Filter by kāraka role (agent/object/instrument/recipient/source/locus).
        memory_type: Filter by memory type (episodic/semantic/procedural).
        mood: Filter by mood — "actual", "planned", "uncertain", or "normative".
        negated: Filter — true=only negated memories, false=only positive.
        since: Only memories created after this ISO date string.
        before: Only memories created before this ISO date string.
        limit: Maximum results to return.
        min_strength: Minimum memory strength threshold.
    """
    mem = _get_memory()
    # Parse ISO date strings
    since_dt = datetime.fromisoformat(since) if since else None
    before_dt = datetime.fromisoformat(before) if before else None

    results = mem.recall(
        entity=entity,
        action=action,
        role=role,
        memory_type=memory_type,
        mood=mood,
        negated=negated,
        since=since_dt,
        before=before_dt,
        limit=limit,
        min_strength=min_strength,
    )
    return {
        "count": len(results),
        "memories": [
            {
                "edge_id": r["edge"].id,
                "type": r["edge"].type,
                "node_ids": r["edge"].node_ids,
                "text": r["text"],
                "score": r["score"],
                "strength": r["strength"],
                "source": r["edge"].source,
                "confidence": r["edge"].confidence,
                "action": r.get("action"),
                "memory_type": r.get("memory_type"),
                "mood": r.get("mood", "actual"),
                "negated": r.get("negated", False),
                "roles": r.get("roles", {}),
            }
            for r in results
        ],
    }


@mcp.tool()
@_safe_tool
def consolidate(entity: str | None = None) -> dict:
    """Compress repeated episodic memories into semantic knowledge.

    Call periodically to keep memory efficient.

    Args:
        entity: Only consolidate memories involving this entity.
    """
    mem = _get_memory()
    return {"summaries": mem.consolidate(entity=entity)}


@mcp.tool()
@_safe_tool
def forget(
    older_than_days: float | None = None,
    min_strength: float | None = None,
    entity: str | None = None,
    memory_type: str | None = None,
    mood: str | None = None,
) -> dict:
    """Expire old or low-strength memories (soft delete).

    Args:
        older_than_days: Expire memories older than this many days.
        min_strength: Expire memories below this strength threshold.
        entity: Only forget memories involving this entity.
        memory_type: Only forget memories of this type (episodic/semantic/procedural).
        mood: Only forget memories of this mood (actual/planned/uncertain/normative).
    """
    mem = _get_memory()
    older_than = None
    if older_than_days is not None:
        older_than = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    return mem.forget(
        older_than=older_than,
        min_strength=min_strength,
        entity=entity,
        memory_type=memory_type,
        mood=mood,
    )


@mcp.tool()
@_safe_tool
def connections(
    entity: str,
    max_hops: int = 2,
    role: str | None = None,
) -> dict:
    """Explore an entity's neighborhood in the memory graph.

    Args:
        entity: The entity name to explore.
        max_hops: Maximum traversal depth.
        role: Filter to edges where any entity has this kāraka role.
    """
    mem = _get_memory()
    return mem.connections(entity, max_hops=max_hops, role=role)


@mcp.tool()
@_safe_tool
def who_knows_what() -> dict:
    """Summary of what the memory system knows.

    Returns entity counts, edge counts by type (including memory type breakdown),
    top entities by degree, and provenance sources.
    """
    mem = _get_memory()
    hb = mem.hb
    stats = hb.stats()
    sources = hb.sources()

    # Find top entities by edge count
    top_entities = [
        {"entity": node.id, "type": node.type, "connections": degree}
        for node, degree in hb.top_nodes_by_degree(20)
    ]

    # Memory type and mood breakdown (exclude infrastructure edges)
    all_edges = hb.edges(active=True)
    memory_edges = [e for e in all_edges if e.type not in ("same_as", "consolidated")]
    memory_types: dict[str, int] = {}
    moods: dict[str, int] = {}
    for e in memory_edges:
        mt = e.properties.get("memory_type")
        if mt:
            memory_types[mt] = memory_types.get(mt, 0) + 1
        m = e.properties.get("mood", "actual")
        moods[m] = moods.get(m, 0) + 1

    return {
        "node_count": stats.node_count,
        "edge_count": stats.edge_count,
        "memory_count": len(memory_edges),
        "nodes_by_type": stats.nodes_by_type,
        "edges_by_type": stats.edges_by_type,
        "memory_types": memory_types,
        "moods": moods,
        "sources": sources,
        "top_entities": top_entities,
    }


@mcp.tool()
@_safe_tool
def resolve_contradiction(
    new_edge_id: str,
    old_edge_id: str,
    resolution: str,
) -> dict:
    """Resolve a contradiction between two memories.

    Call this after ``remember()`` returns contradictions. The agent decides
    how to resolve the conflict.

    Args:
        new_edge_id: The edge ID of the newer (contradicting) memory.
        old_edge_id: The edge ID of the older (existing) memory.
        resolution: How to resolve — "supersede" (expire old and keep new),
            "keep_both" (both remain active), or "keep_old" (expire new).
    """
    mem = _get_memory()
    return mem.resolve_contradiction(new_edge_id, old_edge_id, resolution)


# ===================================================================
# Entry point
# ===================================================================


def run() -> None:
    """Run the Hypabase Memory MCP server over stdio."""
    mcp.run(transport="stdio")
