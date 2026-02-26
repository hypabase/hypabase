"""Hypabase Memory MCP server -- 4 tools for AI agent persistent memory."""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from hypabase.client import Hypabase
from hypabase.memory import Memory
from hypabase.memory.types import KarakaRole, MemoryType, Mood

# All logging goes to stderr -- stdout is reserved for JSON-RPC
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("hypabase.memory")

# ---------------------------------------------------------------------------
# Singletons -- safe for single-process stdio MCP
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
        "You have persistent memory. Use PENMAN notation to store and recall facts.\n\n"
        "remember(penman='(verb :role entity ...)') -- store memories\n"
        "recall(entity='...') -- find memories\n"
        "consolidate() -- merge similar entities and compress memories\n"
        "forget(older_than_days=30) -- clean up\n\n"
        "Example: remember(penman='(prefers :subject Alice :object Python :memory_type semantic)')\n\n"
        "8 roles: :subject (who), :object (what), :recipient (to whom), "
        ":instrument (how), :origin (from where), :locus (where/when), "
        ":attribute (property), :value (its value).\n"
        "Nest atoms for beliefs/causes: (believes :subject X :object (is ...))\n"
        "Entity names are matched by normalized cache (exact match after lowercasing)."
    ),
    lifespan=app_lifespan,
)


def _get_memory() -> Memory:
    """Return the memory agent, raising if not initialized."""
    if _MEMORY is None:
        raise RuntimeError("Memory module is not enabled. Server lifespan has not run yet.")
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
# Output helpers
# ===================================================================


def _reliability_label(strength: float) -> str:
    """Map strength score to a word the agent can reason with."""
    if strength >= 0.7:
        return "strong"
    if strength >= 0.4:
        return "moderate"
    return "faint"


def _detect_contradictions(results: list[dict]) -> list[dict]:
    """Find memories that contradict each other (same action + entities, opposite negation)."""
    contradictions: list[dict] = []
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            if a.get("action") != b.get("action"):
                continue
            a_neg = a.get("negated", False)
            b_neg = b.get("negated", False)
            if a_neg == b_neg:
                continue
            # Same action, opposite negation -- check entity overlap
            a_entities = set(a.get("roles", {}).values()) if isinstance(a.get("roles"), dict) else set()
            b_entities = set(b.get("roles", {}).values()) if isinstance(b.get("roles"), dict) else set()
            # Flatten lists from multi-valued roles
            a_flat: set[str] = set()
            for v in a_entities:
                if isinstance(v, list):
                    a_flat.update(v)
                else:
                    a_flat.add(v)
            b_flat: set[str] = set()
            for v in b_entities:
                if isinstance(v, list):
                    b_flat.update(v)
                else:
                    b_flat.add(v)
            shared = a_flat & b_flat
            if len(shared) >= 1:
                pos, neg = (a, b) if not a_neg else (b, a)
                contradictions.append({
                    "positive": pos["text"],
                    "negative": neg["text"],
                    "shared": sorted(shared),
                })
    return contradictions


def _format_remember(raw: dict) -> dict:
    """Reshape agent.py remember() output for agent consumption."""
    memories = []
    for r in raw["edges"]:
        m: dict[str, Any] = {
            "text": r["text"],
            "action": r["action"],
            "roles": {e["role"]: e["name"] for e in r["entities"]},
        }
        if r.get("memory_type"):
            m["type"] = r["memory_type"]
        if r.get("mood") and r["mood"] != "actual":
            m["mood"] = r["mood"]
        # Entity recognition feedback: did the system know these entities?
        resolved = {e["name"]: e["status"] for e in r["entities"]}
        if any(v == "new" for v in resolved.values()):
            m["resolved"] = resolved
        memories.append(m)

    result: dict[str, Any] = {"stored": raw["stored"], "memories": memories}

    # Associative activation: what related memories were triggered?
    if raw.get("related"):
        result["activated"] = [
            {"text": r["text"], "shared": r["shared_entities"]}
            for r in raw["related"]
        ]
    return result


def _format_recall(results: list[dict]) -> dict:
    """Reshape agent.py recall() output for agent consumption."""
    memories = []
    for r in results:
        m: dict[str, Any] = {
            "text": r["text"],
            "action": r.get("action"),
            "roles": r.get("roles", {}),
            "when": r.get("created_at"),
            "reliability": _reliability_label(r["strength"]),
        }
        # Classification -- only include when set
        mt = r.get("memory_type")
        if mt:
            m["type"] = mt
        # Modality -- only include when non-default
        mood = r.get("mood", "actual")
        if mood != "actual":
            m["mood"] = mood
        if r.get("negated"):
            m["negated"] = True
        memories.append(m)

    result: dict[str, Any] = {"count": len(memories), "memories": memories}

    # Surface contradictions the agent should be aware of
    contradictions = _detect_contradictions(results)
    if contradictions:
        result["contradictions"] = contradictions

    return result


# ===================================================================
# Memory tools (4)
# ===================================================================


@mcp.tool()
@_safe_tool
def remember(
    penman: str,
    source: str = "memory",
    confidence: float = 1.0,
) -> dict:
    """Store memories as PENMAN atoms: (verb :role entity ...)

    FORMAT
    ------
    Each memory is a verb with participants in role slots:

        (verb :role "entity" :role "entity" ...)

    ROLES (fill in what applies, skip what doesn't)
    -----
    :subject     who or what it's about
    :object      what is acted on
    :recipient   who receives or benefits
    :instrument  tool, method, or means used
    :origin      where it came from, previous state
    :locus       where, when, or in what context
    :attribute   a named property or dimension
    :value       the specific value of that property

    MODIFIERS (metadata about the fact)
    ---------
    :tense        past, present, future
    :mood         actual (default), planned, uncertain, normative, conditional
    :negated      true or false
    :memory_type  episodic (events), semantic (facts), procedural (how-to)
    :importance   0.0 to 1.0

    CONTEXT (why, for what, under what condition -- can be nested)
    -------
    :cause       why it happened
    :purpose     what for
    :condition   if/when/unless

    NESTING
    -------
    Any slot can hold a nested atom instead of a string:

        (believes :subject Alice :object (is :subject deadline :value Friday))

    EXAMPLES
    --------
    Preference:
        (prefers :subject Alice :object Python :memory_type semantic)

    Event:
        (assigned :subject Alice :object "billing task" :recipient Bob
         :instrument Jira :locus Monday :tense past :memory_type episodic)

    Property:
        (has :subject "quick sort" :attribute "time complexity"
         :value "O(n log n)" :memory_type semantic)

    Negation:
        (uses :subject Django :object "Python 2" :negated true :memory_type semantic)

    Multiple facts:
        (deployed :subject Alice :object API :locus Monday :tense past)
        (reviewed :subject Bob :object API :locus Tuesday :tense past)

    Args:
        penman: One or more PENMAN atoms.
        source: Provenance source identifier.
        confidence: Confidence score between 0.0 and 1.0.
    """
    mem = _get_memory()
    raw = mem.remember(penman=penman, source=source, confidence=confidence)
    return _format_remember(raw)


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
    - entity: WHO/WHAT -- "Alice", "API", or ["Alice", "API"] for both
    - action: verb -- "assign", "decide", "deploy"
    - role: karaka role -- agent/object/instrument/recipient/source/locus

    Classification:
    - memory_type: "episodic" / "semantic" / "procedural"
    - mood: "actual" / "planned" / "uncertain" / "normative"
    - negated: true = negations only

    Temporal:
    - since / before: ISO date strings

    Examples:
    - recall(entity="Alice")                                 -- everything about Alice
    - recall(entity="Alice", action="assign", role="subject")  -- what Alice assigned
    - recall(entity="Bob", role="recipient")                 -- what was done TO Bob
    - recall(entity=["Alice", "API"])                        -- memories involving both
    - recall(mood="planned")                                 -- all plans
    - recall(action="deploy", negated=true)                  -- what should NOT be deployed

    Args:
        entity: Entity name or list of names for lookup.
        action: Filter by action type (the verb).
        role: Filter by karaka role (agent/object/instrument/recipient/source/locus).
        memory_type: Filter by memory type (episodic/semantic/procedural).
        mood: Filter by mood -- "actual", "planned", "uncertain", or "normative".
        negated: Filter -- true=only negated memories, false=only positive.
        since: Only memories created after this ISO date string.
        before: Only memories created before this ISO date string.
        limit: Maximum results to return.
        min_strength: Minimum memory strength threshold.
    """
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    before_dt = datetime.fromisoformat(before) if before else None

    results = mem.recall(
        entity=entity,
        action=action,
        role=cast(KarakaRole | None, role),
        memory_type=cast(MemoryType | None, memory_type),
        mood=cast(Mood | None, mood),
        negated=negated,
        since=since_dt,
        before=before_dt,
        limit=limit,
        min_strength=min_strength,
    )
    return _format_recall(results)


@mcp.tool()
@_safe_tool
def consolidate(entity: str | None = None) -> dict:
    """Merge similar entities and compress repeated memories.

    Phase 1: Merges semantically similar entity nodes (cosine >= 0.95).
    Phase 2: Groups edges sharing the same vertex set into summaries.

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
) -> dict:
    """Expire old or low-strength memories (soft delete).

    Args:
        older_than_days: Expire memories older than this many days.
        min_strength: Expire memories below this strength threshold.
        entity: Only forget memories involving this entity.
    """
    mem = _get_memory()
    older_than = None
    if older_than_days is not None:
        older_than = datetime.now(UTC) - timedelta(days=older_than_days)
    return mem.forget(
        older_than=older_than,
        min_strength=min_strength,
        entity=entity,
    )


# ===================================================================
# Entry point
# ===================================================================


def run() -> None:
    """Run the Hypabase Memory MCP server over stdio."""
    mcp.run(transport="stdio")
