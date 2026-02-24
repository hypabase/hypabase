"""Type definitions for the Hypabase Memory Module.

Kāraka roles (Sanskrit semantic cases) classify how entities participate
in a memory.  Memory types (neuroscience-informed) control decay rates.
"""

from __future__ import annotations

from typing import Literal, get_args

# ---------------------------------------------------------------------------
# Kāraka roles — semantic roles on each participant in a memory
# ---------------------------------------------------------------------------

KarakaRole = Literal["agent", "object", "instrument", "recipient", "source", "locus"]

KARAKA_ROLES: set[str] = set(get_args(KarakaRole))

KARAKA_LABELS: dict[str, str] = {
    "agent": "kartā",
    "object": "karma",
    "instrument": "karaṇa",
    "recipient": "sampradāna",
    "source": "apādāna",
    "locus": "adhikaraṇa",
}

# ---------------------------------------------------------------------------
# Memory types — neuroscience-informed categories with distinct decay rates
# ---------------------------------------------------------------------------

MemoryType = Literal["episodic", "semantic", "procedural"]

MEMORY_TYPES: set[str] = set(get_args(MemoryType))

MEMORY_DECAY_RATES: dict[str, float] = {
    "episodic": 0.15,   # Events — fade fast
    "semantic": 0.02,   # Facts — persist
    "procedural": 0.01, # How-to — most durable
}

DEFAULT_DECAY_RATE: float = 0.1

# ---------------------------------------------------------------------------
# Mood — modality of the memory (what kind of truth it represents)
# ---------------------------------------------------------------------------

Mood = Literal["actual", "planned", "uncertain", "normative"]

MOODS: set[str] = set(get_args(Mood))

DEFAULT_MOOD: str = "actual"

# ---------------------------------------------------------------------------
# Role weights — used by spreading activation for role-weighted propagation
# ---------------------------------------------------------------------------

ROLE_WEIGHTS: dict[str, float] = {
    "agent": 1.0,
    "object": 0.9,
    "recipient": 0.8,
    "instrument": 0.7,
    "source": 0.6,
    "locus": 0.4,
}

DEFAULT_ROLE_WEIGHT: float = 0.5

# ---------------------------------------------------------------------------
# Resolution action types
# ---------------------------------------------------------------------------

ResolutionAction = Literal["supersede", "keep_both", "keep_old"]
