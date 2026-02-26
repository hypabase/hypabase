"""Type definitions for the Hypabase Memory Module.

Kāraka roles (Sanskrit semantic cases) classify how entities participate
in a memory.  Memory types (neuroscience-informed) control decay rates.
"""

from __future__ import annotations

from typing import Literal, get_args

# ---------------------------------------------------------------------------
# Kāraka roles — semantic roles on each participant in a memory
# ---------------------------------------------------------------------------

KarakaRole = Literal["subject", "object", "instrument", "recipient", "source", "locus", "attribute", "value"]

KARAKA_ROLES: set[str] = set(get_args(KarakaRole))

KARAKA_LABELS: dict[str, str] = {
    "subject": "kartā",
    "object": "karma",
    "instrument": "karaṇa",
    "recipient": "sampradāna",
    "source": "apādāna",
    "locus": "adhikaraṇa",
    "attribute": "viśeṣaṇa",
    "value": "māna",
}

# ---------------------------------------------------------------------------
# Memory types — neuroscience-informed categories with distinct decay rates
# ---------------------------------------------------------------------------

MemoryType = Literal["episodic", "semantic", "procedural"]

MEMORY_TYPES: set[str] = set(get_args(MemoryType))

MEMORY_DECAY_RATES: dict[str, float] = {
    "episodic": 0.15,  # Events — fade fast
    "semantic": 0.02,  # Facts — persist
    "procedural": 0.01,  # How-to — most durable
}

DEFAULT_DECAY_RATE: float = 0.1

# ---------------------------------------------------------------------------
# Mood — modality of the memory (what kind of truth it represents)
# ---------------------------------------------------------------------------

Mood = Literal["actual", "planned", "uncertain", "normative", "conditional"]

MOODS: set[str] = set(get_args(Mood))

DEFAULT_MOOD: str = "actual"

# ---------------------------------------------------------------------------
# Tense — temporal framing of the memory
# ---------------------------------------------------------------------------

Tense = Literal["past", "present", "future"]

# ---------------------------------------------------------------------------
# Role weights — used by spreading activation for role-weighted propagation
# ---------------------------------------------------------------------------

ROLE_WEIGHTS: dict[str, float] = {
    "subject": 1.0,
    "object": 0.9,
    "recipient": 0.8,
    "instrument": 0.7,
    "source": 0.6,
    "attribute": 0.5,
    "value": 0.5,
    "locus": 0.4,
}

DEFAULT_ROLE_WEIGHT: float = 0.5

# ---------------------------------------------------------------------------
# PENMAN → kāraka mapping (external names → internal kāraka names)
# ---------------------------------------------------------------------------

PENMAN_TO_KARAKA: dict[str, str] = {
    "subject": "subject",
    "object": "object",
    "instrument": "instrument",
    "recipient": "recipient",
    "origin": "source",
    "locus": "locus",
    "attribute": "attribute",
    "value": "value",
}
