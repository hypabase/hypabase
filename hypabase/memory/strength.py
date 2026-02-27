"""Memory strength calculation for Hypabase Memory Module.

Combines recency, frequency, salience, and confidence into a single strength score.
"""

from __future__ import annotations

import math
import time

from hypabase.memory.types import DEFAULT_DECAY_RATE, MEMORY_DECAY_RATES


def memory_strength(
    *,
    created_at: float | None = None,
    access_count: int = 0,
    confidence: float = 1.0,
    salience: float = 1.0,
    decay: float = DEFAULT_DECAY_RATE,
    memory_type: str | None = None,
    now: float | None = None,
) -> float:
    """Calculate memory strength.

    Formula: exp(-decay * age_days) * (1 + log(1 + access_count)) * salience * confidence

    Args:
        created_at: Unix timestamp when the memory was created.
        access_count: Number of times the memory has been accessed.
        confidence: Provenance confidence score (0-1).
        salience: Importance weight (0-1, default 1.0).
        decay: Exponential decay rate per day (default 0.1).
        memory_type: If set and decay is at default, use the type-specific
            decay rate (episodic=0.15, semantic=0.02, procedural=0.01).
        now: Current time as Unix timestamp. Defaults to time.time().

    Returns:
        Strength score (non-negative float). Higher = stronger memory.
    """
    confidence = max(0.0, min(1.0, confidence))
    salience = max(0.0, salience)

    # Use memory-type-specific decay when caller hasn't overridden the default
    if memory_type is not None and decay == DEFAULT_DECAY_RATE:
        decay = MEMORY_DECAY_RATES.get(memory_type, DEFAULT_DECAY_RATE)

    if now is None:
        now = time.time()

    # Recency: exponential decay based on age in days
    if created_at is not None:
        age_days = max(0.0, (now - created_at)) / 86400.0
        recency = math.exp(-decay * age_days)
    else:
        recency = 1.0

    # Frequency: baseline of 1.0 + logarithmic scaling of access count
    frequency = 1.0 + math.log(1 + access_count)

    # Combine all factors
    return recency * frequency * salience * confidence
