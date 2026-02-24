"""Pure-Python vector math utilities for Hypabase.

Cosine similarity and struct-based embedding packing/unpacking.
No numpy dependency required.
"""

from __future__ import annotations

import math
import struct

# --- Utilities (used in tests and available for custom search logic) ---


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0.
    Returns 0.0 if either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary BLOB (little-endian float32).

    L2-normalizes the vector before packing so that cosine distance in
    sqlite-vec produces accurate scores regardless of the embedding provider.
    """
    if vec:
        mag = math.sqrt(sum(x * x for x in vec))
        if mag < 1e-30:
            raise ValueError(
                "Cannot normalize zero or near-zero vector for embedding storage. "
                f"Magnitude: {mag:.2e}. "
                "Check that your embedding provider is returning valid vectors."
            )
        vec = [x / mag for x in vec]
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_embedding(blob: bytes, dimension: int) -> list[float]:
    """Unpack a binary BLOB back into a float vector."""
    expected = dimension * 4
    if len(blob) != expected:
        raise ValueError(f"Expected {expected} bytes for dimension {dimension}, got {len(blob)}")
    return list(struct.unpack(f"<{dimension}f", blob))
