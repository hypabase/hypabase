"""Hypabase — A Python hypergraph library with provenance and SQLite persistence."""

__version__ = "0.2.3"

from hypabase.client import Hypabase
from hypabase.models import Edge, HypergraphStats, Incidence, Node, ValidationResult

__all__ = [
    "Edge",
    "Hypabase",
    "HypergraphStats",
    "Incidence",
    "Node",
    "ValidationResult",
    "__version__",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional modules."""
    if name == "Memory":
        from hypabase.memory import Memory

        return Memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
