"""Hypabase — A Python hypergraph library with provenance and SQLite persistence."""

__version__ = "0.1.2"

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


def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "Memory":
        from hypabase.memory import Memory
        return Memory
    if name == "MemoryAgent":
        from hypabase.memory import MemoryAgent
        return MemoryAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
