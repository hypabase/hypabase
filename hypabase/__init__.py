"""Hypabase — A Python hypergraph library with provenance and SQLite persistence."""

__version__ = "0.1.0"

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
