"""Hypabase — The hypergraph database for AI."""

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
