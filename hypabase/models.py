"""Pydantic models for Hypabase public API.

These are thin wrappers over the core engine types (engine.core), providing
Pydantic validation and serialization for the client-facing API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Node(BaseModel):
    """An entity in the hypergraph.

    Each node has an ID, a type for classification, and optional key-value
    properties. Nodes are auto-created when referenced in an edge.
    """

    id: str
    type: str = "unknown"
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        parts = [f"Node({self.id!r}, type={self.type!r}"]
        if self.properties:
            parts.append(f", properties={self.properties!r}")
        parts.append(")")
        return "".join(parts)


class Incidence(BaseModel):
    """How a node or edge participates in a hyperedge.

    Each incidence links one node (or one edge reference) to an edge,
    with an optional direction. Exactly one of node_id or edge_ref_id
    must be set.
    """

    node_id: str | None = None
    edge_ref_id: str | None = None
    direction: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_node_or_edge_ref(self) -> Incidence:
        if self.node_id is None and self.edge_ref_id is None:
            raise ValueError("Incidence must have either node_id or edge_ref_id")
        if self.node_id is not None and self.edge_ref_id is not None:
            raise ValueError("Incidence cannot have both node_id and edge_ref_id")
        return self


class Edge(BaseModel):
    """A hyperedge: one relationship linking two or more nodes.

    Each edge has a type, provenance (source and confidence), and can carry
    arbitrary properties. Node order within the edge is preserved.
    """

    id: str
    type: str
    incidences: list[Incidence] = Field(default_factory=list)
    directed: bool = False
    source: str = "unknown"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        node_ids = [inc.node_id for inc in self.incidences if inc.node_id is not None]
        return (
            f"Edge({self.type!r}: {node_ids}, source={self.source!r}, confidence={self.confidence})"
        )

    @property
    def node_ids(self) -> list[str]:
        """Ordered list of node IDs (backward compat)."""
        return [inc.node_id for inc in self.incidences if inc.node_id is not None]

    @property
    def node_set(self) -> set[str]:
        """Deduplicated set of node IDs."""
        return {inc.node_id for inc in self.incidences if inc.node_id is not None}


class ValidationResult(BaseModel):
    """Result of a hypergraph consistency check.

    Contains a pass/fail flag, a list of errors, and a list of warnings
    found during validation.
    """

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HypergraphStats(BaseModel):
    """Summary counts for a hypergraph.

    Reports total node and edge counts, broken down by type.
    """

    node_count: int
    edge_count: int
    nodes_by_type: dict[str, int]
    edges_by_type: dict[str, int]
