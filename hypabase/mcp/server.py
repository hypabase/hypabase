"""Hypabase MCP server — exposes hypergraph operations as tools for AI agents."""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from hypabase.client import Hypabase

# All logging goes to stderr — stdout is reserved for JSON-RPC
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("hypabase.mcp")

# ---------------------------------------------------------------------------
# Client singleton — safe for single-process stdio MCP
# ---------------------------------------------------------------------------

_CLIENT: Hypabase | None = None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    global _CLIENT
    db_path = os.environ.get("HYPABASE_DB_PATH", "hypabase.db")
    logger.info("Opening Hypabase database: %s", db_path)
    _CLIENT = Hypabase(db_path)
    try:
        yield {}
    finally:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None


mcp = FastMCP(
    "Hypabase",
    instructions=(
        "Hypabase is a hypergraph store. "
        "Key behaviors: Edges connect 2+ nodes. "
        "Nodes are auto-created when referenced in an edge — you don't need to create them first. "
        "Every edge carries a source string and a confidence score (0-1) for provenance tracking. "
        "Use the database parameter on any tool to scope operations to a namespace. "
        "Use upsert_edge for idempotent writes (matches by vertex set + type)."
    ),
    lifespan=app_lifespan,
)


def _get_client(database: str | None = None) -> Hypabase:
    """Return the active Hypabase client, optionally scoped to a namespace."""
    if _CLIENT is None:
        raise RuntimeError("Hypabase client is not initialized")
    if database:
        return _CLIENT.database(database)
    return _CLIENT


def _safe_tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception("Tool %s failed", fn.__name__)
            return {"error": True, "message": f"{type(exc).__name__}: {exc}"}
    return wrapper


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _node_dict(node: Any) -> dict:
    return {
        "id": node.id,
        "type": node.type,
        "properties": node.properties,
    }


def _edge_dict(edge: Any) -> dict:
    return {
        "id": edge.id,
        "type": edge.type,
        "node_ids": edge.node_ids,
        "directed": edge.directed,
        "source": edge.source,
        "confidence": edge.confidence,
        "properties": edge.properties,
    }


# ===================================================================
# Node tools (4)
# ===================================================================


@mcp.tool()
@_safe_tool
def create_node(
    id: str,
    type: str = "unknown",
    properties: dict[str, Any] | None = None,
    database: str | None = None,
) -> dict:
    """Create or update a node in the hypergraph.

    Args:
        id: Unique node identifier.
        type: Node classification (e.g. "person", "medication").
        properties: Arbitrary key-value metadata for the node.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    props = properties or {}
    node = hb.node(id, type=type, **props)
    return _node_dict(node)


@mcp.tool()
@_safe_tool
def get_node(
    id: str,
    database: str | None = None,
) -> dict:
    """Get a node by its ID.

    Args:
        id: The node ID to look up.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    node = hb.get_node(id)
    if node is None:
        return {"found": False, "id": id}
    return _node_dict(node)


@mcp.tool()
@_safe_tool
def search_nodes(
    type: str | None = None,
    properties: dict[str, Any] | None = None,
    database: str | None = None,
) -> dict:
    """Search for nodes by type and/or property values.

    Args:
        type: Filter to nodes of this type.
        properties: Key-value pairs that must match node properties.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    if properties:
        results = hb.find_nodes(**properties)
        if type:
            results = [n for n in results if n.type == type]
    else:
        results = hb.nodes(type=type)
    return {"count": len(results), "nodes": [_node_dict(n) for n in results]}


@mcp.tool()
@_safe_tool
def delete_node(
    id: str,
    database: str | None = None,
) -> dict:
    """Delete a node and all its connected edges (cascade).

    Args:
        id: The node ID to delete.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    edges_removed = len(hb.edges_of_node(id))
    deleted = hb.delete_node(id, cascade=True)
    return {"deleted": deleted, "edges_removed": edges_removed if deleted else 0}


# ===================================================================
# Edge tools (7)
# ===================================================================


@mcp.tool()
@_safe_tool
def create_edge(
    nodes: list[str],
    type: str,
    source: str = "mcp",
    confidence: float = 1.0,
    directed: bool = False,
    properties: dict[str, Any] | None = None,
    database: str | None = None,
) -> dict:
    """Create a hyperedge connecting two or more nodes.

    Nodes are auto-created if they don't exist. This is a hypergraph — one edge
    can connect any number of nodes (not just pairs).

    Args:
        nodes: List of node IDs to connect (minimum 2).
        type: Edge type (e.g. "treatment", "knows").
        source: Provenance source identifier.
        confidence: Confidence score between 0.0 and 1.0.
        directed: If true, first node is tail, last node is head.
        properties: Arbitrary key-value metadata for the edge.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    edge = hb.edge(
        nodes,
        type=type,
        source=source,
        confidence=confidence,
        directed=directed,
        properties=properties,
    )
    return _edge_dict(edge)


@mcp.tool()
@_safe_tool
def batch_create_edges(
    edges: list[dict[str, Any]],
    source: str = "mcp",
    confidence: float = 1.0,
    database: str | None = None,
) -> dict:
    """Create multiple hyperedges in a single batch for efficient bulk ingestion.

    Each entry in ``edges`` must have ``nodes`` (list of node IDs) and ``type``.
    Optional per-edge overrides: ``source``, ``confidence``, ``directed``, ``properties``.
    Top-level ``source`` and ``confidence`` are defaults applied when an edge omits them.

    Args:
        edges: List of edge specifications. Each dict must contain:
            - nodes: list of node IDs (minimum 2)
            - type: edge classification string
            Optional keys: source, confidence, directed, properties
        source: Default provenance source for edges that don't specify one.
        confidence: Default confidence score for edges that don't specify one.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    created = []
    with hb.batch():
        for spec in edges:
            edge = hb.edge(
                spec["nodes"],
                type=spec["type"],
                source=spec.get("source", source),
                confidence=spec.get("confidence", confidence),
                directed=spec.get("directed", False),
                properties=spec.get("properties"),
            )
            created.append(_edge_dict(edge))
    return {"count": len(created), "edges": created}


@mcp.tool()
@_safe_tool
def get_edge(
    id: str,
    database: str | None = None,
) -> dict:
    """Get an edge by its ID.

    Args:
        id: The edge ID to look up.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    edge = hb.get_edge(id)
    if edge is None:
        return {"found": False, "id": id}
    return _edge_dict(edge)


@mcp.tool()
@_safe_tool
def search_edges(
    containing: list[str] | None = None,
    type: str | None = None,
    source: str | None = None,
    min_confidence: float | None = None,
    match_all: bool = False,
    properties: dict[str, Any] | None = None,
    database: str | None = None,
) -> dict:
    """Search for edges by contained nodes, type, provenance, or properties.

    All filters are combined with AND logic.

    Args:
        containing: Node IDs that must appear in the edge.
        type: Filter to edges of this type.
        source: Filter to edges from this provenance source.
        min_confidence: Filter to edges with confidence >= this value.
        match_all: If true with containing, edges must contain ALL specified nodes.
        properties: Key-value pairs that must match edge properties.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    if properties:
        results = hb.find_edges(**properties)
        if type:
            results = [e for e in results if e.type == type]
        if containing:
            containing_set = set(containing)
            if match_all:
                results = [e for e in results if containing_set.issubset(e.node_set)]
            else:
                results = [e for e in results if containing_set & e.node_set]
        if source is not None:
            results = [e for e in results if e.source == source]
        if min_confidence is not None:
            results = [e for e in results if e.confidence >= min_confidence]
    else:
        results = hb.edges(
            containing=containing,
            type=type,
            source=source,
            min_confidence=min_confidence,
            match_all=match_all,
        )
    return {"count": len(results), "edges": [_edge_dict(e) for e in results]}


@mcp.tool()
@_safe_tool
def upsert_edge(
    nodes: list[str],
    type: str,
    source: str = "mcp",
    confidence: float = 1.0,
    properties: dict[str, Any] | None = None,
    database: str | None = None,
) -> dict:
    """Create or update an edge by its exact set of nodes (idempotent).

    If an edge with the same vertex set and type already exists, it is updated.
    Otherwise a new edge is created. Useful for repeated ingestion.

    Args:
        nodes: List of node IDs for the edge (order-independent for matching).
        type: Edge type.
        source: Provenance source identifier.
        confidence: Confidence score between 0.0 and 1.0.
        properties: Key-value metadata (merged on update).
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    edge = hb.upsert_edge_by_vertex_set(
        node_ids=set(nodes),
        edge_type=type,
        properties=properties,
        source=source,
        confidence=confidence,
    )
    return _edge_dict(edge)


@mcp.tool()
@_safe_tool
def delete_edge(
    id: str,
    database: str | None = None,
) -> dict:
    """Delete an edge by its ID.

    Args:
        id: The edge ID to delete.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    deleted = hb.delete_edge(id)
    return {"deleted": deleted}


@mcp.tool()
@_safe_tool
def lookup_edges_by_nodes(
    nodes: list[str],
    database: str | None = None,
) -> dict:
    """O(1) lookup: find edges with exactly this set of nodes.

    Uses a hash index for constant-time lookup. Node order does not matter.

    Args:
        nodes: The exact set of node IDs to match.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    results = hb.edges_by_vertex_set(nodes)
    return {"count": len(results), "edges": [_edge_dict(e) for e in results]}


# ===================================================================
# Traversal & Analysis tools (3)
# ===================================================================


@mcp.tool()
@_safe_tool
def get_neighbors(
    node_id: str,
    edge_types: list[str] | None = None,
    database: str | None = None,
) -> dict:
    """Find all nodes connected to a given node via shared edges.

    Args:
        node_id: The node to find neighbors of.
        edge_types: If provided, only traverse edges of these types.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    results = hb.neighbors(node_id, edge_types=edge_types)
    return {"count": len(results), "nodes": [_node_dict(n) for n in results]}


@mcp.tool()
@_safe_tool
def find_paths(
    start: str,
    end: str,
    max_hops: int = 5,
    edge_types: list[str] | None = None,
    database: str | None = None,
) -> dict:
    """Find paths between two nodes through hyperedges.

    Uses breadth-first search. Each path is a list of node IDs from start to end.

    Args:
        start: Starting node ID.
        end: Target node ID.
        max_hops: Maximum number of hops (default 5).
        edge_types: If provided, only traverse edges of these types.
        database: Optional namespace to scope the operation.
    """
    hb = _get_client(database)
    results = hb.paths(start, end, max_hops=max_hops, edge_types=edge_types)
    return {"count": len(results), "paths": results}


@mcp.tool()
@_safe_tool
def get_stats(
    database: str | None = None,
) -> dict:
    """Get database statistics, provenance sources, and available namespaces.

    Args:
        database: Optional namespace to scope node/edge stats.
    """
    hb = _get_client(database)
    stats = hb.stats()
    sources = hb.sources()
    # _CLIENT is guaranteed non-None here (_get_client raises otherwise)
    databases = _CLIENT.databases()  # type: ignore[union-attr]
    return {
        "node_count": stats.node_count,
        "edge_count": stats.edge_count,
        "nodes_by_type": stats.nodes_by_type,
        "edges_by_type": stats.edges_by_type,
        "sources": sources,
        "databases": databases,
    }


# ===================================================================
# Resources (2)
# ===================================================================


@mcp.resource("hypabase://schema")
def schema_resource() -> str:
    """Hypabase data model reference.

    Describes the core concepts: nodes, edges, provenance, and namespaces.
    """
    return (
        "# Hypabase Data Model\n\n"
        "## Nodes\n"
        "Each node has an `id` (string), a `type` (string), and arbitrary `properties` (dict).\n\n"
        "## Edges (Hyperedges)\n"
        "A hyperedge connects 2 or more nodes in a single atomic relationship.\n"
        "- `id`: Unique edge identifier\n"
        "- `type`: Classification string (e.g. 'treatment', 'knows')\n"
        "- `node_ids`: Ordered list of connected node IDs\n"
        "- `directed`: If true, first node is tail, last is head\n"
        "- `properties`: Arbitrary key-value metadata\n\n"
        "## Provenance\n"
        "Every edge carries provenance metadata:\n"
        "- `source`: Origin identifier (e.g. 'clinical_records', 'llm_extraction')\n"
        "- `confidence`: Float score between 0.0 and 1.0\n\n"
        "## Namespaces\n"
        "Data is isolated into namespaces (databases). "
        "Pass `database` to any tool to scope operations. "
        "Default namespace is 'default'.\n"
    )


@mcp.resource("hypabase://stats")
def stats_resource() -> str:
    """Live database statistics and namespace listing."""
    if _CLIENT is None:
        raise RuntimeError("Hypabase client is not initialized")
    stats = _CLIENT.stats()
    sources = _CLIENT.sources()
    databases = _CLIENT.databases()
    lines = [
        "# Hypabase Statistics\n",
        f"Nodes: {stats.node_count}",
        f"Edges: {stats.edge_count}",
    ]
    if stats.nodes_by_type:
        lines.append("\n## Nodes by Type")
        for t, c in stats.nodes_by_type.items():
            lines.append(f"- {t}: {c}")
    if stats.edges_by_type:
        lines.append("\n## Edges by Type")
        for t, c in stats.edges_by_type.items():
            lines.append(f"- {t}: {c}")
    if sources:
        lines.append("\n## Provenance Sources")
        for s in sources:
            lines.append(
                f"- {s['source']}: {s['edge_count']} edges, "
                f"avg confidence {s['avg_confidence']}"
            )
    lines.append(f"\n## Databases\n{', '.join(databases)}")
    return "\n".join(lines)


# ===================================================================
# Entry point
# ===================================================================


def run_server() -> None:
    """Run the Hypabase MCP server over stdio."""
    mcp.run(transport="stdio")
