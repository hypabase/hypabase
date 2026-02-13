"""Hypabase MCP server — exposes hypergraph operations as tools for AI agents."""

from hypabase.mcp.server import mcp, run_server

__all__ = ["mcp", "run_server"]
