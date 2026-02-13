"""Allow running the MCP server via ``python -m hypabase.mcp``."""

from hypabase.mcp.server import run_server

if __name__ == "__main__":
    run_server()
