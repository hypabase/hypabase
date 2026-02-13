# MCP Server

Hypabase ships an [MCP](https://modelcontextprotocol.io/) server that exposes the full hypergraph API as tools for AI agents. Any MCP-compatible client — Claude Code, Claude Desktop, Cursor, Windsurf, or custom agents — can create nodes, build hyperedges, query the graph, and traverse paths.

## Installation

Install Hypabase with the `mcp` extra:

```bash
uv add "hypabase[mcp]"
```

## Starting the server

The MCP server runs over stdio (JSON-RPC):

```bash
hypabase-mcp
```

By default it opens `hypabase.db` in the current directory. Set `HYPABASE_DB_PATH` to use a different file:

```bash
HYPABASE_DB_PATH=/path/to/knowledge.db hypabase-mcp
```

## Client configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hypabase": {
      "command": "hypabase-mcp",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

### Claude Code

Add to `.mcp.json` in your project root (shared with the team):

```json
{
  "mcpServers": {
    "hypabase": {
      "type": "stdio",
      "command": "hypabase-mcp",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

Or add via the CLI:

```bash
claude mcp add --transport stdio --env HYPABASE_DB_PATH=/path/to/knowledge.db hypabase -- hypabase-mcp
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "hypabase": {
      "command": "hypabase-mcp",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

### Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "hypabase": {
      "command": "hypabase-mcp",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

## Tools

The server exposes 14 tools across three categories.

### Node tools (4)

| Tool | Description |
|------|-------------|
| `create_node` | Create or update a node in the hypergraph |
| `get_node` | Get a node by its ID |
| `search_nodes` | Search for nodes by type and/or property values |
| `delete_node` | Delete a node and all its connected edges (cascade) |

### Edge tools (7)

| Tool | Description |
|------|-------------|
| `create_edge` | Create a hyperedge connecting two or more nodes |
| `batch_create_edges` | Create hyperedges in a single batch |
| `get_edge` | Get an edge by its ID |
| `search_edges` | Search for edges by contained nodes, type, provenance, or properties |
| `upsert_edge` | Create or update an edge by its exact set of nodes (idempotent) |
| `delete_edge` | Delete an edge by its ID |
| `lookup_edges_by_nodes` | O(1) lookup: find edges with exactly this set of nodes |

### Traversal & analysis tools (3)

| Tool | Description |
|------|-------------|
| `get_neighbors` | Find all nodes connected to a given node via shared edges |
| `find_paths` | Find paths between two nodes through hyperedges (BFS) |
| `get_stats` | Get database statistics, provenance sources, and available namespaces |

## Resources

The server also exposes 2 MCP resources:

| Resource URI | Description |
|--------------|-------------|
| `hypabase://schema` | Hypabase data model reference — nodes, edges, provenance, namespaces |
| `hypabase://stats` | Live database statistics and namespace listing |

## Namespace support

Every tool accepts an optional `database` parameter to scope operations to a namespace. This lets an agent maintain isolated graphs (e.g., separate knowledge domains) within a single database file:

```
create_node(id="aspirin", type="drug", database="pharma")
create_node(id="session_1", type="session", database="agent_memory")
```

## Example workflow

A typical agent session:

1. **Create nodes** for entities discovered during conversation
2. **Create edges** to record relationships between entities (with provenance)
3. **Search edges** to recall what the agent knows about a topic
4. **Find paths** to discover indirect connections
5. **Get stats** to understand the current state of the knowledge graph

```
# Agent discovers entities
create_node(id="alice", type="person")
create_node(id="project_x", type="project")
create_node(id="rust", type="language")

# Agent records a relationship
create_edge(
    nodes=["alice", "project_x", "rust"],
    type="works_on",
    source="conversation_2024_01_15",
    confidence=0.95
)

# Later: agent recalls what it knows about Alice
search_edges(containing=["alice"])

# Agent explores connections
get_neighbors(node_id="project_x")
find_paths(start="alice", end="rust")
```
