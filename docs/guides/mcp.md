# MCP Server

Hypabase ships an [MCP](https://modelcontextprotocol.io/) server that gives AI agents persistent, structured memory. Any MCP-compatible client — Claude Code, Claude Desktop, Cursor, Windsurf, or custom agents — can store memories, recall them, and explore connections.

## Installation

```bash
uv add hypabase
```

## Starting the server

The MCP server runs over stdio (JSON-RPC):

```bash
hypabase-memory
```

By default it opens `hypabase.db` in the current directory. Set `HYPABASE_DB_PATH` to use a different file:

```bash
HYPABASE_DB_PATH=/path/to/knowledge.db hypabase-memory
```

## Client configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hypabase-memory": {
      "command": "hypabase-memory",
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
    "hypabase-memory": {
      "type": "stdio",
      "command": "hypabase-memory",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

Or add via the CLI:

```bash
claude mcp add --transport stdio --env HYPABASE_DB_PATH=/path/to/knowledge.db hypabase-memory -- hypabase-memory
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "hypabase-memory": {
      "command": "hypabase-memory",
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
    "hypabase-memory": {
      "command": "hypabase-memory",
      "env": {
        "HYPABASE_DB_PATH": "/path/to/knowledge.db"
      }
    }
  }
}
```

## Tools

The server exposes 4 memory tools.

| Tool | Description |
|------|-------------|
| `remember` | Store memories as PENMAN atoms: `(verb :role entity ...)` |
| `recall` | Recall memories by entity, action, role, type, mood, or time |
| `consolidate` | Merge similar entities and compress repeated memories |
| `forget` | Expire old or low-strength memories (soft delete) |

## Example workflow

A typical agent session:

1. **Remember** structured facts and events using PENMAN notation
2. **Recall** what the agent knows about an entity or topic
3. **Consolidate** periodically to merge naming variants and compress episodic clusters
4. **Forget** old or low-strength memories to keep the graph efficient

```
# Agent stores a memory using PENMAN notation
remember(penman='(assigned :subject Alice :object "API task" :recipient Bob :memory_type episodic :importance 0.7)')

# Later: agent recalls what it knows about Alice
recall(entity="Alice")

# What did Alice assign?
recall(entity="Alice", action="assign", role="subject")

# Merge naming variants and compress repeated memories
consolidate()

# Clean up old memories
forget(older_than_days=90, min_strength=0.3)
```
