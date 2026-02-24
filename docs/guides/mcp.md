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

The server exposes 7 memory tools.

| Tool | Description |
|------|-------------|
| `remember` | Store a memory: ACTION + ENTITIES in ROLES |
| `recall` | Recall memories by entity, action, role, type, mood, or time |
| `forget` | Expire old or low-strength memories (soft delete) |
| `consolidate` | Compress repeated episodic memories into semantic knowledge |
| `connections` | Explore an entity's neighborhood in the memory graph |
| `who_knows_what` | Summary of what the memory system knows |
| `resolve_contradiction` | Resolve a contradiction between two memories |

## Example workflow

A typical agent session:

1. **Remember** structured facts and events as they come up
2. **Recall** what the agent knows about an entity or topic
3. **Resolve contradictions** when new information conflicts with existing memories
4. **Consolidate** periodically to compress episodic clusters into semantic knowledge
5. **Forget** old or low-strength memories to keep the graph efficient

```
# Agent stores a memory
remember(
    action="assigned",
    entities=[
        {"name": "Alice", "type": "person", "role": "agent"},
        {"name": "API task", "type": "task", "role": "object"},
        {"name": "Bob", "type": "person", "role": "recipient"},
    ],
    memory_type="episodic",
    importance=0.7
)

# Later: agent recalls what it knows about Alice
recall(entity="Alice")

# Agent explores connections
connections(entity="Alice")

# Summary of everything in memory
who_knows_what()
```
