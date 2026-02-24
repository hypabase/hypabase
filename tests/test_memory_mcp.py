"""Tests for Memory MCP tools."""

from __future__ import annotations

import pytest

import hypabase.memory.server as srv
from hypabase import Hypabase
from hypabase.memory import Memory
from tests.conftest import MockEmbedder


@pytest.fixture()
def memory_client(tmp_path):
    """Patch MCP server globals with a fresh Memory-enabled client."""
    old_client, old_memory = srv._CLIENT, srv._MEMORY
    hb = Hypabase(str(tmp_path / "test.db"))
    srv._CLIENT = hb
    srv._MEMORY = Memory(hb=hb)
    yield hb
    srv._CLIENT, srv._MEMORY = old_client, old_memory
    hb.close()


@pytest.fixture()
def memory_client_with_embedder(tmp_path):
    """Patch MCP server globals with a fresh Memory-enabled client + MockEmbedder."""
    old_client, old_memory = srv._CLIENT, srv._MEMORY
    embedder = MockEmbedder()
    hb = Hypabase(str(tmp_path / "test.db"), embedder=embedder)
    srv._CLIENT = hb
    srv._MEMORY = Memory(hb=hb, embedder=embedder)
    yield hb
    srv._CLIENT, srv._MEMORY = old_client, old_memory
    hb.close()


class TestToolRegistration:
    def test_memory_tools_registered(self):
        from hypabase.memory.server import mcp

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "remember" in tool_names
        assert "recall" in tool_names
        assert "forget" in tool_names
        assert "connections" in tool_names
        assert "who_knows_what" in tool_names
        assert "resolve_contradiction" in tool_names
        assert "consolidate" in tool_names

    def test_total_tool_count(self):
        from hypabase.memory.server import mcp

        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert names == {
            "remember", "recall", "forget", "consolidate",
            "connections", "who_knows_what", "resolve_contradiction",
        }


class TestMemoryToolsWithoutInit:
    def test_remember_without_memory(self):
        result = srv.remember(
            action="met",
            entities=[
                {"name": "Alice", "role": "agent"},
                {"name": "Bob", "role": "object"},
            ],
        )
        assert "error" in result
        assert "not enabled" in result["message"]

    def test_recall_without_memory(self):
        result = srv.recall(entity="Alice")
        assert "error" in result

    def test_forget_without_memory(self):
        result = srv.forget()
        assert "error" in result

    def test_connections_without_memory(self):
        result = srv.connections("Alice")
        assert "error" in result

    def test_who_knows_what_without_memory(self):
        result = srv.who_knows_what()
        assert "error" in result

    def test_consolidate_without_memory(self):
        result = srv.consolidate()
        assert "error" in result


class TestMemoryToolsIntegration:
    def test_remember_and_recall(self, memory_client_with_embedder):
        result = srv.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
                {"name": "park", "type": "location", "role": "locus"},
            ],
            text="Alice Smith met Bob Jones at the park",
        )
        assert "error" not in result
        assert result.get("edge_id") is not None

        result = srv.recall(entity="Alice Smith")
        assert "error" not in result
        assert result["count"] >= 1

        result = srv.connections("Alice Smith")
        assert "error" not in result
        assert result["entity"] == "Alice Smith"

        result = srv.who_knows_what()
        assert "error" not in result
        assert result["node_count"] >= 2

        result = srv.forget(older_than_days=0.0001)
        assert "error" not in result

    def test_forget_with_min_strength(self, memory_client):
        srv.remember(
            action="met",
            entities=[
                {"name": "Alice Smith", "type": "person", "role": "agent"},
                {"name": "Bob Jones", "type": "person", "role": "object"},
            ],
        )
        # New memories have strength ~1.0, so low threshold should not forget them
        result_low = srv.forget(min_strength=0.01)
        assert "error" not in result_low
        assert result_low["expired_count"] == 0
        # High threshold should forget them
        result = srv.forget(min_strength=2.0)
        assert "error" not in result
        assert result["expired_count"] >= 1

    def test_connections_nonexistent(self, memory_client):
        """Returns valid empty structure for nonexistent entity."""
        result = srv.connections("Nonexistent Entity")
        assert "error" not in result
        assert result["entity"] == "Nonexistent Entity"
        assert result["edge_count"] == 0
        assert result["neighbor_count"] == 0

    def test_forget_no_args_returns_error(self, memory_client):
        """forget() with no filters returns error dict."""
        result = srv.forget()
        assert result.get("error") is True
        assert "type" in result
        assert result["type"] == "ValueError"

    def test_remember_with_entities_and_action(self, memory_client):
        """remember() accepts structured entities with roles."""
        result = srv.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
                {"name": "Bob", "type": "person", "role": "recipient"},
            ],
            memory_type="episodic",
            importance=0.7,
        )
        assert "error" not in result
        assert result["edge_id"] is not None
        assert result["action"] == "assigned"

    def test_recall_with_entity_and_role(self, memory_client):
        """recall() filters by entity and kāraka role."""
        srv.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        result = srv.recall(entity="Alice", role="agent")
        assert "error" not in result
        assert result["count"] >= 1
        for m in result["memories"]:
            assert m["roles"].get("Alice") == "agent"

    def test_recall_with_action_filter(self, memory_client):
        """recall() filters by action type."""
        srv.remember(
            action="assigned",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "task", "type": "task", "role": "object"},
            ],
        )
        srv.remember(
            action="reviewed",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "code", "type": "artifact", "role": "object"},
            ],
        )
        result = srv.recall(entity="Alice", action="assigned")
        assert "error" not in result
        assert result["count"] >= 1
        assert all(m["action"] == "assigned" for m in result["memories"])

    def test_recall_multi_entity(self, memory_client):
        """recall(entity=[...]) finds memories involving multiple entities."""
        srv.remember(
            action="worked",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Bob", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        result = srv.recall(entity=["Alice", "Bob"])
        assert "error" not in result
        assert result["count"] >= 1

    def test_recall_by_mood(self, memory_client):
        """recall(mood=...) filters by mood."""
        srv.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "service", "type": "system", "role": "object"},
            ],
            mood="planned",
        )
        result = srv.recall(mood="planned")
        assert "error" not in result
        assert result["count"] >= 1
        assert all(m["mood"] == "planned" for m in result["memories"])

    def test_recall_no_params_returns_error(self, memory_client):
        """recall() with no params returns validation error."""
        result = srv.recall()
        assert result.get("error") is True
        assert result["type"] == "ValueError"

    def test_resolve_contradiction_tool(self, memory_client):
        """resolve_contradiction tool works through MCP."""
        srv.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Python", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        r2 = srv.remember(
            action="prefers",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "Java", "type": "language", "role": "object"},
            ],
            memory_type="semantic",
        )
        assert len(r2.get("contradictions", [])) >= 1
        old_id = r2["contradictions"][0]["existing_edge_id"]
        result = srv.resolve_contradiction(r2["edge_id"], old_id, "supersede")
        assert "error" not in result
        assert result["resolved"] is True

    def test_resolve_contradiction_without_memory(self):
        """resolve_contradiction returns error when memory not initialized."""
        result = srv.resolve_contradiction("e1", "e2", "supersede")
        assert "error" in result

    def test_consolidate_tool(self, memory_client):
        """consolidate() MCP tool works."""
        srv.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        srv.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        result = srv.consolidate()
        assert "error" not in result
        assert "summaries" in result

    def test_consolidate_with_entity(self, memory_client):
        """consolidate(entity=...) scopes to that entity."""
        srv.remember(
            action="met",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        srv.remember(
            action="met_again",
            entities=[
                {"name": "Alice", "type": "person"},
                {"name": "Bob", "type": "person"},
            ],
        )
        result = srv.consolidate(entity="Alice")
        assert "error" not in result
        assert "summaries" in result

    def test_recall_with_temporal_filter(self, memory_client):
        """recall with since/before ISO strings."""
        srv.remember(
            action="deploy",
            entities=[
                {"name": "Alice", "type": "person", "role": "agent"},
                {"name": "API", "type": "system", "role": "object"},
            ],
        )
        # since far in the past should return results
        result = srv.recall(entity="Alice", since="2020-01-01T00:00:00+00:00")
        assert "error" not in result
        assert result["count"] >= 1

        # since far in the future should return nothing
        result = srv.recall(entity="Alice", since="2099-01-01T00:00:00+00:00")
        assert "error" not in result
        assert result["count"] == 0
