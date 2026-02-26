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
        assert "consolidate" in tool_names

    def test_total_tool_count(self):
        from hypabase.memory.server import mcp

        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"remember", "recall", "forget", "consolidate"}


class TestMemoryToolsWithoutInit:
    def test_remember_without_memory(self):
        result = srv.remember(
            penman="(met :subject Alice :object Bob)",
        )
        assert "error" in result
        assert "not enabled" in result["message"]

    def test_recall_without_memory(self):
        result = srv.recall(entity="Alice")
        assert "error" in result

    def test_forget_without_memory(self):
        result = srv.forget()
        assert "error" in result

    def test_consolidate_without_memory(self):
        result = srv.consolidate()
        assert "error" in result


class TestMemoryToolsIntegration:
    def test_remember_and_recall(self, memory_client_with_embedder):
        result = srv.remember(
            penman='(met :subject "Alice Smith" :object "Bob Jones" :locus park)',
        )
        assert "error" not in result
        assert result["stored"] == 1

        result = srv.recall(entity="Alice Smith")
        assert "error" not in result
        assert result["count"] >= 1

        result = srv.forget(older_than_days=0.0001)
        assert "error" not in result

    def test_forget_with_min_strength(self, memory_client):
        srv.remember(
            penman='(met :subject "Alice Smith" :object "Bob Jones")',
        )
        # New memories have strength ~1.0, so low threshold should not forget them
        result_low = srv.forget(min_strength=0.01)
        assert "error" not in result_low
        assert result_low["expired_count"] == 0
        # High threshold should forget them
        result = srv.forget(min_strength=2.0)
        assert "error" not in result
        assert result["expired_count"] >= 1

    def test_forget_no_args_returns_error(self, memory_client):
        """forget() with no filters returns error dict."""
        result = srv.forget()
        assert result.get("error") is True
        assert "type" in result
        assert result["type"] == "ValueError"

    def test_remember_with_penman(self, memory_client):
        """remember() accepts PENMAN notation."""
        result = srv.remember(
            penman="(assigned :subject Alice :object task :recipient Bob :memory_type episodic :importance 0.7)",
        )
        assert "error" not in result
        assert result["stored"] == 1

    def test_recall_with_entity_and_role(self, memory_client):
        """recall() filters by entity and karaka role."""
        srv.remember(penman="(assigned :subject Alice :object task)")
        result = srv.recall(entity="Alice", role="subject")
        assert "error" not in result
        assert result["count"] >= 1
        for m in result["memories"]:
            assert m["roles"].get("subject") == "Alice"

    def test_recall_with_action_filter(self, memory_client):
        """recall() filters by action type."""
        srv.remember(penman="(assigned :subject Alice :object task)")
        srv.remember(penman="(reviewed :subject Alice :object code)")
        result = srv.recall(entity="Alice", action="assigned")
        assert "error" not in result
        assert result["count"] >= 1
        assert all(m["action"] == "assigned" for m in result["memories"])

    def test_recall_multi_entity(self, memory_client):
        """recall(entity=[...]) finds memories involving multiple entities."""
        srv.remember(
            penman="(worked :subject Alice :object API :recipient Bob)",
        )
        result = srv.recall(entity=["Alice", "Bob"])
        assert "error" not in result
        assert result["count"] >= 1

    def test_recall_by_mood(self, memory_client):
        """recall(mood=...) filters by mood."""
        srv.remember(
            penman="(deploy :subject Alice :object service :mood planned)",
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

    def test_consolidate_tool(self, memory_client):
        """consolidate() MCP tool works."""
        srv.remember(penman="(met :subject Alice :object Bob)")
        srv.remember(penman="(met :subject Alice :object Bob)")
        result = srv.consolidate()
        assert "error" not in result
        assert "summaries" in result

    def test_consolidate_with_entity(self, memory_client):
        """consolidate(entity=...) scopes to that entity."""
        srv.remember(penman="(met :subject Alice :object Bob)")
        srv.remember(penman="(met_again :subject Alice :object Bob)")
        result = srv.consolidate(entity="Alice")
        assert "error" not in result
        assert "summaries" in result

    def test_recall_with_temporal_filter(self, memory_client):
        """recall with since/before ISO strings."""
        srv.remember(penman="(deploy :subject Alice :object API)")
        result = srv.recall(entity="Alice", since="2020-01-01T00:00:00+00:00")
        assert "error" not in result
        assert result["count"] >= 1
        result = srv.recall(entity="Alice", since="2099-01-01T00:00:00+00:00")
        assert "error" not in result
        assert result["count"] == 0


class TestAgentOutputFormat:
    """Tests for the agent-friendly output format."""

    def test_recall_has_when_and_reliability(self, memory_client):
        """recall returns temporal grounding and reliability label."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        result = srv.recall(entity="Alice")
        assert result["count"] >= 1
        m = result["memories"][0]
        assert "when" in m
        assert m["when"] is not None  # ISO timestamp
        assert m["reliability"] in ("strong", "moderate", "faint")

    def test_recall_omits_default_mood_and_negated(self, memory_client):
        """Default mood='actual' and negated=False are omitted from output."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        result = srv.recall(entity="Alice")
        m = result["memories"][0]
        assert "mood" not in m  # actual is default, omitted
        assert "negated" not in m  # False is default, omitted

    def test_recall_includes_non_default_mood(self, memory_client):
        """Non-default mood is included in output."""
        srv.remember(penman="(deploy :subject Alice :object API :mood planned)")
        result = srv.recall(mood="planned")
        m = result["memories"][0]
        assert m["mood"] == "planned"

    def test_recall_includes_negated_when_true(self, memory_client):
        """negated=True is included in output."""
        srv.remember(penman="(use :subject team :object Java :negated true)")
        result = srv.recall(action="use", negated=True)
        m = result["memories"][0]
        assert m["negated"] is True

    def test_recall_roles_are_role_to_entity(self, memory_client):
        """Roles mapping is {role: entity}, not {entity: role}."""
        srv.remember(penman="(assigned :subject Alice :object task :recipient Bob)")
        result = srv.recall(entity="Alice")
        m = result["memories"][0]
        assert m["roles"]["subject"] == "Alice"
        assert m["roles"]["object"] == "task"
        assert m["roles"]["recipient"] == "Bob"

    def test_recall_memory_type_as_type(self, memory_client):
        """memory_type is returned as 'type' field."""
        srv.remember(penman="(knows :subject Alice :object Python :memory_type semantic)")
        result = srv.recall(entity="Alice")
        m = result["memories"][0]
        assert m["type"] == "semantic"
        assert "memory_type" not in m

    def test_recall_no_internal_ids(self, memory_client):
        """Output has no internal IDs (edge_id, node_ids)."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        result = srv.recall(entity="Alice")
        m = result["memories"][0]
        assert "edge_id" not in m
        assert "node_ids" not in m
        assert "score" not in m
        assert "strength" not in m
        assert "source" not in m
        assert "confidence" not in m

    def test_recall_detects_contradictions(self, memory_client):
        """Opposite negation on same action + entities surfaces as contradiction."""
        srv.remember(penman="(use :subject team :object Python)")
        srv.remember(penman="(use :subject team :object Python :negated true)")
        result = srv.recall(action="use")
        assert "contradictions" in result
        assert len(result["contradictions"]) >= 1
        c = result["contradictions"][0]
        assert "positive" in c
        assert "negative" in c
        assert "shared" in c

    def test_remember_output_format(self, memory_client):
        """remember() returns agent-friendly format with roles and resolved."""
        result = srv.remember(penman="(prefers :subject Alice :object Python :memory_type semantic)")
        assert result["stored"] == 1
        m = result["memories"][0]
        assert m["text"]
        assert m["action"] == "prefers"
        assert m["roles"]["subject"] == "Alice"
        assert m["roles"]["object"] == "Python"
        assert m["type"] == "semantic"
        # No internal IDs
        assert "edge_id" not in m
        assert "entities" not in m

    def test_remember_reports_new_entities(self, memory_client):
        """First-time entities are flagged in 'resolved'."""
        result = srv.remember(penman="(likes :subject NewPerson :object NewThing)")
        m = result["memories"][0]
        assert "resolved" in m
        assert m["resolved"]["NewPerson"] == "new"
        assert m["resolved"]["NewThing"] == "new"

    def test_remember_omits_resolved_when_all_known(self, memory_client):
        """resolved is omitted when all entities are known."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        result = srv.remember(penman="(teaches :subject Alice :object Python)")
        m = result["memories"][0]
        assert "resolved" not in m

    def test_remember_activated_on_related_memories(self, memory_client):
        """Related memories appear as 'activated'."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        result = srv.remember(penman="(teaches :subject Alice :object Python)")
        assert "activated" in result
        assert len(result["activated"]) >= 1
        a = result["activated"][0]
        assert "text" in a
        assert "shared" in a
