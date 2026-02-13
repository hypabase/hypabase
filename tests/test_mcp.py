"""Tests for the Hypabase MCP server tools and resources."""

from __future__ import annotations

import pytest

from hypabase import Hypabase
from hypabase.mcp import server as mcp_server
from hypabase.mcp.server import (
    create_edge,
    create_node,
    delete_edge,
    delete_node,
    find_paths,
    get_edge,
    get_neighbors,
    get_node,
    get_stats,
    lookup_edges_by_nodes,
    mcp,
    schema_resource,
    search_edges,
    search_nodes,
    stats_resource,
    upsert_edge,
)


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    """Patch the module-level _CLIENT with a fresh in-memory Hypabase for each test."""
    client = Hypabase()
    monkeypatch.setattr(mcp_server, "_CLIENT", client)
    yield
    client.close()


class TestNodeTools:
    def test_create_node(self):
        result = create_node(id="alice", type="person")
        assert result["id"] == "alice"
        assert result["type"] == "person"
        assert result["properties"] == {}

    def test_create_node_with_properties(self):
        result = create_node(id="alice", type="person", properties={"age": 30, "role": "admin"})
        assert result["properties"]["age"] == 30
        assert result["properties"]["role"] == "admin"

    def test_get_node(self):
        create_node(id="alice", type="person")
        result = get_node(id="alice")
        assert result["id"] == "alice"
        assert result["type"] == "person"

    def test_get_node_not_found(self):
        result = get_node(id="nonexistent")
        assert result["found"] is False
        assert result["id"] == "nonexistent"

    def test_search_nodes_by_type(self):
        create_node(id="alice", type="person")
        create_node(id="bob", type="person")
        create_node(id="aspirin", type="medication")
        result = search_nodes(type="person")
        assert result["count"] == 2
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"alice", "bob"}

    def test_search_nodes_by_properties(self):
        create_node(id="alice", type="person", properties={"role": "admin"})
        create_node(id="bob", type="person", properties={"role": "user"})
        create_node(id="carol", type="person", properties={"role": "admin"})
        result = search_nodes(properties={"role": "admin"})
        assert result["count"] == 2
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"alice", "carol"}

    def test_search_nodes_by_type_and_properties(self):
        create_node(id="alice", type="person", properties={"role": "admin"})
        create_node(id="bot1", type="bot", properties={"role": "admin"})
        result = search_nodes(type="person", properties={"role": "admin"})
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "alice"

    def test_delete_node(self):
        create_node(id="alice", type="person")
        create_edge(nodes=["alice", "bob"], type="knows")
        result = delete_node(id="alice")
        assert result["deleted"] is True
        assert result["edges_removed"] == 1
        assert get_node(id="alice")["found"] is False

    def test_delete_node_nonexistent(self):
        result = delete_node(id="nobody")
        assert result["deleted"] is False
        assert result["edges_removed"] == 0


class TestEdgeTools:
    def test_create_edge(self):
        result = create_edge(nodes=["alice", "bob"], type="knows")
        assert result["type"] == "knows"
        assert result["node_ids"] == ["alice", "bob"]
        assert result["source"] == "mcp"
        assert result["confidence"] == 1.0

    def test_create_edge_with_provenance(self):
        result = create_edge(
            nodes=["a", "b", "c"],
            type="treatment",
            source="clinical_records",
            confidence=0.85,
        )
        assert result["source"] == "clinical_records"
        assert result["confidence"] == 0.85

    def test_create_edge_with_properties(self):
        result = create_edge(
            nodes=["a", "b"],
            type="link",
            properties={"weight": 5},
        )
        assert result["properties"]["weight"] == 5

    def test_get_edge(self):
        created = create_edge(nodes=["a", "b"], type="link")
        result = get_edge(id=created["id"])
        assert result["id"] == created["id"]
        assert result["type"] == "link"

    def test_get_edge_not_found(self):
        result = get_edge(id="nonexistent")
        assert result["found"] is False

    def test_search_edges_by_type(self):
        create_edge(nodes=["a", "b"], type="knows")
        create_edge(nodes=["c", "d"], type="knows")
        create_edge(nodes=["e", "f"], type="works_with")
        result = search_edges(type="knows")
        assert result["count"] == 2

    def test_search_edges_containing(self):
        create_edge(nodes=["alice", "bob"], type="knows")
        create_edge(nodes=["alice", "carol"], type="knows")
        create_edge(nodes=["bob", "carol"], type="knows")
        result = search_edges(containing=["alice"])
        assert result["count"] == 2

    def test_search_edges_by_source(self):
        create_edge(nodes=["a", "b"], type="link", source="s1")
        create_edge(nodes=["c", "d"], type="link", source="s2")
        result = search_edges(source="s1")
        assert result["count"] == 1

    def test_search_edges_by_min_confidence(self):
        create_edge(nodes=["a", "b"], type="link", confidence=0.5)
        create_edge(nodes=["c", "d"], type="link", confidence=0.9)
        result = search_edges(min_confidence=0.8)
        assert result["count"] == 1

    def test_upsert_edge_idempotent(self):
        e1 = upsert_edge(nodes=["alice", "bob"], type="knows", source="mcp")
        e2 = upsert_edge(nodes=["bob", "alice"], type="knows", source="mcp", confidence=0.9)
        assert e1["id"] == e2["id"]

    def test_upsert_edge_with_properties(self):
        upsert_edge(nodes=["a", "b"], type="link", properties={"weight": 1})
        result = upsert_edge(nodes=["a", "b"], type="link", properties={"weight": 5})
        assert result["properties"]["weight"] == 5

    def test_lookup_edges_by_nodes(self):
        create_edge(nodes=["alice", "bob", "carol"], type="group")
        create_edge(nodes=["alice", "bob"], type="pair")
        result = lookup_edges_by_nodes(nodes=["carol", "alice", "bob"])
        assert result["count"] == 1
        assert result["edges"][0]["type"] == "group"

    def test_lookup_edges_by_nodes_no_match(self):
        create_edge(nodes=["a", "b"], type="link")
        result = lookup_edges_by_nodes(nodes=["x", "y"])
        assert result["count"] == 0

    def test_delete_edge(self):
        created = create_edge(nodes=["a", "b"], type="link")
        result = delete_edge(id=created["id"])
        assert result["deleted"] is True
        assert get_edge(id=created["id"])["found"] is False

    def test_delete_edge_nonexistent(self):
        result = delete_edge(id="nonexistent")
        assert result["deleted"] is False


class TestTraversalTools:
    def test_get_neighbors(self):
        create_edge(nodes=["alice", "bob", "carol"], type="group")
        create_edge(nodes=["alice", "dave"], type="pair")
        result = get_neighbors(node_id="alice")
        assert result["count"] == 3
        neighbor_ids = {n["id"] for n in result["nodes"]}
        assert neighbor_ids == {"bob", "carol", "dave"}

    def test_get_neighbors_filtered(self):
        create_edge(nodes=["alice", "bob"], type="knows")
        create_edge(nodes=["alice", "carol"], type="works_with")
        result = get_neighbors(node_id="alice", edge_types=["knows"])
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "bob"

    def test_find_paths(self):
        create_edge(nodes=["alice", "bob"], type="knows")
        create_edge(nodes=["bob", "carol"], type="knows")
        result = find_paths(start="alice", end="carol")
        assert result["count"] >= 1
        assert result["paths"][0] == ["alice", "bob", "carol"]

    def test_find_paths_no_path(self):
        create_edge(nodes=["a", "b"], type="link")
        create_edge(nodes=["c", "d"], type="link")
        result = find_paths(start="a", end="d")
        assert result["count"] == 0

    def test_find_paths_max_hops(self):
        create_edge(nodes=["a", "b"], type="link")
        create_edge(nodes=["b", "c"], type="link")
        create_edge(nodes=["c", "d"], type="link")
        result = find_paths(start="a", end="d", max_hops=1)
        assert result["count"] == 0


class TestStatsTools:
    def test_get_stats(self):
        create_node(id="alice", type="person")
        create_node(id="bob", type="person")
        create_edge(
            nodes=["alice", "bob"],
            type="knows",
            source="mcp",
            confidence=0.9,
        )
        result = get_stats()
        assert result["node_count"] == 2
        assert result["edge_count"] == 1
        assert result["nodes_by_type"]["person"] == 2
        assert result["edges_by_type"]["knows"] == 1
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "mcp"
        assert "default" in result["databases"]

    def test_get_stats_empty(self):
        result = get_stats()
        assert result["node_count"] == 0
        assert result["edge_count"] == 0
        assert result["sources"] == []


class TestNamespaces:
    def test_tools_with_database_param(self):
        create_node(id="aspirin", type="drug", database="drugs")
        create_node(id="s1", type="session", database="sessions")

        drugs_nodes = search_nodes(database="drugs")
        sessions_nodes = search_nodes(database="sessions")

        assert drugs_nodes["count"] == 1
        assert drugs_nodes["nodes"][0]["id"] == "aspirin"
        assert sessions_nodes["count"] == 1
        assert sessions_nodes["nodes"][0]["id"] == "s1"

    def test_namespace_isolation_edges(self):
        create_edge(nodes=["a", "b"], type="link", database="ns1")
        create_edge(nodes=["c", "d"], type="link", database="ns2")

        ns1_edges = search_edges(database="ns1")
        ns2_edges = search_edges(database="ns2")
        default_edges = search_edges()

        assert ns1_edges["count"] == 1
        assert ns2_edges["count"] == 1
        assert default_edges["count"] == 0

    def test_stats_show_databases(self):
        create_node(id="a", type="t", database="alpha")
        create_node(id="b", type="t", database="beta")
        result = get_stats()
        assert "alpha" in result["databases"]
        assert "beta" in result["databases"]


class TestResources:
    def test_schema_resource(self):
        text = schema_resource()
        assert "Hypabase Data Model" in text
        assert "Nodes" in text
        assert "Edges" in text
        assert "Provenance" in text
        assert "Namespaces" in text

    def test_stats_resource(self):
        create_node(id="alice", type="person")
        create_edge(nodes=["alice", "bob"], type="knows", source="test_src")
        text = stats_resource()
        assert "Hypabase Statistics" in text
        assert "Nodes: 2" in text  # alice + auto-created bob
        assert "Edges: 1" in text
        assert "test_src" in text

    def test_stats_resource_empty(self):
        text = stats_resource()
        assert "Nodes: 0" in text
        assert "Edges: 0" in text


class TestServerRegistration:
    def test_all_tools_registered(self):
        tool_names = set()
        for tool in mcp._tool_manager.list_tools():
            tool_names.add(tool.name)
        expected = {
            "create_node",
            "get_node",
            "search_nodes",
            "delete_node",
            "create_edge",
            "get_edge",
            "search_edges",
            "upsert_edge",
            "delete_edge",
            "lookup_edges_by_nodes",
            "get_neighbors",
            "find_paths",
            "get_stats",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"

    def test_all_resources_registered(self):
        resource_uris = set()
        # The resource manager stores templates and static resources
        for template in mcp._resource_manager.list_templates():
            resource_uris.add(str(template.uri_template))
        for resource in mcp._resource_manager.list_resources():
            resource_uris.add(str(resource.uri))
        assert "hypabase://schema" in resource_uris, f"schema not found in {resource_uris}"
        assert "hypabase://stats" in resource_uris, f"stats not found in {resource_uris}"

    def test_tool_count(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 14
