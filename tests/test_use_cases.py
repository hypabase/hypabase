"""Real-world use case integration tests.

Each test class simulates an early-adopter workflow using only the public
Hypabase API. Domain-specific graphs are built in per-class fixtures.
"""

import pytest

from hypabase import Hypabase

# ---------------------------------------------------------------------------
# Medical Knowledge Graph
# ---------------------------------------------------------------------------


class TestMedicalKnowledgeGraph:
    @pytest.fixture()
    def med_hb(self):
        hb = Hypabase()
        hb.node("dr_smith", type="doctor")
        hb.node("dr_jones", type="doctor")
        hb.node("patient_a", type="patient")
        hb.node("patient_b", type="patient")
        hb.node("aspirin", type="medication")
        hb.node("ibuprofen", type="medication")
        hb.node("headache", type="condition")
        hb.node("fever", type="condition")
        hb.node("mercy_hospital", type="hospital")

        with hb.context(source="clinical_records", confidence=0.95):
            hb.edge(
                ["dr_smith", "patient_a", "aspirin", "headache", "mercy_hospital"],
                type="treatment",
            )
            hb.edge(
                ["dr_jones", "patient_b", "ibuprofen", "fever"],
                type="treatment",
            )

        with hb.context(source="lab_results", confidence=0.88):
            hb.edge(
                ["dr_smith", "patient_a", "headache"],
                type="diagnosis",
            )
        return hb

    def test_patient_query(self, med_hb):
        edges = med_hb.edges(containing=["patient_a"])
        assert len(edges) == 2  # treatment + diagnosis

    def test_provenance_filter(self, med_hb):
        high_conf = med_hb.edges(min_confidence=0.9)
        assert len(high_conf) == 2  # both treatments at 0.95
        assert all(e.confidence >= 0.9 for e in high_conf)

    def test_path_between_entities(self, med_hb):
        paths = med_hb.paths("dr_smith", "mercy_hospital")
        assert len(paths) >= 1
        assert paths[0][0] == "dr_smith"
        assert paths[0][-1] == "mercy_hospital"

    def test_nary_preservation(self, med_hb):
        treatments = med_hb.edges(type="treatment")
        five_node = [e for e in treatments if len(e.node_ids) == 5]
        assert len(five_node) == 1
        assert set(five_node[0].node_ids) == {
            "dr_smith",
            "patient_a",
            "aspirin",
            "headache",
            "mercy_hospital",
        }

    def test_sources_overview(self, med_hb):
        sources = med_hb.sources()
        src_map = {s["source"]: s for s in sources}
        assert "clinical_records" in src_map
        assert "lab_results" in src_map
        assert src_map["clinical_records"]["edge_count"] == 2
        assert src_map["lab_results"]["edge_count"] == 1


# ---------------------------------------------------------------------------
# RAG Knowledge Extraction
# ---------------------------------------------------------------------------


class TestRAGKnowledgeExtraction:
    @pytest.fixture()
    def rag_hb(self):
        hb = Hypabase()
        # Simulate 3 "documents" extracting facts
        with hb.context(source="doc_arxiv_2401", confidence=0.92):
            hb.edge(["transformer", "attention", "nlp"], type="concept_link")
            hb.edge(["bert", "transformer", "pretraining"], type="builds_on")

        with hb.context(source="doc_blog_post", confidence=0.75):
            hb.edge(["transformer", "gpu", "training"], type="requires")
            hb.edge(["attention", "memory", "scaling"], type="tradeoff")

        with hb.context(source="doc_textbook_ch5", confidence=0.5):
            hb.edge(["rnn", "lstm", "attention"], type="evolution")
        return hb

    def test_entity_retrieval(self, rag_hb):
        edges = rag_hb.edges(containing=["transformer"])
        assert len(edges) == 3

    def test_filter_by_source(self, rag_hb):
        edges = rag_hb.edges(source="doc_arxiv_2401")
        assert len(edges) == 2

    def test_filter_by_confidence(self, rag_hb):
        high_quality = rag_hb.edges(min_confidence=0.8)
        assert len(high_quality) == 2
        assert all(e.confidence >= 0.8 for e in high_quality)

    def test_multi_hop_path(self, rag_hb):
        paths = rag_hb.paths("bert", "nlp")
        assert len(paths) >= 1

    def test_nary_preservation(self, rag_hb):
        concept_links = rag_hb.edges(type="concept_link")
        assert len(concept_links) == 1
        assert len(concept_links[0].node_ids) == 3


# ---------------------------------------------------------------------------
# Agent Memory (multi-session persistence)
# ---------------------------------------------------------------------------


class TestAgentMemory:
    def test_multi_session_memory(self, tmp_db_path):
        # Session 1: agent records task context
        with Hypabase(tmp_db_path) as hb:
            with hb.context(source="session_1", confidence=0.9):
                hb.node("user_alice", type="user")
                hb.node("task_write_report", type="task")
                hb.node("doc_quarterly", type="document")
                hb.edge(
                    ["user_alice", "task_write_report", "doc_quarterly"],
                    type="assigned",
                )

        # Session 2: reopen, query session 1, add new data
        with Hypabase(tmp_db_path) as hb:
            # Verify session 1 data accessible
            alice_edges = hb.edges(containing=["user_alice"])
            assert len(alice_edges) == 1

            with hb.context(source="session_2", confidence=0.85):
                hb.node("tool_spreadsheet", type="tool")
                hb.edge(
                    ["user_alice", "task_write_report", "tool_spreadsheet"],
                    type="uses_tool",
                )

        # Session 3: verify all data present
        with Hypabase(tmp_db_path) as hb:
            assert len(hb.nodes()) == 4
            assert len(hb.edges()) == 2

            # Cross-session path works
            paths = hb.paths("doc_quarterly", "tool_spreadsheet")
            assert len(paths) >= 1

            # Sources track sessions
            sources = hb.sources()
            src_names = {s["source"] for s in sources}
            assert "session_1" in src_names
            assert "session_2" in src_names


# ---------------------------------------------------------------------------
# Decision Audit Trail
# ---------------------------------------------------------------------------


class TestDecisionAuditTrail:
    @pytest.fixture()
    def audit_hb(self):
        hb = Hypabase()
        hb.node("decision_expand_apac", type="decision")
        hb.node("market_japan", type="market")
        hb.node("market_korea", type="market")
        hb.node("budget_5m", type="budget")
        hb.node("risk_currency", type="risk")
        hb.node("timeline_q3", type="timeline")

        hb.edge(
            ["decision_expand_apac", "market_japan", "market_korea", "budget_5m"],
            type="scope",
            source="board_meeting",
            confidence=0.95,
        )
        hb.edge(
            ["decision_expand_apac", "risk_currency"],
            type="risk_assessment",
            source="legal_review",
            confidence=0.85,
        )
        hb.edge(
            ["decision_expand_apac", "budget_5m", "timeline_q3"],
            type="financial_plan",
            source="financial_analysis",
            confidence=0.92,
        )
        hb.edge(
            ["decision_expand_apac", "market_japan"],
            type="market_entry",
            source="consultant_x",
            confidence=0.6,
        )
        return hb

    def test_query_by_source(self, audit_hb):
        edges = audit_hb.edges(source="board_meeting")
        assert len(edges) == 1
        assert len(edges[0].node_ids) == 4

    def test_filter_by_confidence(self, audit_hb):
        trusted = audit_hb.edges(min_confidence=0.85)
        assert len(trusted) == 3
        sources = {e.source for e in trusted}
        assert "consultant_x" not in sources

    def test_sources_overview(self, audit_hb):
        sources = audit_hb.sources()
        assert len(sources) == 4
        src_map = {s["source"]: s for s in sources}
        assert src_map["board_meeting"]["avg_confidence"] == 0.95
        assert src_map["consultant_x"]["avg_confidence"] == 0.6

    def test_atomic_decision_context(self, audit_hb):
        scope = audit_hb.edges(type="scope")
        assert len(scope) == 1
        assert set(scope[0].node_ids) == {
            "decision_expand_apac",
            "market_japan",
            "market_korea",
            "budget_5m",
        }
