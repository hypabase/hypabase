"""Verify that README code examples actually run."""

from hypabase import Hypabase


class TestReadmeExamples:
    def test_quick_example(self, tmp_db_path):
        hb = Hypabase(tmp_db_path)

        hb.node("dr_smith", type="doctor")
        hb.node("patient_123", type="patient")
        hb.node("aspirin", type="medication")
        hb.node("headache", type="condition")
        hb.node("mercy_hospital", type="hospital")

        hb.edge(
            ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
            type="treatment",
            source="clinical_records",
            confidence=0.95,
        )

        edges = hb.edges(containing=["patient_123"])
        assert len(edges) == 1

        edges = hb.edges(containing=["patient_123", "aspirin"], match_all=True)
        assert len(edges) == 1

        paths = hb.paths("dr_smith", "mercy_hospital")
        assert len(paths) >= 1
        hb.close()

    def test_provenance_example(self):
        hb = Hypabase()

        hb.edge(
            ["patient_123", "aspirin", "ibuprofen"],
            type="drug_interaction",
            source="clinical_decision_support_v3",
            confidence=0.92,
        )

        with hb.context(source="schema_analysis", confidence=0.9):
            hb.edge(["a", "b"], type="fk")
            hb.edge(["b", "c"], type="fk")

        # Provenance queries from README
        edges = hb.edges(source="clinical_decision_support_v3")
        assert len(edges) == 1
        edges = hb.edges(min_confidence=0.9)
        assert len(edges) == 3

        sources = hb.sources()
        assert len(sources) == 2

    def test_vs_neo4j_example(self):
        hb = Hypabase()

        hb.edge(
            ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"],
            type="treatment",
        )

        results = hb.edges_by_vertex_set(
            ["dr_smith", "patient_123", "aspirin", "headache", "mercy_hospital"]
        )
        assert len(results) == 1
        assert results[0].type == "treatment"
        assert len(results[0].node_ids) == 5
