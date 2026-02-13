"""Shared fixtures for Hypabase tests."""

import pytest

from hypabase import Hypabase


@pytest.fixture()
def hb():
    """Fresh in-memory Hypabase instance."""
    return Hypabase()


@pytest.fixture()
def tmp_db_path(tmp_path):
    """Temporary database path with automatic cleanup."""
    return str(tmp_path / "test.db")


@pytest.fixture()
def populated_hb():
    """In-memory Hypabase with a reference medical graph.

    Nodes (5):
        dr_smith (doctor), dr_jones (doctor),
        patient_123 (patient), aspirin (medication),
        headache (condition)

    Edges (4):
        treatment: dr_smith + patient_123 + aspirin + headache
            source="clinical_records", confidence=0.95
        diagnosis: dr_jones + patient_123 + headache
            source="lab_results", confidence=0.88
        prescribes: dr_smith + aspirin
            source="clinical_records", confidence=0.92
        consult: dr_smith + dr_jones + patient_123
            source="hospital_system", confidence=0.75
    """
    hb = Hypabase()
    hb.node("dr_smith", type="doctor")
    hb.node("dr_jones", type="doctor")
    hb.node("patient_123", type="patient")
    hb.node("aspirin", type="medication")
    hb.node("headache", type="condition")

    hb.edge(
        ["dr_smith", "patient_123", "aspirin", "headache"],
        type="treatment",
        source="clinical_records",
        confidence=0.95,
    )
    hb.edge(
        ["dr_jones", "patient_123", "headache"],
        type="diagnosis",
        source="lab_results",
        confidence=0.88,
    )
    hb.edge(
        ["dr_smith", "aspirin"],
        type="prescribes",
        source="clinical_records",
        confidence=0.92,
    )
    hb.edge(
        ["dr_smith", "dr_jones", "patient_123"],
        type="consult",
        source="hospital_system",
        confidence=0.75,
    )
    return hb
