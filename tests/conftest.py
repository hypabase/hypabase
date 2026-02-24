"""Shared fixtures for Hypabase tests."""

import math

import pytest

from hypabase import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider


class MockEmbedder(EmbeddingProvider):
    """Deterministic embedder for testing: hashes the text into a 4-dim vector."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        raw = [
            ((h >> 0) & 0xFF) / 255.0,
            ((h >> 8) & 0xFF) / 255.0,
            ((h >> 16) & 0xFF) / 255.0,
            ((h >> 24) & 0xFF) / 255.0,
        ]
        mag = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / mag for x in raw]

    @property
    def dimension(self) -> int:
        return 4


@pytest.fixture()
def mock_embedder():
    """A MockEmbedder instance for tests."""
    return MockEmbedder()


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
