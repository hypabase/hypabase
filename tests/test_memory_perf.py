"""Performance regression tests for the memory module.

These tests verify that key operations complete within acceptable time bounds.
They are intentionally generous — failures indicate a real regression, not
tight-tolerance flakiness.
"""

from __future__ import annotations

import time

import pytest

from tests.conftest import MockEmbedder
from hypabase.memory.agent import Memory


class TestMemoryPerformance:
    def test_remember_100_memories(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        start = time.monotonic()
        for i in range(100):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": f"Person{i} Jones", "type": "person", "role": "object"},
                    {"name": f"Location{i} Park", "type": "location", "role": "locus"},
                ],
            )
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"100 remember() calls took {elapsed:.1f}s (limit: 5s)"
        mem.hb.close()

    def test_recall_latency_10(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for i in range(10):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": f"Person{i} Jones", "type": "person", "role": "object"},
                ],
            )
        start = time.monotonic()
        results = mem.recall(entity="Alice Smith")
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"recall over 10 memories took {elapsed:.1f}s (limit: 0.5s)"
        assert len(results) >= 1
        mem.hb.close()

    def test_recall_latency_100(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for i in range(100):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": f"Person{i} Jones", "type": "person", "role": "object"},
                    {"name": f"Location{i} Park", "type": "location", "role": "locus"},
                ],
            )
        start = time.monotonic()
        results = mem.recall(entity="Alice Smith")
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"recall over 100 memories took {elapsed:.1f}s (limit: 2s)"
        assert len(results) >= 1
        mem.hb.close()

    def test_forget_batch_50(self, tmp_db_path):
        mem = Memory(path=tmp_db_path)
        for i in range(50):
            mem.remember(
                action="met",
                entities=[
                    {"name": "Alice Smith", "type": "person", "role": "agent"},
                    {"name": f"Person{i} Jones", "type": "person", "role": "object"},
                ],
            )
        start = time.monotonic()
        count = mem.forget(min_strength=999.0)["expired_count"]
        elapsed = time.monotonic() - start
        assert count >= 50
        assert elapsed < 2.0, f"forgetting 50 edges took {elapsed:.1f}s (limit: 2s)"
        mem.hb.close()
