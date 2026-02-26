"""Product tests for the Memory system -- from the agent's perspective.

Real neural embedder (all-MiniLM-L6-v2 via FastEmbed/ONNX, no mocks).
Server-level API (same interface an agent hits through MCP).  Each test
is a meaningful scenario, not a unit assertion.

The embedder is downloaded once from GCS on first run (~79 MB) and
cached for subsequent runs.  This gives us true semantic similarity:
"machine learning" ↔ "ML algorithms" = high cosine, vs unrelated
text = low cosine.
"""

from __future__ import annotations

import logging
import os
import tarfile
import urllib.request

import pytest

import hypabase.memory.server as srv
from hypabase import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.memory import Memory

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.slow

# ------------------------------------------------------------------
# Real neural embedder via FastEmbed + ONNX
# ------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_GCS_URL = (
    "https://storage.googleapis.com/qdrant-fastembed/"
    "sentence-transformers-all-MiniLM-L6-v2.tar.gz"
)
_CACHE_DIR = os.path.join(os.environ.get("FASTEMBED_CACHE_DIR", "/tmp/fastembed_cache"))
_MODEL_DIR_NAME = "fast-all-MiniLM-L6-v2"


def _ensure_model() -> None:
    """Download the ONNX model from GCS if not already cached."""
    model_dir = os.path.join(_CACHE_DIR, _MODEL_DIR_NAME)
    if os.path.isdir(model_dir) and any(f.endswith(".onnx") for f in os.listdir(model_dir)):
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tar_path = os.path.join(_CACHE_DIR, "model.tar.gz")
    logger.info("Downloading %s (%s) ...", _MODEL_NAME, _MODEL_GCS_URL)
    urllib.request.urlretrieve(_MODEL_GCS_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(path=_CACHE_DIR)
    os.unlink(tar_path)


class _FastEmbedLocalProvider(EmbeddingProvider):
    """FastEmbed backed by a locally-cached ONNX model (no network at embed time)."""

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        _ensure_model()
        self._model = TextEmbedding(
            model_name=_MODEL_NAME,
            cache_dir=_CACHE_DIR,
            local_files_only=True,
        )
        self._dim = 384  # all-MiniLM-L6-v2

    def embed(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [r.tolist() for r in self._model.embed(texts)]

    @property
    def dimension(self) -> int:
        return self._dim


_EMBEDDER: EmbeddingProvider | None = None


def _get_embedder() -> EmbeddingProvider:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _FastEmbedLocalProvider()
    return _EMBEDDER


@pytest.fixture()
def memory(tmp_path):
    """Patch MCP server globals with a real-embedder Memory instance."""
    old_client, old_memory = srv._CLIENT, srv._MEMORY
    embedder = _get_embedder()
    hb = Hypabase(str(tmp_path / "product.db"), embedder=embedder)
    srv._CLIENT = hb
    srv._MEMORY = Memory(hb=hb, embedder=embedder)
    yield
    srv._CLIENT, srv._MEMORY = old_client, old_memory
    hb.close()


# ==================================================================
# 1. Round-trip fidelity
# ==================================================================


class TestRoundTripFidelity:
    """Store a fact, recall it, verify every field comes back correctly."""

    def test_store_and_recall_all_fields(self, memory):
        res = srv.remember(
            penman='(assigned :subject Alice :object "billing task" :recipient Bob '
            ":memory_type episodic :importance 0.8)",
        )
        assert res["stored"] == 1
        m = res["memories"][0]
        assert m["action"] == "assigned"
        assert m["roles"]["subject"] == "Alice"
        assert m["roles"]["object"] == "billing task"
        assert m["roles"]["recipient"] == "Bob"
        assert m["type"] == "episodic"

        # Recall by entity
        result = srv.recall(entity="Alice")
        assert result["count"] >= 1
        hit = result["memories"][0]
        assert hit["action"] == "assigned"
        assert hit["roles"]["subject"] == "Alice"
        assert hit["type"] == "episodic"
        assert hit["when"] is not None
        assert hit["reliability"] in ("strong", "moderate", "faint")

    def test_recall_by_entity_action_and_role(self, memory):
        srv.remember(
            penman='(assigned :subject Alice :object "billing task" :recipient Bob)',
        )
        result = srv.recall(entity="Alice", action="assigned", role="subject")
        assert result["count"] >= 1
        assert all(m["action"] == "assigned" for m in result["memories"])
        assert all(m["roles"].get("subject") == "Alice" for m in result["memories"])

    def test_recall_through_recipient_role(self, memory):
        srv.remember(
            penman='(assigned :subject Alice :object "billing task" :recipient Bob)',
        )
        result = srv.recall(entity="Bob", role="recipient")
        assert result["count"] >= 1
        assert any(m["roles"].get("recipient") == "Bob" for m in result["memories"])


# ==================================================================
# 2. Multiple facts about one entity
# ==================================================================


class TestMultipleFacts:
    """Store several facts about Alice, recall all by entity."""

    def test_multiple_facts_all_recalled(self, memory):
        facts = [
            "(prefers :subject Alice :object Python :memory_type semantic)",
            '(works_at :subject Alice :object "Acme Corp" :memory_type semantic)',
            "(met :subject Alice :object Bob :locus conference :memory_type episodic)",
            "(manages :subject Alice :object backend :memory_type semantic)",
        ]
        for f in facts:
            srv.remember(penman=f)

        result = srv.recall(entity="Alice", limit=10)
        assert result["count"] >= 4
        actions = {m["action"] for m in result["memories"]}
        assert "prefers" in actions
        assert "works_at" in actions
        assert "met" in actions
        assert "manages" in actions


# ==================================================================
# 3. Semantic recall -- real embeddings at work
# ==================================================================


class TestSemanticRecall:
    """With a real neural embedder, test true semantic similarity in recall.

    The recall path (agent.py:334-341) expands anchor nodes via
    hb.search(name, kind="node", min_score=0.7) when an embedder is
    available.  This means recalling a *synonym* or *abbreviation* can
    find memories stored under a different but semantically close name.
    """

    def test_entity_isolation(self, memory):
        """Recalling Alice finds Alice's fact, not Bob's."""
        srv.remember(penman="(prefers :subject Alice :object Python)")
        srv.remember(penman="(prefers :subject Bob :object JavaScript)")

        alice_result = srv.recall(entity="Alice")
        assert alice_result["count"] >= 1
        assert all(
            m["roles"].get("subject") == "Alice" for m in alice_result["memories"]
        )

        bob_result = srv.recall(entity="Bob")
        assert bob_result["count"] >= 1
        assert all(
            m["roles"].get("subject") == "Bob" for m in bob_result["memories"]
        )

    def test_embedded_text_is_natural_language(self, memory):
        """The text field should be a readable sentence, not raw PENMAN."""
        res = srv.remember(
            penman='(deployed :subject Alice :object API :locus "Monday morning")',
        )
        text = res["memories"][0]["text"]
        # Should read like English, not like "(deployed :subject Alice ...)"
        assert "Alice" in text
        assert "deployed" in text
        assert "(" not in text

    def test_semantic_node_search_finds_synonym(self, memory):
        """Recall with a synonym finds memories stored under a different name.

        Store a fact about "artificial intelligence", then recall with "AI".
        The neural embedder produces cosine ~0.79 between these node
        embeddings (above the 0.7 threshold), expanding the anchor set
        to include the original node.
        """
        srv.remember(
            penman='(studies :subject Alice :object "artificial intelligence" :memory_type semantic)'
        )
        # Recall with the abbreviation — no entity named "AI" was ever stored
        result = srv.recall(entity="AI")
        # The neural embedder should bridge "AI" → "artificial intelligence"
        assert result["count"] >= 1, (
            "Semantic node search should find 'artificial intelligence' when queried with 'AI'"
        )

    def test_semantic_search_does_not_match_unrelated(self, memory):
        """Unrelated query should NOT match via semantic search."""
        srv.remember(
            penman='(studies :subject Alice :object "machine learning" :memory_type semantic)'
        )
        # "Italian cooking" should not semantically match "machine learning"
        result = srv.recall(entity="Italian cooking")
        assert result["count"] == 0

    def test_semantic_search_near_miss(self, memory):
        """Conceptually related terms can find stored memories."""
        srv.remember(
            penman='(uses :subject team :object "deep learning" :memory_type semantic)'
        )
        # "neural networks" is closely related to "deep learning"
        result = srv.recall(entity="neural networks")
        # This might or might not match (depends on cosine > 0.7 threshold)
        # But it should definitely not crash
        assert "error" not in result


# ==================================================================
# 4. Entity consolidation with real cosine similarity
# ==================================================================


class TestEntityConsolidation:
    """Real embeddings let consolidate() find semantically similar nodes.

    Phase 1 of consolidate() computes pairwise cosine between all node
    embeddings and merges at >= 0.95.  With a neural model, genuinely
    duplicate or near-duplicate entity names can be merged.
    """

    def test_consolidation_runs_cleanly(self, memory):
        srv.remember(penman="(likes :subject Alice :object tea)")
        srv.remember(penman="(likes :subject Bob :object coffee)")
        srv.remember(penman="(likes :subject Carol :object juice)")

        result = srv.consolidate()
        assert "error" not in result
        assert "summaries" in result

    def test_consolidation_with_entity_scope(self, memory):
        srv.remember(penman="(likes :subject Alice :object tea)")
        srv.remember(penman="(likes :subject Alice :object tea)")  # duplicate

        result = srv.consolidate(entity="Alice")
        assert "error" not in result
        assert "summaries" in result

    def test_consolidation_does_not_merge_unrelated(self, memory):
        """Consolidate should NOT merge semantically different entities."""
        srv.remember(penman="(likes :subject Alice :object Python)")
        srv.remember(penman="(likes :subject Bob :object JavaScript)")
        srv.remember(penman="(uses :subject Carol :object Kubernetes)")

        result = srv.consolidate()
        # These are distinct entities, no merges should happen
        # (edge grouping may still produce summaries for duplicate edge patterns)
        node_merges = [s for s in result["summaries"] if s.get("action") == "merged_nodes"]
        # Alice, Bob, Carol, Python, JavaScript, Kubernetes are all distinct
        # Any merge would be a false positive
        for merge in node_merges:
            merged_set = set(merge.get("members", []))
            assert not ({"Alice", "Bob"} <= merged_set), "Should not merge Alice and Bob"
            assert not ({"Python", "Kubernetes"} <= merged_set), "Should not merge Python and Kubernetes"


# ==================================================================
# 5. Memory types stored and recalled correctly
# ==================================================================


class TestMemoryTypes:
    """Episodic, semantic, and procedural memories are properly segregated."""

    def test_filter_by_each_type(self, memory):
        srv.remember(
            penman="(met :subject Alice :object Bob :locus Monday :memory_type episodic)"
        )
        srv.remember(
            penman='(is :subject Alice :attribute role :value "tech lead" :memory_type semantic)'
        )
        srv.remember(
            penman='(requires :subject deployment :object "run tests first" :memory_type procedural)'
        )

        epi = srv.recall(entity="Alice", memory_type="episodic")
        assert epi["count"] >= 1
        assert all(m["type"] == "episodic" for m in epi["memories"])

        sem = srv.recall(entity="Alice", memory_type="semantic")
        assert sem["count"] >= 1
        assert all(m["type"] == "semantic" for m in sem["memories"])

        proc = srv.recall(action="requires", memory_type="procedural")
        assert proc["count"] >= 1
        assert all(m["type"] == "procedural" for m in proc["memories"])

    def test_recall_without_type_gets_all(self, memory):
        srv.remember(
            penman="(met :subject Alice :object Bob :memory_type episodic)"
        )
        srv.remember(
            penman="(knows :subject Alice :object Python :memory_type semantic)"
        )

        all_result = srv.recall(entity="Alice")
        assert all_result["count"] >= 2


# ==================================================================
# 6. Mood as a query dimension
# ==================================================================


class TestMoodFiltering:
    """Plans, facts, uncertainties, and recommendations are separate."""

    def test_four_moods_segregated(self, memory):
        srv.remember(
            penman="(deployed :subject team :object API :mood actual)"
        )
        srv.remember(
            penman='(migrate :subject team :object API :recipient "new cloud" :mood planned)'
        )
        srv.remember(
            penman='(approve :subject board :object budget :mood uncertain)'
        )
        srv.remember(
            penman="(use :subject team :object Kubernetes :mood normative)"
        )

        for mood_val in ("actual", "planned", "uncertain", "normative"):
            result = srv.recall(action={
                "actual": "deployed",
                "planned": "migrate",
                "uncertain": "approve",
                "normative": "use",
            }[mood_val], mood=mood_val)
            assert result["count"] >= 1, f"No results for mood={mood_val}"
            for m in result["memories"]:
                if mood_val != "actual":
                    assert m["mood"] == mood_val


# ==================================================================
# 7. Negation separates what IS from what ISN'T
# ==================================================================


class TestNegation:
    """Negated and positive assertions don't mix."""

    def test_negation_filter(self, memory):
        srv.remember(penman="(uses :subject team :object Python)")
        srv.remember(penman="(uses :subject team :object Java :negated true)")

        negated = srv.recall(action="uses", negated=True)
        assert negated["count"] >= 1
        assert all(m.get("negated") is True for m in negated["memories"])
        # The negated one should mention Java
        assert any("Java" in m["text"] for m in negated["memories"])

        positive = srv.recall(action="uses", negated=False)
        assert positive["count"] >= 1
        # Positive should not include the negated one
        assert all(not m.get("negated") for m in positive["memories"])

    def test_unfiltered_recall_gets_both(self, memory):
        srv.remember(penman="(uses :subject team :object Python)")
        srv.remember(penman="(uses :subject team :object Java :negated true)")

        both = srv.recall(entity="team")
        assert both["count"] >= 2


# ==================================================================
# 8. Nested PENMAN -- beliefs and compound facts
# ==================================================================


class TestNestedPenman:
    """Nested atoms store both the outer and inner facts."""

    def test_believes_with_nested_is(self, memory):
        res = srv.remember(
            penman="(believes :subject Alice :object (is :subject deadline :value Friday))"
        )
        # Outer atom + inner atom = 2 stored
        # Note: remember reports top-level stored count as 1 (the outer atom),
        # but the inner atom is stored as a separate edge recursively.
        assert res["stored"] >= 1

        # Can recall through Alice (the believer)
        alice_result = srv.recall(entity="Alice")
        assert alice_result["count"] >= 1
        assert any(m["action"] == "believes" for m in alice_result["memories"])

        # Can recall through deadline (the inner fact)
        deadline_result = srv.recall(entity="deadline")
        assert deadline_result["count"] >= 1


# ==================================================================
# 9. Context slots -- cause, purpose, condition
# ==================================================================


class TestContextSlots:
    """Cause, purpose, and condition produce richer embedding text."""

    def test_cause_and_purpose_in_text(self, memory):
        res = srv.remember(
            penman='(deployed :subject Alice :object API '
            ':cause (crashed :subject "old system" :object users) '
            ':purpose "improve reliability")'
        )
        text = res["memories"][0]["text"]
        # The sentence generator adds "because" for :cause and "in order to" for :purpose
        assert "because" in text.lower() or "crashed" in text.lower()
        assert "in order to" in text.lower() or "improve reliability" in text.lower()

    def test_recall_context_memories(self, memory):
        srv.remember(
            penman='(deployed :subject Alice :object API '
            ':cause (crashed :subject "old system" :object users) '
            ':purpose "improve reliability")'
        )
        # Should be findable through Alice
        result = srv.recall(entity="Alice")
        assert result["count"] >= 1

        # The sub-atom (crashed :subject "old system" :object users) is also an edge
        old_system = srv.recall(entity="old system")
        assert old_system["count"] >= 1


# ==================================================================
# 10. Associative activation on remember
# ==================================================================


class TestAssociativeActivation:
    """New memories trigger activation of related older memories."""

    def test_activated_surfaces_related_memories(self, memory):
        # Build up a context
        srv.remember(penman="(likes :subject Alice :object Python)")
        srv.remember(penman="(works_on :subject Alice :object backend)")
        srv.remember(penman="(uses :subject backend :object Python)")

        # New memory sharing entities with earlier ones
        result = srv.remember(penman="(teaches :subject Alice :object Python)")
        # Should activate memories that share "Alice" and/or "Python"
        assert "activated" in result
        assert len(result["activated"]) >= 1
        # Activated memories should have shared entities
        for a in result["activated"]:
            assert "text" in a
            assert "shared" in a


# ==================================================================
# 11. Multi-entity recall -- path finding through the graph
# ==================================================================


class TestMultiEntityRecall:
    """Multi-entity queries find paths through shared intermediate nodes."""

    def test_path_through_shared_entity(self, memory):
        srv.remember(
            penman='(assigned :subject Alice :object "auth module" :recipient Bob)'
        )
        srv.remember(
            penman='(completed :subject Bob :object "auth module")'
        )
        srv.remember(
            penman='(deployed :subject Bob :object "auth module" :locus production)'
        )

        # Alice and production are connected through "auth module" and "Bob"
        result = srv.recall(entity=["Alice", "production"])
        assert result["count"] >= 1
        # We should find at least some of the chain
        actions = {m["action"] for m in result["memories"]}
        assert len(actions) >= 1  # found something connecting them


# ==================================================================
# 12. Confidence and importance affect reliability
# ==================================================================


class TestReliabilityLabels:
    """Low confidence/importance → faint; high → strong."""

    def test_strong_reliability(self, memory):
        srv.remember(
            penman="(knows :subject Alice :object Python :importance 1.0)",
            confidence=1.0,
        )
        result = srv.recall(entity="Alice")
        assert result["count"] >= 1
        assert result["memories"][0]["reliability"] == "strong"

    def test_low_confidence_lower_reliability(self, memory):
        srv.remember(
            penman="(rumored :subject Bob :object departure :importance 0.3)",
            confidence=0.3,
        )
        result = srv.recall(entity="Bob")
        assert result["count"] >= 1
        # Low confidence + low importance → should NOT be "strong"
        assert result["memories"][0]["reliability"] in ("moderate", "faint")


# ==================================================================
# 13. Forget lifecycle
# ==================================================================


class TestForgetLifecycle:
    """Forget expires memories; new ones still work after."""

    def test_forget_and_verify_gone(self, memory):
        srv.remember(penman="(likes :subject Alice :object tea)")
        srv.remember(penman="(likes :subject Bob :object coffee)")

        # Fresh memories have strength ~0.5-1.0, so min_strength=2.0 expires all
        result = srv.forget(min_strength=2.0)
        assert result["expired_count"] >= 2

        # They should be gone from recall
        assert srv.recall(entity="Alice")["count"] == 0
        assert srv.recall(entity="Bob")["count"] == 0

    def test_new_memories_work_after_forget(self, memory):
        srv.remember(penman="(likes :subject Alice :object tea)")
        srv.forget(min_strength=2.0)

        # Store a new memory -- system should still work
        srv.remember(penman="(likes :subject Carol :object juice)")
        result = srv.recall(entity="Carol")
        assert result["count"] >= 1


# ==================================================================
# 14. Full agent workflow -- code review scenario
# ==================================================================


class TestCodeReviewWorkflow:
    """Simulate an agent helping with code review across multiple turns."""

    def test_full_code_review_lifecycle(self, memory):
        # Step 1: Agent learns codebase structure
        srv.remember(
            penman='(owns :subject "backend team" :object "auth service" :memory_type semantic)'
        )
        srv.remember(
            penman='(owns :subject "frontend team" :object dashboard :memory_type semantic)'
        )
        srv.remember(
            penman='(manages :subject Alice :object "backend team" :memory_type semantic)'
        )

        # Step 2: Agent learns about a PR
        srv.remember(
            penman='(submitted :subject Bob :object "PR #42" :memory_type episodic)'
        )
        srv.remember(
            penman='(changes :subject "PR #42" :object "auth service" :memory_type episodic)'
        )
        srv.remember(
            penman='(adds :subject "PR #42" :object "OAuth support" :instrument "auth service")'
        )

        # Step 3: Agent needs to answer "who should review PR #42?"
        # First, what does PR #42 touch?
        pr_result = srv.recall(entity="PR #42")
        assert pr_result["count"] >= 1
        # Should find that PR #42 changes "auth service"
        pr_texts = " ".join(m["text"] for m in pr_result["memories"])
        assert "auth service" in pr_texts.lower() or any(
            "auth service" in str(m["roles"].values()) for m in pr_result["memories"]
        )

        # Who owns auth service?
        auth_result = srv.recall(entity="auth service")
        assert auth_result["count"] >= 1
        # Should find backend team owns it
        auth_texts = " ".join(m["text"] for m in auth_result["memories"])
        assert "backend team" in auth_texts.lower() or any(
            "backend team" in str(m["roles"].values()) for m in auth_result["memories"]
        )

        # Who manages backend team?
        team_result = srv.recall(entity="backend team")
        assert team_result["count"] >= 1

        # Step 4: Agent learns the review outcome
        srv.remember(
            penman='(approved :subject Alice :object "PR #42" :memory_type episodic)'
        )
        srv.remember(
            penman='(merged :subject Bob :object "PR #42" :locus "main branch" :memory_type episodic)'
        )

        # Step 5: Later, "what happened with OAuth?"
        oauth_result = srv.recall(entity="OAuth support")
        assert oauth_result["count"] >= 1

        # Full PR lifecycle is visible
        full_pr = srv.recall(entity="PR #42", limit=10)
        assert full_pr["count"] >= 3  # submitted + changes/adds + approved + merged
        pr_actions = {m["action"] for m in full_pr["memories"]}
        assert "submitted" in pr_actions
        assert "approved" in pr_actions or "merged" in pr_actions


# ==================================================================
# 15. Emerging relationships -- implicit graph connections
# ==================================================================


class TestEmergingRelationships:
    """Facts stored separately become connected through shared entities."""

    def test_rust_connects_alice_and_new_project(self, memory):
        # Session 1: Alice and Rust
        srv.remember(
            penman="(prefers :subject Alice :object Rust :memory_type semantic)"
        )
        srv.remember(
            penman='(implemented :subject Alice :object "CLI tool" :instrument Rust)'
        )

        # Session 2: Rust and the new project (separate topic)
        srv.remember(
            penman='(has :subject Rust :attribute "memory safety" :value excellent :memory_type semantic)'
        )
        srv.remember(
            penman='(requires :subject "new project" :object "memory safety" :memory_type semantic)'
        )

        # Recall Rust → finds BOTH Alice's preference AND the language property
        rust_result = srv.recall(entity="Rust")
        assert rust_result["count"] >= 2
        actions = {m["action"] for m in rust_result["memories"]}
        assert "prefers" in actions or "implemented" in actions  # Alice's connection
        assert "has" in actions  # Rust's property

    def test_shared_entities_create_implicit_links(self, memory):
        # Two unrelated facts share "Python"
        srv.remember(
            penman="(teaches :subject Alice :object Python :memory_type semantic)"
        )
        srv.remember(
            penman="(uses :subject backend :object Python :memory_type semantic)"
        )

        # Recall Python → finds both
        result = srv.recall(entity="Python")
        assert result["count"] >= 2


# ==================================================================
# 16. Multiple atoms in one remember call
# ==================================================================


class TestMultipleAtoms:
    """One remember() call can store multiple atoms."""

    def test_two_atoms_one_call(self, memory):
        res = srv.remember(
            penman="(likes :subject Alice :object tea) (likes :subject Bob :object coffee)"
        )
        assert res["stored"] == 2

        alice = srv.recall(entity="Alice")
        assert alice["count"] >= 1
        assert any("tea" in m["text"] for m in alice["memories"])

        bob = srv.recall(entity="Bob")
        assert bob["count"] >= 1
        assert any("coffee" in m["text"] for m in bob["memories"])
