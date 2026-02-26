"""Product tests for the Memory system -- from the agent's perspective.

Real embedder (no mocks of Memory/Hypabase internals).  Server-level
API (same interface an agent hits through MCP).  Each test is a
meaningful scenario, not a unit assertion.

Uses a bag-of-words embedder that produces semantically meaningful
vectors (similar texts → close vectors).  Falls back from FastEmbed
if the model can't be loaded (e.g. no network).  Either way, the full
product pipeline is exercised: embed_node, embed_edge, cosine search,
consolidation, strength scoring.
"""

from __future__ import annotations

import math
import re

import pytest

import hypabase.memory.server as srv
from hypabase import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.memory import Memory

pytestmark = pytest.mark.slow

# ------------------------------------------------------------------
# Bag-of-words embedder: lightweight, semantic, no network needed
# ------------------------------------------------------------------

# Vocabulary of ~100 common words → each gets a dimension.  Texts
# containing the same words produce similar vectors.  This is a real
# embedder (implements EmbeddingProvider, runs through the full vector
# pipeline) -- it just doesn't need a 400 MB model download.
_VOCAB = (
    "alice bob carol dave team backend frontend project api service "
    "auth oauth module deploy deployed deployment production staging "
    "python rust javascript java react kubernetes docker aws "
    "prefers likes uses manages owns works assigned completed reviewed "
    "approved merged submitted changes adds created built implemented "
    "knows believes decided learned teaches met told asked requested "
    "task code pr review test testing meeting sprint monday tuesday "
    "friday billing dashboard cli tool new old system memory safety "
    "excellent budget reliability tea coffee juice programming language "
    "deadline senior lead engineer role type time complexity sort "
    "has is will should might not never because purpose condition "
    "episodic semantic procedural planned actual uncertain normative "
).split()
_WORD_TO_DIM = {w: i for i, w in enumerate(_VOCAB)}
_DIM = len(_VOCAB)


class BagOfWordsEmbedder(EmbeddingProvider):
    """Produce a normalised bag-of-words vector over a fixed vocabulary.

    Similar texts → high cosine similarity.  Different texts → low.
    This is enough for the product pipeline to exercise real vector
    search, consolidation, and spreading activation.
    """

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        words = re.findall(r"[a-z]+", text.lower())
        for w in words:
            idx = _WORD_TO_DIM.get(w)
            if idx is not None:
                vec[idx] += 1.0
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]

    @property
    def dimension(self) -> int:
        return _DIM


def _make_embedder() -> EmbeddingProvider:
    """Try FastEmbed first; fall back to BagOfWordsEmbedder."""
    try:
        from hypabase.engine.embeddings import FastEmbedProvider

        return FastEmbedProvider()
    except Exception:
        return BagOfWordsEmbedder()


_EMBEDDER: EmbeddingProvider | None = None


def _get_embedder() -> EmbeddingProvider:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _make_embedder()
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
    """With a real embedder, verify graph-based recall works faithfully."""

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


# ==================================================================
# 4. Entity consolidation with real cosine similarity
# ==================================================================


class TestEntityConsolidation:
    """Real embeddings let consolidate() find semantically similar nodes."""

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
