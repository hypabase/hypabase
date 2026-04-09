"""Pluggable retrieval algorithms for LongMemEval benchmark.

Each algorithm conforms to RecallAlgorithm:
    (hb, embedder, question, limit) -> (formatted_text, count, trace_dict)

Usage:
    from algorithms import get_algorithm
    recall_fn = get_algorithm("qbppr")
    text, count, trace = recall_fn(hb, embedder, question, 30)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from hypabase import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider

RecallAlgorithm = Callable[
    [Hypabase, EmbeddingProvider | None, str, int],
    tuple[str, int, dict],
]

_INFRA_TYPES = frozenset({"same_as", "consolidated"})

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, RecallAlgorithm] = {}


def register(name: str):
    def decorator(fn: RecallAlgorithm) -> RecallAlgorithm:
        ALGORITHMS[name] = fn
        return fn

    return decorator


def get_algorithm(name: str) -> RecallAlgorithm:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}. Available: {sorted(ALGORITHMS.keys())}")
    return ALGORITHMS[name]


# ---------------------------------------------------------------------------
# QB-PPR: Query-Biased Personalized PageRank on Hypergraph
#
# Chitra & Raphael (2019) hypergraph random walk with three corrections
# for star-topology personal memory graphs:
#   1. Inverse-degree vertex weights γ_e(v) = (1/deg(v)) / Σ(1/deg(u))
#      Suppresses the "user" hub node, amplifies specific entities.
#   2. Non-linear edge weight amplification ω_q(e) = cosine^β
#      Stretches the ~1.3x cosine ratio to ~3x for meaningful walk bias.
#   3. High teleportation α=0.5, few iterations n=5
#      Keeps mass near query-relevant seeds on small graphs.
#
# Combined with cosine KNN via Reciprocal Rank Fusion (RRF).
# Validated at 87.4% on LongMemEval 500-question benchmark.
# ---------------------------------------------------------------------------


def _query_to_anchors(
    hb: Hypabase,
    query: str,
    min_score: float = 0.65,
) -> set[str]:
    """Derive anchor nodes from query via embedding similarity.

    Per-keyword (limit=2, min_score=0.65) + full-query (limit=8, min_score=0.5).
    """
    anchors: set[str] = set()

    for word in query.split():
        clean = word.strip("?.,!\"'()[]")
        if len(clean) <= 2:
            continue
        try:
            hits = hb.search(clean, kind="node", limit=2, min_score=min_score)
            for h in hits:
                anchors.add(h["ref_id"])
        except Exception:
            pass

    try:
        hits = hb.search(query, kind="node", limit=8, min_score=0.5)
        for h in hits:
            anchors.add(h["ref_id"])
    except Exception:
        pass

    return anchors


def _build_graph(
    hb: Hypabase,
    query: str,
    beta: float = 4.0,
) -> tuple[
    list[tuple[str, set[str], float, dict[str, float]]],  # edges: (id, nodes, ω, γ)
    dict[str, int],  # node degrees
]:
    """Build the weighted hypergraph in a single pass.

    Returns edge list with precomputed query-biased weights ω_q(e) = cosine^β
    and inverse-degree vertex weights γ_e(v) = (1/deg(v)) / Σ(1/deg(u)).
    Also returns node degree map.

    Combines what was previously three separate edge iterations
    (_bias_edge_weights, _compute_node_degrees, first half of _ppr)
    into one pass.
    """
    # Get cosine similarities for all edges via embedding search
    try:
        search_results = hb.search(query, kind="edge", limit=200, min_score=0.0)
    except Exception:
        search_results = []
    cosine_scores: dict[str, float] = {r["ref_id"]: max(r["score"], 0.0) for r in search_results}

    # Two passes: first compute degrees from ALL non-infra edges (including
    # single-node edges), then build edge data for edges with >=2 nodes.
    # Degree must count all edges to match the original _compute_node_degrees.
    degrees: dict[str, int] = {}
    active_edges: list[tuple[str, set[str]]] = []
    for edge in hb.edges(active=True):
        if edge.type in _INFRA_TYPES:
            continue
        enodes = edge.node_set
        for nid in enodes:
            degrees[nid] = degrees.get(nid, 0) + 1
        if len(enodes) >= 2:
            active_edges.append((edge.id, enodes))

    # Build edge data with ω and γ
    # Sort nodes within each edge for deterministic float accumulation
    # (set iteration order varies with PYTHONHASHSEED between processes).
    edge_data: list[tuple[str, set[str], float, dict[str, float]]] = []
    for eid, enodes in active_edges:
        cos = cosine_scores.get(eid, 0.0)
        w = max(cos**beta, 1e-8)

        sorted_nodes = sorted(enodes)
        inv_degs = {nid: 1.0 / max(degrees.get(nid, 1), 1) for nid in sorted_nodes}
        total_inv = sum(inv_degs[nid] for nid in sorted_nodes)
        gamma = {nid: inv_degs[nid] / total_inv for nid in sorted_nodes}

        edge_data.append((eid, enodes, w, gamma))

    return edge_data, degrees


def _ppr(
    edge_data: list[tuple[str, set[str], float, dict[str, float]]],
    seed_nodes: set[str],
    alpha: float = 0.5,
    n_iter: int = 5,
) -> dict[str, float]:
    """Personalized PageRank on hypergraph with inverse-degree vertex weights.

    Chitra & Raphael (2019) random walk with γ_e(v) = (1/deg(v)) / Σ(1/deg(u)).
    The walker preferentially moves to specific (low-degree) nodes.
    On star-topology memory graphs, the "user" hub (degree ~150) gets
    γ ≈ 0.004 while a specific entity (degree 2) gets γ ≈ 0.3.
    """
    # Build adjacency index
    node_edges_idx: dict[str, list[int]] = {}
    all_node_ids: set[str] = set()

    for idx, (_, enodes, w, _) in enumerate(edge_data):
        if w <= 0:
            continue
        all_node_ids.update(enodes)
        for nid in enodes:
            node_edges_idx.setdefault(nid, []).append(idx)

    if not all_node_ids:
        return {}

    # Weighted degree: d_w(v) = Σ_{e∋v} ω_q(e)
    node_wdeg: dict[str, float] = {}
    for _, enodes, w, _ in edge_data:
        for nid in enodes:
            node_wdeg[nid] = node_wdeg.get(nid, 0.0) + w

    # Teleportation vector: uniform over valid seed nodes
    valid_seeds = seed_nodes & all_node_ids
    s_t: dict[str, float] = {}
    if valid_seeds:
        seed_val = 1.0 / len(valid_seeds)
        for nid in valid_seeds:
            s_t[nid] = seed_val
    else:
        uniform_val = 1.0 / len(all_node_ids)
        for nid in all_node_ids:
            s_t[nid] = uniform_val

    pi = dict(s_t)

    for _ in range(n_iter):
        pi_new: dict[str, float] = {}

        for w_id in sorted(pi):
            pi_w = pi[w_id]
            if pi_w <= 0 or w_id not in node_edges_idx:
                continue
            d_w = node_wdeg.get(w_id, 1.0)
            for idx in node_edges_idx[w_id]:
                _, enodes, w, gamma = edge_data[idx]
                base = pi_w * (w / d_w)
                for v_id in sorted(enodes):
                    if v_id != w_id:
                        pi_new[v_id] = pi_new.get(v_id, 0.0) + base * gamma[v_id]

        pi_next: dict[str, float] = {}
        for nid in all_node_ids:
            pi_next[nid] = alpha * s_t.get(nid, 0.0) + (1 - alpha) * pi_new.get(nid, 0.0)

        total = sum(pi_next.values())
        if total > 0:
            pi = {k: v / total for k, v in pi_next.items()}
        else:
            pi = pi_next

    return pi


def _score_hyperedges(
    edge_data: list[tuple[str, set[str], float, dict[str, float]]],
    pi: dict[str, float],
) -> list[tuple[str, float]]:
    """Score edges: score(e) = Σ π(v) × γ_e(v), using precomputed γ."""
    scored: list[tuple[str, float]] = []
    for eid, enodes, _, gamma in edge_data:
        score = sum(pi.get(nid, 0.0) * gamma[nid] for nid in sorted(enodes))
        if score > 0:
            scored.append((eid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _format_edges(
    hb: Hypabase,
    scored_edges: list[tuple[str, float]],
    elapsed: float,
    algorithm_name: str = "qbppr",
) -> tuple[str, int, dict]:
    """Format scored edges for the answering LLM."""
    if not scored_edges:
        return "(No memories found)", 0, {"total_results": 0, "elapsed_s": round(elapsed, 2)}

    parts = []
    n = 0
    for edge_id, _score in scored_edges:
        edge = hb.get_edge(edge_id)
        if edge is None:
            continue
        n += 1
        text = edge.properties.get("text", "")

        line = f"- {text}"

        extras = []
        mt = edge.properties.get("memory_type", "")
        if mt:
            extras.append(mt)
        mood = edge.properties.get("mood", "actual")
        if mood != "actual":
            extras.append(f"mood: {mood}")
        if edge.properties.get("negated"):
            extras.append("NEGATED")
        tense = edge.properties.get("tense")
        if tense:
            extras.append(f"tense: {tense}")
        if edge.source:
            extras.append(f"recorded: {edge.source}")
        if extras:
            line += f" ({', '.join(extras)})"
        parts.append(line)

    trace: dict[str, Any] = {
        "total_results": len(parts),
        "elapsed_s": round(elapsed, 2),
        "algorithm": algorithm_name,
    }
    return "\n".join(parts), len(parts), trace


@register("qbppr")
def recall_qbppr(
    hb: Hypabase,
    embedder: EmbeddingProvider | None,
    question: str,
    limit: int,
) -> tuple[str, int, dict]:
    """QB-PPR + Cosine hybrid via RRF.

    Structural arm: PPR on hypergraph with inverse-degree γ, cosine^4 edge
    weights, α=0.5, 5 iterations.
    Semantic arm: Cosine KNN on edge embeddings.
    Combined via Reciprocal Rank Fusion (k=60).
    """
    t0 = time.monotonic()
    hb = hb.database("memory")

    # Structural arm: build graph + PPR
    seeds = _query_to_anchors(hb, question)
    edge_data, _ = _build_graph(hb, question, beta=4.0)
    pi = _ppr(edge_data, seeds, alpha=0.5, n_iter=5)
    ppr_scored = _score_hyperedges(edge_data, pi)

    # Semantic arm: Cosine KNN
    try:
        cosine_results = hb.search(question, kind="edge", limit=limit * 3, min_score=0.3)
        cosine_scored = [(r["ref_id"], r["score"]) for r in cosine_results if r.get("edge") is not None]
    except Exception:
        cosine_scored = []

    # Reciprocal Rank Fusion
    rrf_k = 60
    ppr_rank = {eid: i + 1 for i, (eid, _) in enumerate(ppr_scored)}
    cos_rank = {eid: i + 1 for i, (eid, _) in enumerate(cosine_scored)}
    n_ppr = len(ppr_scored)
    n_cos = len(cosine_scored)

    all_eids = set(ppr_rank) | set(cos_rank)
    combined: list[tuple[str, float]] = []
    for eid in all_eids:
        pr = ppr_rank.get(eid, n_ppr + 1)
        cr = cos_rank.get(eid, n_cos + 1)
        combined.append((eid, 1.0 / (rrf_k + pr) + 1.0 / (rrf_k + cr)))
    combined.sort(key=lambda x: x[1], reverse=True)

    top_edges = combined[:limit]
    return _format_edges(hb, top_edges, time.monotonic() - t0)
