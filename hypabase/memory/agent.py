"""Memory -- opinionated AI agent memory built on Hypabase.

Implements Stewart & Buehler's (2026) higher-order knowledge representation
patterns with penman grammar + karaka roles.

Four operations: remember, recall, consolidate, forget.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from hypabase.client import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.memory.penman import Atom, atom_to_sentence, parse_penman
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength
from hypabase.memory.types import (
    MEMORY_TYPES,
    MOODS,
    PENMAN_TO_KARAKA,
    KarakaRole,
    MemoryType,
    Mood,
)

logger = logging.getLogger(__name__)

# Edge types used for infrastructure (excluded from user-facing queries)
_INFRA_TYPES = frozenset({"same_as", "consolidated"})


class Memory:
    """High-level memory interface for AI agents.

    Builds on Hypabase to provide:
    - ``remember(penman='...')``: store memories as PENMAN atoms (S&B write)
    - ``recall()``: retrieve memories via IS-constrained path finding (S&B retrieve)
    - ``consolidate()``: batch node merging + edge grouping (S&B consolidate)
    - ``forget()``: expire old or weak memories

    Args:
        hb: An existing Hypabase instance to use. If None, one is created.
        path: Path for SQLite database (used only if hb is None).
        database: Namespace to use (default "memory").
        embedder: Embedding provider for semantic search.
    """

    def __init__(
        self,
        hb: Hypabase | None = None,
        *,
        path: str | None = None,
        database: str = "memory",
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        if hb is not None:
            self._hb = hb.database(database)
        else:
            self._hb = Hypabase(path, database=database, embedder=embedder)
        self._embedder = embedder or getattr(self._hb, "_embedder", None)
        self._resolver = EntityResolver(self._hb)
        self._resolver.warm_cache()

    @property
    def hb(self) -> Hypabase:
        """The underlying Hypabase instance."""
        return self._hb

    # ==================================================================
    # remember  (S&B: always write new, verbatim labels)
    # ==================================================================

    def remember(
        self,
        penman: str,
        *,
        source: str = "memory",
        confidence: float = 1.0,
    ) -> dict:
        """Store memories as PENMAN atoms.

        Always writes new edges (S&B approach -- no dedup at write time).
        Entity names use normalized cache only (no embedding search).

        Args:
            penman: One or more PENMAN atoms.
            source: Provenance source identifier.
            confidence: Provenance confidence 0.0-1.0.

        Returns:
            Dict with ``stored`` count, ``edges`` (per-atom reports with
            entity resolution status), and ``related`` memories.
        """
        atoms = parse_penman(penman)
        atom_reports = []
        for atom in atoms:
            report = self._remember_atom(atom, source, confidence)
            atom_reports.append(report)

        # Build activation report
        all_entity_ids: list[str] = []
        for r in atom_reports:
            all_entity_ids.extend(e["node_id"] for e in r["entities"])

        related = self._find_related_memories(
            entity_ids=list(set(all_entity_ids)),
            exclude_edge_ids={r["edge_id"] for r in atom_reports},
            limit=5,
        )

        return {
            "stored": len(atom_reports),
            "edges": atom_reports,
            "related": related,
        }

    def _remember_atom(
        self,
        atom: Atom,
        source: str,
        confidence: float,
    ) -> dict:
        """Store a single Atom as a hyperedge.

        S&B pipeline: normalize name -> write verbatim -> embed.
        """
        incidences: list[dict[str, Any]] = []
        entity_reports: list[dict[str, str]] = []

        all_slots: list[tuple[str, str | Atom]] = [(PENMAN_TO_KARAKA.get(rn, rn), rv) for rn, rv in atom.roles] + [
            (cn, cv) for cn, cv in atom.contexts
        ]

        for role_key, slot_val in all_slots:
            if isinstance(slot_val, Atom):
                nested = self._remember_atom(slot_val, source, confidence)
                incidences.append(
                    {
                        "edge_ref_id": nested["edge_id"],
                        "properties": {"role": role_key},
                    }
                )
            else:
                # S&B: normalized cache only, no embedding search
                known = self._resolver.is_known(slot_val)
                resolved = self._resolver.resolve(slot_val)
                status = "existing" if known else "new"
                node_id = resolved

                # Create node if new (verbatim name)
                if status == "new":
                    self._hb.node(node_id, type="entity")
                    # Register in cache for future exact matches
                    self._resolver.register(node_id, [])
                    # Embed node for future consolidation/search
                    if self._embedder is not None and self._hb.storage is not None:
                        try:
                            self._hb.embed_node(node_id, text=node_id)
                        except Exception:
                            logger.warning("Failed to embed node %s", node_id)

                incidences.append(
                    {
                        "node_id": node_id,
                        "properties": {"role": role_key},
                    }
                )
                entity_reports.append(
                    {
                        "name": slot_val,
                        "node_id": node_id,
                        "status": status,
                        "role": role_key,
                    }
                )

        if len(incidences) < 2:
            raise ValueError(
                f"A memory needs at least 2 participants -- it's a hyperedge. Atom ({atom.verb}) has {len(incidences)}."
            )

        # Create hyperedge (always new -- S&B approach)
        props = self._extract_modifiers(atom)
        embed_text = atom_to_sentence(atom)
        props["text"] = embed_text

        edge = self._hb.edge(
            incidences=incidences,
            type=atom.verb.strip().lower(),
            source=source,
            confidence=confidence,
            properties=props,
        )

        # Embed edge for retrieval
        if self._embedder is not None and self._hb.storage is not None:
            try:
                self._hb.embed_edge(edge.id, embed_text)
            except Exception:
                logger.warning("Failed to embed edge %s", edge.id)

        return {
            "edge_id": edge.id,
            "action": atom.verb.strip().lower(),
            "text": embed_text,
            "entities": entity_reports,
            "memory_type": props.get("memory_type"),
            "mood": props.get("mood"),
        }

    def _extract_modifiers(self, atom: Atom) -> dict[str, Any]:
        """Extract and validate edge properties from atom modifiers."""
        props: dict[str, Any] = {}
        mood = atom.modifiers.get("mood")
        if mood is not None:
            if mood not in MOODS:
                raise ValueError(f"Unknown mood {mood!r}. Must be one of: {', '.join(sorted(MOODS))}.")
            props["mood"] = mood
        memory_type = atom.modifiers.get("memory_type")
        if memory_type is not None:
            if memory_type not in MEMORY_TYPES:
                raise ValueError(
                    f"Unknown memory_type {memory_type!r}. Must be one of: {', '.join(sorted(MEMORY_TYPES))}."
                )
            props["memory_type"] = memory_type
        importance = atom.modifiers.get("importance")
        if importance is not None:
            props["importance"] = importance
        if atom.modifiers.get("negated"):
            props["negated"] = True
        tense = atom.modifiers.get("tense")
        if tense is not None:
            props["tense"] = tense
        return props

    def _find_related_memories(
        self,
        entity_ids: list[str],
        exclude_edge_ids: set[str],
        limit: int = 5,
    ) -> list[dict]:
        """Find existing memories that share entities with newly stored ones."""
        seen: set[str] = set()
        related: list[dict] = []
        for nid in entity_ids:
            for edge in self._hb.edges_of_node(nid):
                if (
                    edge.id not in exclude_edge_ids
                    and edge.id not in seen
                    and edge.is_active
                    and edge.type not in _INFRA_TYPES
                ):
                    seen.add(edge.id)
                    shared = set(edge.node_ids) & set(entity_ids)
                    related.append(
                        {
                            "edge_id": edge.id,
                            "text": edge.properties.get("text", ""),
                            "action": edge.type,
                            "shared_entities": sorted(shared),
                        }
                    )
        related.sort(key=lambda x: len(x["shared_entities"]), reverse=True)
        return related[:limit]

    # ==================================================================
    # recall  (S&B: embed query -> match nodes -> IS-constrained paths)
    # ==================================================================

    def recall(
        self,
        *,
        entity: str | list[str] | None = None,
        action: str | None = None,
        role: KarakaRole | None = None,
        memory_type: MemoryType | None = None,
        mood: Mood | None = None,
        negated: bool | None = None,
        since: datetime | None = None,
        before: datetime | None = None,
        limit: int = 10,
        min_strength: float = 0.0,
    ) -> list[dict]:
        """Recall memories using S&B's IS-constrained path finding.

        Pipeline:
        1. **Match**: Resolve entity names to anchor nodes (cache + same_as + search)
        2. **Find**: IS-constrained paths (2+ anchors) or neighborhood (1 anchor)
        3. **Filter + Rank**: Grammar filters, then score by activation x strength

        Args:
            entity: Entity name(s) for lookup. String or list of strings.
            action: Filter to edges of this action type (verb).
            role: Filter to edges where an entity has this karaka role.
            memory_type: Filter by memory type (episodic/semantic/procedural).
            mood: Filter by mood.
            negated: Filter -- True=only negated, False=only positive, None=all.
            since: Only include edges created after this datetime.
            before: Only include edges created before this datetime.
            limit: Maximum results.
            min_strength: Minimum memory strength threshold.

        Returns:
            List of dicts with 'edge', 'score', 'strength', 'text',
            'action', 'memory_type', 'mood', 'negated', and 'roles'.
        """
        if entity is None and action is None and memory_type is None and mood is None:
            raise ValueError("At least one of entity, action, memory_type, or mood must be provided.")
        if role is not None and entity is None:
            raise ValueError(
                "role filter requires entity to be specified. "
                "Use recall(entity=..., role=...) to find where an entity has a specific role."
            )

        entity_list: list[str] = []
        if entity is not None:
            entity_list = [entity] if isinstance(entity, str) else list(entity)

        # ---- Step 1: MATCH -- resolve to anchor nodes ----
        anchor_nodes: set[str] = set()
        if entity_list:
            for name in entity_list:
                # Resolve via cache
                resolved = self._resolver.resolve(name)
                anchor_nodes.add(resolved)
                # Expand via same_as aliases from graph
                for alias in self._get_same_as_aliases(resolved):
                    anchor_nodes.add(alias)
                # If embedder available, also try semantic node search
                if self._embedder is not None and self._hb.storage is not None:
                    try:
                        similar = self._hb.search(name, kind="node", limit=3, min_score=0.7)
                        for s in similar:
                            anchor_nodes.add(s["ref_id"])
                    except Exception:
                        logger.debug("Semantic node search failed for %r", name)

        # ---- Step 2: FIND -- collect candidate edges ----
        candidates: list[dict] = []
        seen_edge_ids: set[str] = set()

        if len(anchor_nodes) >= 2 and len(entity_list) >= 2:
            # S&B: IS-constrained path finding between anchor groups
            # Split anchors into start/end based on original entity resolution
            resolved_list = [self._resolver.resolve(name) for name in entity_list]
            start_set = set()
            end_set = set()
            # First entity -> start, rest -> end (plus their aliases)
            first_resolved = resolved_list[0]
            start_set.add(first_resolved)
            for alias in self._get_same_as_aliases(first_resolved):
                start_set.add(alias)
            for r in resolved_list[1:]:
                end_set.add(r)
                for alias in self._get_same_as_aliases(r):
                    end_set.add(alias)

            paths_found = False
            try:
                paths = self._hb.find_paths(
                    start_nodes=start_set,
                    end_nodes=end_set,
                    min_intersection=1,
                    max_hops=3,
                    max_paths=20,
                )
                for path in paths:
                    paths_found = True
                    for edge in path:
                        if edge.id not in seen_edge_ids and edge.is_active and edge.type not in _INFRA_TYPES:
                            seen_edge_ids.add(edge.id)
                            # Score by overlap with anchors
                            overlap = len(anchor_nodes & edge.node_set)
                            score = overlap / len(anchor_nodes) if anchor_nodes else 0.5
                            candidates.append(
                                {
                                    "edge": edge,
                                    "score": max(score, 0.3),
                                    "text": edge.properties.get("text", ""),
                                }
                            )
            except Exception:
                logger.debug("find_paths failed, falling back to neighborhood")
                paths_found = False

            # If no paths found, fall back to edges containing ANY anchor
            if not paths_found:
                for nid in anchor_nodes:
                    for edge in self._hb.edges_of_node(nid):
                        if edge.id not in seen_edge_ids and edge.is_active and edge.type not in _INFRA_TYPES:
                            seen_edge_ids.add(edge.id)
                            overlap = len(anchor_nodes & edge.node_set)
                            score = overlap / len(anchor_nodes) if anchor_nodes else 0.5
                            candidates.append(
                                {
                                    "edge": edge,
                                    "score": max(score, 0.1),
                                    "text": edge.properties.get("text", ""),
                                }
                            )

        elif anchor_nodes:
            # 1 anchor: neighborhood lookup
            for nid in anchor_nodes:
                for edge in self._hb.edges_of_node(nid):
                    if edge.id not in seen_edge_ids and edge.is_active and edge.type not in _INFRA_TYPES:
                        seen_edge_ids.add(edge.id)
                        overlap = len(anchor_nodes & edge.node_set)
                        score = overlap / len(anchor_nodes) if anchor_nodes else 0.5
                        candidates.append(
                            {
                                "edge": edge,
                                "score": max(score, 0.3),
                                "text": edge.properties.get("text", ""),
                            }
                        )
        else:
            # No entity provided -- scan all active edges (filter-only mode)
            for edge in self._hb.edges(active=True):
                if edge.type not in _INFRA_TYPES:
                    candidates.append(
                        {
                            "edge": edge,
                            "score": 0.5,
                            "text": edge.properties.get("text", ""),
                        }
                    )

        if not candidates:
            return []

        # ---- Step 3: FILTER + RANK ----
        if action is not None:
            candidates = [c for c in candidates if c["edge"].type == action]
        if memory_type is not None:
            candidates = [c for c in candidates if c["edge"].properties.get("memory_type") == memory_type]
        if role is not None:
            if anchor_nodes:
                candidates = [c for c in candidates if self._edge_has_role_for_seeds(c["edge"], role, anchor_nodes)]
            else:
                candidates = [c for c in candidates if self._edge_has_role(c["edge"], role)]
        if mood is not None:
            candidates = [c for c in candidates if c["edge"].properties.get("mood", "actual") == mood]
        if negated is not None:
            candidates = [c for c in candidates if c["edge"].properties.get("negated", False) == negated]
        if since is not None:
            since_ts = since.timestamp()
            candidates = [
                c
                for c in candidates
                if c["edge"].created_at is not None and c["edge"].created_at.timestamp() >= since_ts
            ]
        if before is not None:
            before_ts = before.timestamp()
            candidates = [
                c
                for c in candidates
                if c["edge"].created_at is not None and c["edge"].created_at.timestamp() <= before_ts
            ]

        if not candidates:
            return []

        # Record access
        all_edge_ids = [c["edge"].id for c in candidates]
        if self._hb.storage is not None:
            self._hb.storage.record_access_batch(self._hb.current_namespace, "edge", all_edge_ids)

        # Batch fetch access stats
        batch_stats = self._get_batch_access_stats("edge", all_edge_ids)

        # Compute final score and filter by strength
        results: list[dict] = []
        for c in candidates:
            edge = c["edge"]
            access = batch_stats.get(edge.id, {"access_count": 0, "last_accessed": 0.0})
            edge_memory_type = edge.properties.get("memory_type")
            edge_importance = edge.properties.get("importance")
            strength = memory_strength(
                created_at=edge.created_at.timestamp() if edge.created_at else None,
                access_count=access.get("access_count", 0),
                confidence=edge.confidence,
                salience=edge_importance if edge_importance is not None else 0.5,
                memory_type=edge_memory_type,
            )
            if strength >= min_strength:
                edge_roles = self._extract_roles(edge)
                results.append(
                    {
                        "edge": edge,
                        "score": round(c["score"], 6),
                        "strength": round(strength, 6),
                        "text": c["text"],
                        "action": edge.type,
                        "memory_type": edge_memory_type,
                        "mood": edge.properties.get("mood", "actual"),
                        "negated": edge.properties.get("negated", False),
                        "roles": edge_roles,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                    }
                )

        return heapq.nlargest(limit, results, key=lambda x: x["score"] * x["strength"])

    # ==================================================================
    # consolidate  (S&B: embed nodes -> cosine >= 0.95 -> components -> merge)
    # ==================================================================

    def consolidate(self, entity: str | None = None) -> list[dict]:
        """S&B batch consolidation: node merging + edge grouping.

        Phase 1 -- Node merging (S&B's algorithm):
          Embed all nodes -> cosine >= 0.95 -> connected components ->
          representative = highest-degree node -> create same_as edges.

        Phase 2 -- Edge consolidation:
          Group active edges by vertex set -> create consolidated summary.

        Args:
            entity: If provided, only consolidate memories involving this entity.

        Returns:
            List of dicts with consolidation results.
        """
        if entity is not None:
            edges = self._hb.edges(containing=[entity], active=True)
        else:
            edges = self._hb.edges(active=True)

        edges = [e for e in edges if e.type not in _INFRA_TYPES]

        summaries: list[dict] = []

        # Phase 1: Node merging (S&B)
        if self._embedder is not None and self._hb.storage is not None:
            node_merges = self._consolidate_nodes(edges)
            summaries.extend(node_merges)

        # Phase 2: Edge consolidation
        if len(edges) >= 2:
            edge_summaries = self._consolidate_edges(edges)
            summaries.extend(edge_summaries)

        return summaries

    def _consolidate_nodes(self, edges: list[Any]) -> list[dict]:
        """S&B Phase 1: node merging via embedding similarity.

        1. Collect all entity nodes from edges
        2. Ensure all have embeddings
        3. Pairwise cosine >= 0.95 -> similarity graph
        4. Connected components -> equivalence classes
        5. Representative = highest-degree node
        6. Create same_as edges between members and representative
        """
        from hypabase.engine.vector import cosine_similarity, unpack_embedding

        # Collect unique nodes from edges
        all_node_ids: set[str] = set()
        for edge in edges:
            all_node_ids.update(edge.node_ids)

        if len(all_node_ids) < 2:
            return []

        # Ensure all nodes have embeddings
        for nid in all_node_ids:
            try:
                self._hb.embed_node(nid, text=nid)
            except Exception:
                pass

        # Load embeddings
        if self._hb.storage is None:
            return []
        raw_embeddings = self._hb.storage.load_embeddings(self._hb.current_namespace, kind="node")
        node_vectors: dict[str, list[float]] = {}
        for row in raw_embeddings:
            ref_id = row["ref_id"]
            if ref_id in all_node_ids and row.get("embedding") and row.get("dimension"):
                node_vectors[ref_id] = unpack_embedding(row["embedding"], row["dimension"])

        if len(node_vectors) < 2:
            return []

        # Build similarity graph (cosine >= 0.95)
        node_list = list(node_vectors.keys())
        adj: dict[str, set[str]] = defaultdict(set)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                sim = cosine_similarity(node_vectors[a], node_vectors[b])
                if sim >= 0.95:
                    adj[a].add(b)
                    adj[b].add(a)

        if not adj:
            return []

        # Connected components
        visited: set[str] = set()
        components: list[set[str]] = []
        for node in node_list:
            if node in visited or node not in adj:
                continue
            component: set[str] = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adj.get(current, set()) - visited)
            if len(component) >= 2:
                components.append(component)

        if not components:
            return []

        # For each component: representative = highest degree, create same_as edges
        summaries: list[dict] = []
        with self._hb.batch():
            for component in components:
                # S&B heuristic: representative = arg max deg(v)
                representative = max(
                    component,
                    key=lambda nid: len(self._hb.edges_of_node(nid)),
                )
                aliases = [nid for nid in component if nid != representative]

                for alias in aliases:
                    # Check if same_as already exists
                    existing = self._hb.edges_by_vertex_set([alias, representative])
                    if any(e.type == "same_as" and e.is_active for e in existing):
                        continue
                    self._hb.edge(
                        [alias, representative],
                        type="same_as",
                        source="consolidation",
                        confidence=0.95,
                    )

                # Update resolver cache
                self._resolver.register(representative, aliases)

                summaries.append(
                    {
                        "type": "node_merge",
                        "representative": representative,
                        "aliases": aliases,
                        "component_size": len(component),
                    }
                )

        return summaries

    def _consolidate_edges(self, edges: list[Any]) -> list[dict]:
        """Phase 2: group edges by vertex set, create consolidated summaries."""
        vertex_groups: dict[frozenset[str], list[Any]] = defaultdict(list)
        for edge in edges:
            vertex_groups[frozenset(edge.node_ids)].append(edge)

        summaries: list[dict] = []
        with self._hb.batch():
            for node_set, group_edges in vertex_groups.items():
                if len(group_edges) < 2 or len(node_set) < 2:
                    continue
                node_list = sorted(node_set)
                existing = self._hb.edges_by_vertex_set(node_list)
                if any(e.type == "consolidated" and e.is_active for e in existing):
                    continue
                source_ids = [e.id for e in group_edges]
                edge = self._hb.edge(
                    node_list,
                    type="consolidated",
                    source="consolidation",
                    confidence=min(1.0, len(group_edges) / 10.0),
                    properties={
                        "co_occurrence_count": len(group_edges),
                        "source_edge_ids": source_ids,
                    },
                )
                summaries.append(
                    {
                        "edge_id": edge.id,
                        "entities": node_list,
                        "source_edge_ids": source_ids,
                    }
                )

        return summaries

    # ==================================================================
    # forget  (two knobs: age + strength)
    # ==================================================================

    def forget(
        self,
        *,
        older_than: datetime | None = None,
        min_strength: float | None = None,
        entity: str | None = None,
    ) -> dict:
        """Expire (soft-delete) old or low-strength memories.

        Args:
            older_than: Expire edges created before this time.
            min_strength: Expire edges with strength below this threshold.
            entity: Only expire edges involving this entity.

        Returns:
            Dict with 'expired_count'.
        """
        if older_than is None and min_strength is None and entity is None:
            raise ValueError("At least one filter required: older_than, min_strength, or entity")

        edge_kwargs: dict[str, Any] = {"active": True, "include_expired": False}
        if entity is not None:
            edge_kwargs["containing"] = [entity]
        if older_than is not None:
            edge_kwargs["before"] = older_than

        edges = self._hb.edges(**edge_kwargs)

        # Exclude infrastructure edges
        edges = [e for e in edges if e.type not in _INFRA_TYPES]

        to_expire: list[str] = []

        if min_strength is not None:
            all_ids = [e.id for e in edges]
            batch_stats = self._get_batch_access_stats("edge", all_ids)
            for edge in edges:
                access = batch_stats.get(edge.id, {"access_count": 0, "last_accessed": 0.0})
                strength = memory_strength(
                    created_at=edge.created_at.timestamp() if edge.created_at else None,
                    access_count=access.get("access_count", 0),
                    confidence=edge.confidence,
                    salience=edge.properties.get("importance", 0.5),
                    memory_type=edge.properties.get("memory_type"),
                )
                if strength < min_strength:
                    to_expire.append(edge.id)
        else:
            to_expire = [e.id for e in edges]

        if to_expire:
            with self._hb.batch():
                for eid in to_expire:
                    self._hb.expire_edge(eid)

        return {"expired_count": len(to_expire)}

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _get_same_as_aliases(self, node_id: str) -> list[str]:
        """Get aliases via same_as edges in the graph."""
        aliases: list[str] = []
        for edge in self._hb.edges_of_node(node_id, edge_types=["same_as"]):
            if edge.is_active:
                for nid in edge.node_ids:
                    if nid != node_id:
                        aliases.append(nid)
        return aliases

    @staticmethod
    def _extract_roles(edge: Any) -> dict[str, str | list[str]]:
        """Extract a roles mapping {role: entity} from an edge's incidences."""
        roles: dict[str, str | list[str]] = {}
        for inc in edge.incidences:
            nid = inc.node_id if hasattr(inc, "node_id") else getattr(inc, "node_id", None)
            if nid is not None:
                role = inc.properties.get("role") if hasattr(inc, "properties") else None
                if role:
                    if role in roles:
                        existing = roles[role]
                        if isinstance(existing, list):
                            existing.append(nid)
                        else:
                            roles[role] = [existing, nid]
                    else:
                        roles[role] = nid
        return roles

    @staticmethod
    def _edge_has_role(edge: Any, role: str) -> bool:
        """Check if any node in the edge has the given role."""
        for inc in edge.incidences:
            inc_role = inc.properties.get("role") if hasattr(inc, "properties") else None
            if inc_role == role:
                return True
        return False

    @staticmethod
    def _edge_has_role_for_seeds(edge: Any, role: str, seed_ids: set[str]) -> bool:
        """Check if any seed entity has the given role in the edge."""
        for inc in edge.incidences:
            nid = inc.node_id if hasattr(inc, "node_id") else getattr(inc, "node_id", None)
            if nid in seed_ids:
                inc_role = inc.properties.get("role") if hasattr(inc, "properties") else None
                if inc_role == role:
                    return True
        return False

    def _get_batch_access_stats(self, kind: str, ref_ids: list[str]) -> dict[str, dict]:
        """Get access stats for multiple items in one query."""
        if self._hb.storage is None:
            return {}
        return self._hb.storage.get_batch_access_stats(self._hb.current_namespace, kind, ref_ids)
