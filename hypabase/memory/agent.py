"""Memory — opinionated AI agent memory built on Hypabase.

Provides remember/recall/forget/consolidate/connections operations on top of
the general-purpose hypergraph engine.  Uses kāraka (Sanskrit semantic roles)
to classify how entities participate in each memory.
"""

from __future__ import annotations

import heapq
import logging
from collections import Counter, deque
from datetime import datetime
from typing import Any

from hypabase.client import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.memory.resolution import EntityResolver
from hypabase.memory.strength import memory_strength
from hypabase.memory.types import (
    DEFAULT_ROLE_WEIGHT,
    MEMORY_TYPES,
    MOODS,
    ROLE_WEIGHTS,
    KarakaRole,
    MemoryType,
    Mood,
    ResolutionAction,
)

logger = logging.getLogger(__name__)


class Memory:
    """High-level memory interface for AI agents.

    Builds on Hypabase to provide:
    - ``remember()``: store ACTION + ENTITIES in ROLES as a hyperedge
    - ``recall()``: grammar-based recall with semantic vertex set lookup
    - ``forget()``: expire low-strength or old memories
    - ``consolidate()``: derive summary edges from episodic clusters
    - ``connections()``: explore entity neighborhoods (multi-hop BFS)

    Args:
        hb: An existing Hypabase instance to use. If None, one is created.
        path: Path for SQLite database (used only if hb is None).
        database: Namespace to use (default "memory").
        embedder: Embedding provider for semantic search.
    """

    _SEMANTIC_MIN_SCORE: float = 0.1

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
        self._resolver = EntityResolver(
            self._hb,
            embedder=self._embedder,
        )
        self._resolver.warm_cache()

    @property
    def hb(self) -> Hypabase:
        """The underlying Hypabase instance."""
        return self._hb

    # ==================================================================
    # remember
    # ==================================================================

    def remember(
        self,
        *,
        action: str,
        entities: list[dict[str, str]],
        text: str | None = None,
        memory_type: MemoryType | None = None,
        importance: float | None = None,
        mood: Mood | None = None,
        negated: bool = False,
        source: str = "memory",
        confidence: float = 1.0,
    ) -> dict:
        """Store a memory as ACTION + ENTITIES in ROLES.

        Args:
            action: The verb (e.g. "assigned", "prefers", "deployed").
            entities: Participants — list of dicts with 'name' (required),
                'role' (kāraka: agent/object/instrument/recipient/source/locus),
                and optional 'type' (default "entity").
            text: Optional human-readable form. Stored for display only.
            memory_type: "episodic", "semantic", or "procedural".
            importance: 0.0-1.0 salience rating.
            mood: "actual", "planned", "uncertain", or "normative".
            negated: True if the memory is a negation.
            source: Provenance source identifier.
            confidence: Provenance confidence 0.0-1.0.

        Returns:
            Dict with 'edge_id', 'node_ids', 'action', 'contradictions'.
        """
        # Validate
        if not action or not action.strip():
            raise ValueError("action is required (the verb).")
        if len(entities) < 2:
            raise ValueError(f"A memory needs at least 2 entities — it's a hyperedge. Got {len(entities)}.")
        for i, ent in enumerate(entities):
            if "name" not in ent or not ent["name"].strip():
                raise ValueError(f"Entity at index {i} is missing 'name'.")
        if mood is not None and mood not in MOODS:
            raise ValueError(f"Unknown mood {mood!r}. Must be one of: actual, planned, uncertain, normative.")
        if memory_type is not None and memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory_type {memory_type!r}. Must be one of: episodic, semantic, procedural.")

        action = action.strip().lower()

        # Resolve entities to canonical node IDs
        node_ids: list[str] = []
        roles: list[str | None] = []
        for ent in entities:
            resolved_id = self._resolver.resolve(
                ent["name"],
                entity_type=ent.get("type", "entity"),
            )
            self._hb.node(resolved_id, type=ent.get("type", "entity"))
            node_ids.append(resolved_id)
            roles.append(ent.get("role"))

        # Build edge properties
        props: dict[str, Any] = {}
        if text is not None:
            props["text"] = text
        if memory_type is not None:
            props["memory_type"] = memory_type
        if importance is not None:
            props["importance"] = importance
        if mood is not None:
            props["mood"] = mood
        if negated:
            props["negated"] = True

        # Create hyperedge
        edge_roles = roles if any(r is not None for r in roles) else None
        edge = self._hb.edge(
            node_ids,
            type=action,
            source=source,
            confidence=confidence,
            properties=props,
            roles=edge_roles,
        )

        # Embed for semantic recall
        if self._embedder is not None and self._hb.storage is not None:
            embed_text = text or f"{action} {' '.join(ent['name'] for ent in entities)}"
            try:
                self._hb.embed_edge(edge.id, embed_text)
            except Exception:
                logger.warning("Failed to embed edge %s", edge.id)

        # Contradiction detection (semantic memories only)
        contradictions: list[dict] = []
        if memory_type == "semantic":
            contradictions = self._detect_contradictions(edge, entities, action)

        return {
            "edge_id": edge.id,
            "node_ids": node_ids,
            "action": action,
            "contradictions": contradictions,
        }

    # ==================================================================
    # recall
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
        """Recall memories using the same grammar dimensions used to store them.

        Uses a three-step pipeline:
        1. **Resolve + Expand**: Build a scored seed set from entity names
           via identity resolution, same_as aliases, and semantic expansion.
        2. **Activate + Spread**: Find edges containing seed nodes, score by
           N-ary overlap, then spread activation with role-weighted propagation.
        3. **Filter + Rank**: Apply grammar filters and rank by
           activation_score × memory_strength.

        Args:
            entity: Entity name(s) for lookup. String or list of strings.
                Single: focused lookup. List: vertex set overlap query.
            action: Filter to edges of this action type (verb).
            role: Filter to edges where an entity has this kāraka role.
            memory_type: Filter by memory type (episodic/semantic/procedural).
            mood: Filter by mood — "actual", "planned", "uncertain",
                or "normative". Absent mood is treated as "actual".
            negated: Filter — True=only negated, False=only positive,
                None=all.
            since: Only include edges created after this datetime.
            before: Only include edges created before this datetime.
            limit: Maximum results.
            min_strength: Minimum memory strength threshold.

        Returns:
            List of dicts with 'edge', 'score', 'strength', 'text',
            'action', 'memory_type', 'mood', 'negated', and 'roles'.

        Raises:
            ValueError: If no filter dimension is provided.
        """
        if entity is None and action is None and memory_type is None and mood is None:
            raise ValueError("At least one of entity, action, memory_type, or mood must be provided.")
        if role is not None and entity is None:
            raise ValueError(
                "role filter requires entity to be specified. "
                "Use recall(entity=..., role=...) to find where an entity has a specific role."
            )

        # Normalize entity to list
        entity_list: list[str] = []
        if entity is not None:
            entity_list = [entity] if isinstance(entity, str) else list(entity)

        # ---- Step 1: RESOLVE + EXPAND (build seed set) ----
        seed_nodes: dict[str, float] = {}
        # Track original resolved IDs (before semantic expansion) for role filtering
        original_entity_ids: set[str] = set()
        if entity_list:
            original_entity_ids = {self._resolver.resolve(name) for name in entity_list}
            seed_nodes = self._expand_entities(entity_list)

        # ---- Step 2: ACTIVATE + SPREAD ----
        candidates: list[dict] = []
        seen_edge_ids: set[str] = set()

        if seed_nodes:
            edge_scores = self._spreading_activation(
                seed_nodes,
                depth=2,
                decay=0.5,
            )
            for edge_id, activation in edge_scores.items():
                if edge_id not in seen_edge_ids:
                    edge = self._hb.get_edge(edge_id)
                    if edge is not None and edge.is_active:
                        seen_edge_ids.add(edge_id)
                        candidates.append(
                            {
                                "edge": edge,
                                "score": activation,
                                "text": edge.properties.get("text", ""),
                            }
                        )
        else:
            # No entity provided — scan all active edges (filter-only mode)
            all_edges = self._hb.edges(active=True)
            for edge in all_edges:
                if edge.type in ("same_as", "consolidated"):
                    continue
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
            if original_entity_ids:
                # When entities provided, check role for the original query entities
                # (not the semantically-expanded seed set)
                candidates = [
                    c for c in candidates if self._edge_has_role_for_seeds(c["edge"], role, original_entity_ids)
                ]
            else:
                # No entities — check if any node has the role
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

        # Batch record access — single commit
        all_edge_ids = [c["edge"].id for c in candidates]
        if self._hb.storage is not None:
            self._hb.storage.record_access_batch(self._hb.current_namespace, "edge", all_edge_ids)

        # Batch fetch stats — single query
        batch_stats = self._get_batch_access_stats("edge", all_edge_ids)

        # Compute final score = activation × strength, filter, return top-k
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
                    }
                )

        return heapq.nlargest(limit, results, key=lambda x: x["score"] * x["strength"])

    # ==================================================================
    # forget
    # ==================================================================

    def forget(
        self,
        *,
        older_than: datetime | None = None,
        min_strength: float | None = None,
        entity: str | None = None,
        memory_type: str | None = None,
        mood: str | None = None,
    ) -> dict:
        """Expire (soft-delete) low-strength or old memories.

        Args:
            older_than: Expire edges created before this time.
            min_strength: Expire edges with strength below this threshold.
                Edges with strength >= min_strength are kept.
            entity: Only expire edges involving this entity.
            memory_type: Only expire edges of this memory type.
            mood: Only expire edges with this explicit mood. Memories
                without mood set are not matched (protective default).

        Returns:
            Dict with 'expired_count' and optionally 'skipped_untagged'
            (number of memories without explicit mood that were protected).
        """
        if older_than is None and min_strength is None and entity is None and memory_type is None and mood is None:
            raise ValueError("At least one filter required: older_than, min_strength, entity, memory_type, or mood")

        # Build kwargs for edges() to push filtering down
        edge_kwargs: dict[str, Any] = {"active": True, "include_expired": False}
        if entity is not None:
            edge_kwargs["containing"] = [entity]
        if older_than is not None:
            edge_kwargs["before"] = older_than

        edges = self._hb.edges(**edge_kwargs)

        # Exclude infrastructure edges (same_as) — only forget actual memories
        edges = [e for e in edges if e.type != "same_as"]
        # Apply memory_type filter (not pushable to SQL)
        if memory_type is not None:
            edges = [e for e in edges if e.properties.get("memory_type") == memory_type]
        skipped_untagged = 0
        if mood is not None:
            filtered = []
            for e in edges:
                stored_mood = e.properties.get("mood")
                if stored_mood == mood:
                    filtered.append(e)
                elif stored_mood is None:
                    skipped_untagged += 1
            edges = filtered

        to_expire: list[str] = []

        # Batch-fetch access stats if we need strength filtering
        batch_stats: dict[str, dict] = {}
        if min_strength is not None:
            all_ids = [e.id for e in edges]
            batch_stats = self._get_batch_access_stats("edge", all_ids)

        for edge in edges:
            # If min_strength filter is set, only expire edges below threshold
            if min_strength is not None:
                access = batch_stats.get(edge.id, {"access_count": 0, "last_accessed": 0.0})
                edge_memory_type = edge.properties.get("memory_type")
                strength = memory_strength(
                    created_at=edge.created_at.timestamp() if edge.created_at else None,
                    access_count=access.get("access_count", 0),
                    confidence=edge.confidence,
                    salience=edge.properties.get("importance", 0.5),
                    memory_type=edge_memory_type,
                )
                if strength < min_strength:
                    to_expire.append(edge.id)
            else:
                # No strength filter — all edges matching temporal/entity filters
                # are candidates (already filtered by SQL via before= and containing=)
                to_expire.append(edge.id)

        # Batch all expires — single namespace write instead of N
        if to_expire:
            with self._hb.batch():
                for eid in to_expire:
                    self._hb.expire_edge(eid)

            # Prune orphaned same_as edges for affected entities
            affected_nodes: set[str] = set()
            for eid in to_expire:
                expired_edge = self._hb.get_edge(eid)
                if expired_edge:
                    affected_nodes.update(expired_edge.node_ids)
            self._prune_orphaned_same_as(affected_nodes)

        result = {"expired_count": len(to_expire)}
        if mood is not None and skipped_untagged > 0:
            result["skipped_untagged"] = skipped_untagged
        return result

    # ==================================================================
    # consolidate
    # ==================================================================

    def consolidate(self, entity: str | None = None) -> list[dict]:
        """Derive summary edges from clusters of related memories.

        Groups edges by vertex set and creates N-ary consolidated edges
        preserving hypergraph structure.

        Args:
            entity: If provided, only consolidate memories involving this entity.

        Returns:
            List of dicts with 'edge_id', 'entities', and 'source_edge_ids'.
        """
        # Push entity filter to SQL level instead of filtering in Python
        if entity is not None:
            edges = self._hb.edges(containing=[entity], active=True)
        else:
            edges = self._hb.edges(active=True)

        # Exclude infrastructure and already-consolidated edges
        _SKIP_TYPES = {"consolidated", "same_as"}
        edges = [e for e in edges if e.type not in _SKIP_TYPES]

        if len(edges) < 2:
            return []

        # Group edges by their vertex sets
        from collections import defaultdict

        vertex_groups: dict[frozenset[str], list[Any]] = defaultdict(list)
        for edge in edges:
            vertex_groups[frozenset(edge.node_ids)].append(edge)

        summaries = []
        with self._hb.batch():
            # Process exact vertex-set groups first
            for node_set, group_edges in vertex_groups.items():
                if len(group_edges) >= 2:
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

            # Fall back to pairwise co-occurrence for cross-group patterns
            pair_counts: Counter[tuple[str, str]] = Counter()
            for edge in edges:
                if edge.type != "consolidated":
                    nodes = sorted(edge.node_ids)
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            pair_counts[(nodes[i], nodes[j])] += 1

            for (a, b), count in pair_counts.most_common(10):
                if count < 2:
                    break
                existing = self._hb.edges_by_vertex_set([a, b])
                if any(e.type == "consolidated" and e.is_active for e in existing):
                    continue
                edge = self._hb.edge(
                    [a, b],
                    type="consolidated",
                    source="consolidation",
                    confidence=min(1.0, count / 10.0),
                    properties={"co_occurrence_count": count},
                )
                summaries.append(
                    {
                        "edge_id": edge.id,
                        "entities": [a, b],
                    }
                )

        return summaries

    # ==================================================================
    # connections (multi-hop BFS)
    # ==================================================================

    def connections(
        self,
        entity: str,
        *,
        max_hops: int = 2,
        edge_types: list[str] | None = None,
        role: str | None = None,
    ) -> dict:
        """Explore an entity's neighborhood in the memory graph using BFS.

        Args:
            entity: The entity name to start from.
            max_hops: Maximum traversal depth.
            edge_types: Filter by edge types.
            role: Filter by kāraka role.

        Returns:
            Dict with 'entity', 'aliases', 'neighbors', 'edges', and counts.
        """
        visited_nodes: set[str] = {entity}
        visited_edges: set[str] = set()
        all_neighbors: list[dict] = []
        all_edges: list[dict] = []
        aliases: set[str] = set()

        # BFS queue: (node_id, current_depth)
        queue: deque[tuple[str, int]] = deque([(entity, 0)])

        while queue:
            current_node, depth = queue.popleft()
            if depth >= max_hops:
                continue

            edges = self._hb.edges_of_node(current_node, edge_types=edge_types)
            for edge in edges:
                if not edge.is_active or edge.id in visited_edges:
                    continue

                # Collect aliases from same_as edges instead of traversing them
                if edge.type == "same_as" and edge_types is None:
                    visited_edges.add(edge.id)
                    for nid in edge.node_ids:
                        if nid != current_node and nid not in aliases:
                            aliases.add(nid)
                    continue

                # Apply role filter
                if role is not None and not self._edge_has_role(edge, role):
                    continue

                visited_edges.add(edge.id)
                all_edges.append(
                    {
                        "id": edge.id,
                        "type": edge.type,
                        "node_ids": edge.node_ids,
                        "text": edge.properties.get("text", ""),
                    }
                )

                # Enqueue unvisited neighbor nodes
                for nid in edge.node_ids:
                    if nid not in visited_nodes:
                        visited_nodes.add(nid)
                        node = self._hb.get_node(nid)
                        if node is not None:
                            all_neighbors.append({"id": node.id, "type": node.type})
                        queue.append((nid, depth + 1))

        return {
            "entity": entity,
            "aliases": sorted(aliases),
            "neighbors": all_neighbors,
            "edges": all_edges,
            "edge_count": len(all_edges),
            "neighbor_count": len(all_neighbors),
        }

    # ==================================================================
    # resolve_contradiction
    # ==================================================================

    def resolve_contradiction(
        self,
        new_edge_id: str,
        old_edge_id: str,
        resolution: ResolutionAction,
    ) -> dict:
        """Resolve a detected contradiction between two memories.

        Args:
            new_edge_id: The ID of the newer memory edge.
            old_edge_id: The ID of the older memory edge.
            resolution: One of "supersede" (expire old), "keep_both",
                or "keep_old" (expire new).

        Returns:
            Dict describing the resolution outcome.
        """
        valid = ("supersede", "keep_both", "keep_old")
        if resolution not in valid:
            raise ValueError(f"Unknown resolution: {resolution!r}. Must be one of {valid}.")
        if resolution == "supersede":
            self._hb.expire_edge(old_edge_id)
            return {"resolved": True, "action": "superseded", "expired": old_edge_id}
        elif resolution == "keep_old":
            self._hb.expire_edge(new_edge_id)
            return {"resolved": True, "action": "kept_old", "expired": new_edge_id}
        else:  # keep_both
            return {"resolved": True, "action": "kept_both"}

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _prune_orphaned_same_as(self, node_ids: set[str]) -> None:
        """Remove same_as edges where neither endpoint has active memories."""
        checked: set[str] = set()
        for nid in node_ids:
            for edge in self._hb.edges_of_node(nid):
                if edge.id in checked or edge.type != "same_as" or not edge.is_active:
                    continue
                checked.add(edge.id)
                has_memory = False
                for endpoint in edge.node_ids:
                    node_edges = self._hb.edges_of_node(endpoint)
                    if any(e.is_active and e.type not in ("same_as", "consolidated") for e in node_edges):
                        has_memory = True
                        break
                if not has_memory:
                    self._hb.expire_edge(edge.id)

    def _detect_contradictions(
        self,
        edge: Any,
        entities: list[dict[str, str]],
        action: str,
    ) -> list[dict]:
        """Detect contradictions with existing semantic memories.

        Only checks for semantic memories with the same action type and
        overlapping agent entities where the objects differ.

        Returns:
            List of contradiction dicts with 'existing_edge_id' and 'description'.
        """
        # Find agent entities in the new memory
        agents = [e["name"] for e in entities if e.get("role") == "agent"]
        objects = [e["name"] for e in entities if e.get("role") == "object"]
        if not agents or not objects:
            return []

        contradictions: list[dict] = []
        for agent_name in agents:
            existing_edges = self._hb.edges(containing=[agent_name], active=True)
            for existing in existing_edges:
                if existing.id == edge.id:
                    continue
                if existing.type != action:
                    continue
                if existing.properties.get("memory_type") != "semantic":
                    continue
                # Check if the existing edge has different objects
                existing_roles = self._extract_roles(existing)
                existing_objects = [nid for nid, r in existing_roles.items() if r == "object"]
                if existing_objects and set(existing_objects) != set(objects):
                    contradictions.append(
                        {
                            "existing_edge_id": existing.id,
                            "new_edge_id": edge.id,
                            "description": (
                                f"Existing memory has {agent_name} {action} "
                                f"{existing_objects} but new memory says {objects}"
                            ),
                        }
                    )
        return contradictions

    def _expand_entities(self, names: list[str]) -> dict[str, float]:
        """Expand entity names to a scored seed set.

        Three levels:
        1. Identity resolution → canonical node (score 1.0)
        2. Same_as graph edges → aliases (score 0.95)
        3. Node embedding search → semantically similar (score 0.5-0.8)

        Returns dict of node_id → activation score.
        """
        seed_nodes: dict[str, float] = {}
        for name in names:
            # 1. Identity resolution
            resolved = self._resolver.resolve(name)
            seed_nodes[resolved] = 1.0

            # 2. Same_as aliases from graph
            for alias in self._get_same_as_aliases(resolved):
                seed_nodes.setdefault(alias, 0.95)

            # 3. Semantic expansion via node embeddings
            if self._embedder is not None and self._hb.storage is not None:
                try:
                    similar_nodes = self._hb.search(
                        name,
                        kind="node",
                        limit=3,
                        min_score=0.7,
                    )
                    for s in similar_nodes:
                        if s["ref_id"] not in seed_nodes:
                            seed_nodes[s["ref_id"]] = s["score"] * 0.7
                except Exception:
                    logger.debug("Semantic expansion failed for %r", name)
        return seed_nodes

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
    def _get_node_role_in_edge(edge: Any, node_id: str) -> str | None:
        """Get the kāraka role of a node in an edge."""
        for inc in edge.incidences:
            nid = inc.node_id if hasattr(inc, "node_id") else getattr(inc, "node_id", None)
            if nid == node_id:
                return inc.properties.get("role") if hasattr(inc, "properties") else None
        return None

    def _spreading_activation(
        self,
        seed_nodes: dict[str, float],
        *,
        depth: int = 2,
        decay: float = 0.5,
        min_activation: float = 0.05,
    ) -> dict[str, float]:
        """BFS spreading activation from scored seed nodes.

        Scores edges by combining:
        - N-ary overlap: ``len(seed ∩ edge.nodes) / len(query_entities)``
        - Propagated activation with role-weighted decay

        Returns:
            Dict mapping edge_id → activation_score.
        """
        seed_set = set(seed_nodes.keys())
        num_query = len(seed_set)
        edge_activation: dict[str, float] = {}
        visited_nodes: set[str] = set(seed_nodes.keys())

        queue: deque[tuple[str, float, int]] = deque((nid, score, 0) for nid, score in seed_nodes.items())

        while queue:
            node_id, activation, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            edges = self._hb.edges_of_node(node_id)

            for edge in edges:
                if not edge.is_active or edge.type in ("same_as", "consolidated"):
                    continue

                # N-ary overlap score
                overlap_count = len(seed_set & edge.node_set)
                overlap_score = overlap_count / num_query if num_query > 0 else 0.0

                # Role-weighted propagated activation
                role = self._get_node_role_in_edge(edge, node_id)
                role_weight = ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHT) if role else DEFAULT_ROLE_WEIGHT
                propagated = activation * decay * role_weight

                # Edge score = max of overlap and propagated
                edge_score = max(overlap_score, propagated)

                # Keep max activation per edge
                if edge.id not in edge_activation or edge_activation[edge.id] < edge_score:
                    edge_activation[edge.id] = edge_score

                # Spread to neighbor nodes
                if propagated >= min_activation:
                    for nid in edge.node_ids:
                        if nid not in visited_nodes:
                            visited_nodes.add(nid)
                            queue.append((nid, propagated, current_depth + 1))

        return edge_activation

    @staticmethod
    def _extract_roles(edge: Any) -> dict[str, str]:
        """Extract a roles mapping {node_id: role} from an edge's incidences."""
        roles: dict[str, str] = {}
        for inc in edge.incidences:
            nid = inc.node_id if hasattr(inc, "node_id") else getattr(inc, "node_id", None)
            if nid is not None:
                role = inc.properties.get("role") if hasattr(inc, "properties") else None
                if role:
                    roles[nid] = role
        return roles

    @staticmethod
    def _node_has_role_in_edge(edge: Any, node_id: str, role: str) -> bool:
        """Check if *node_id* has the given *role* in *edge*."""
        for inc in edge.incidences:
            nid = inc.node_id if hasattr(inc, "node_id") else getattr(inc, "node_id", None)
            if nid == node_id:
                inc_role = inc.properties.get("role") if hasattr(inc, "properties") else None
                if inc_role == role:
                    return True
        return False

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

    def _get_access_stats(self, kind: str, ref_id: str) -> dict:
        """Get access stats, returning defaults if not available."""
        if self._hb.storage is None:
            return {"access_count": 0, "last_accessed": 0.0}
        stats = self._hb.storage.get_access_stats(self._hb.current_namespace, kind, ref_id)
        return stats or {"access_count": 0, "last_accessed": 0.0}

    def _get_batch_access_stats(self, kind: str, ref_ids: list[str]) -> dict[str, dict]:
        """Get access stats for multiple items in one query."""
        if self._hb.storage is None:
            return {}
        return self._hb.storage.get_batch_access_stats(self._hb.current_namespace, kind, ref_ids)


# Backward-compat alias
MemoryAgent = Memory
