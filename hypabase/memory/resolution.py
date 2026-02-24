"""Entity resolution for the Hypabase Memory Module.

Resolves entity names to canonical node IDs using normalization, alias
detection, and optional embedding similarity.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from hypabase.engine.embeddings import EmbeddingProvider

if TYPE_CHECKING:
    from hypabase.client import Hypabase


class EntityResolver:
    """Resolve entity names to canonical node IDs.

    Resolution chain (in order):
    1. Exact normalized match (fast, from cache)
    2. Prefix/suffix alias detection ("Bob" <-> "Bob Jones")
    3. Embedding similarity (cosine > threshold) when embedder available

    Args:
        hb: The Hypabase client instance.
        embedder: Optional embedding provider for similarity-based resolution.
        similarity_threshold: Minimum cosine similarity for embedding match (default 0.92).
    """

    def __init__(
        self,
        hb: Hypabase,
        embedder: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.92,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {similarity_threshold}")
        self._hb = hb
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        # normalized_name -> canonical node_id
        self._cache: dict[str, str] = {}

    def warm_cache(self) -> None:
        """Pre-populate the name cache from existing nodes.

        Two passes: first register all node IDs, then override with
        alias mappings so canonical names always win.
        """
        nodes = list(self._hb.nodes())
        for node in nodes:
            norm = self._normalize(node.id)
            self._cache[norm] = node.id
        # Second pass: aliases override standalone registrations
        for node in nodes:
            for alias in node.properties.get("aliases", []):
                alias_norm = self._normalize(alias)
                self._cache[alias_norm] = node.id

    def resolve(self, name: str, entity_type: str = "entity") -> str:
        """Resolve *name* to a canonical node ID.

        Creates the node if no match is found.  When a new alias is
        discovered, it is stored in ``node.properties["aliases"]``.

        Args:
            name: The entity name as provided by the caller.
            entity_type: Node type to use when creating new nodes.

        Returns:
            Canonical node ID.
        """
        norm = self._normalize(name)
        if not norm:
            return name

        # 1. Exact normalized match
        if norm in self._cache:
            return self._cache[norm]

        # 2. Prefix/suffix alias detection
        for cached_norm, canonical_id in self._cache.items():
            if self._is_alias(norm, cached_norm):
                self._cache[norm] = canonical_id
                self._add_alias(canonical_id, name)
                self._create_same_as_edge(name, canonical_id)
                return canonical_id

        # 3. Embedding similarity (when embedder available)
        if self._embedder is not None and self._hb.storage is not None:
            match = self._embedding_match(name)
            if match is not None:
                self._cache[norm] = match
                self._add_alias(match, name)
                self._create_same_as_edge(name, match)
                return match

        # No match — register as new entity
        self._cache[norm] = name
        # Embed the name for future resolution
        if self._embedder is not None and self._hb.storage is not None:
            self._hb.node(name, type=entity_type)
            self._hb.embed_node(name, text=name)
        return name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", name.strip().lower())

    @staticmethod
    def _is_alias(a: str, b: str) -> bool:
        """Check if *a* is a prefix/suffix alias of *b* or vice versa.

        Matches cases like "bob" <-> "bob jones" (first-name alias).
        Requires the shorter string to be a full word boundary in the longer.
        """
        if a == b:
            return False
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        # Short must be at least 3 chars to avoid spurious matches
        if len(short) < 3:
            return False
        # Must match at start or end as a whole word
        return long.startswith(short + " ") or long.endswith(" " + short)

    def _embedding_match(self, name: str) -> str | None:
        """Find a matching node by embedding similarity."""
        try:
            results = self._hb.search(
                name, limit=1, kind="node", min_score=self._similarity_threshold,
            )
        except (ImportError, ValueError):
            return None
        if results:
            return results[0]["ref_id"]
        return None

    def _add_alias(self, node_id: str, alias: str) -> None:
        """Store *alias* in the node's ``properties["aliases"]`` list."""
        node = self._hb.get_node(node_id)
        if node is None:
            return
        aliases: list[str] = list(node.properties.get("aliases", []))
        if alias not in aliases and alias != node_id:
            aliases.append(alias)
            self._hb.node(node_id, type=node.type, aliases=aliases)

    def _create_same_as_edge(self, alias_name: str, canonical_id: str) -> None:
        """Create a same_as edge between alias and canonical entity."""
        self._hb.node(alias_name, type="alias")
        self._hb.upsert_edge_by_vertex_set(
            node_ids={alias_name, canonical_id},
            edge_type="same_as",
            source="entity_resolution",
            confidence=0.95,
        )
