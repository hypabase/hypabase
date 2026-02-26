"""Entity resolution for the Hypabase Memory Module.

Thin normalized-name cache. Populated by warm_cache() (from existing nodes
and same_as edges) and by consolidation (batch synonym merging).
At write time, only exact normalized matches are used (S&B approach).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypabase.client import Hypabase


class EntityResolver:
    """Normalized-name cache for entity resolution.

    Populated by warm_cache() (from existing nodes + same_as edges)
    and by consolidation (batch synonym merging).
    At write time, only exact normalized matches are used (S&B approach).
    """

    def __init__(self, hb: Hypabase) -> None:
        self._hb = hb
        # normalized_name -> canonical node_id
        self._cache: dict[str, str] = {}

    def warm_cache(self) -> None:
        """Pre-populate from existing nodes and same_as edges."""
        nodes = list(self._hb.nodes())
        # Pass 1: all node IDs
        for node in nodes:
            norm = self._normalize(node.id)
            self._cache[norm] = node.id
        # Pass 2: alias properties override standalone registrations
        for node in nodes:
            for alias in node.properties.get("aliases", []):
                alias_norm = self._normalize(alias)
                self._cache[alias_norm] = node.id
        # Pass 3: same_as edge mappings (degree-based canonical)
        seen_edges: set[str] = set()
        for node in nodes:
            for edge in self._hb.edges_of_node(node.id, edge_types=["same_as"]):
                if edge.id in seen_edges or not edge.is_active:
                    continue
                seen_edges.add(edge.id)
                canonical = max(
                    edge.node_ids,
                    key=lambda nid: len(self._hb.edges_of_node(nid)),
                )
                for nid in edge.node_ids:
                    if nid != canonical:
                        self._cache[self._normalize(nid)] = canonical

    def resolve(self, name: str) -> str:
        """Exact normalized match only. Returns name unchanged if no match."""
        norm = self._normalize(name)
        if not norm:
            return name
        return self._cache.get(norm, name)

    def is_known(self, name: str) -> bool:
        """Check if a name is in the cache."""
        norm = self._normalize(name)
        return bool(norm) and norm in self._cache

    def register(self, canonical: str, aliases: list[str]) -> None:
        """Register synonym mappings (called by consolidation)."""
        canon_norm = self._normalize(canonical)
        self._cache[canon_norm] = canonical
        for alias in aliases:
            self._cache[self._normalize(alias)] = canonical

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", name.strip().lower())
