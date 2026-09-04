# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for community merging during an incremental update.

The delta pipeline mints its own entity and relationship ids. When the merge keeps a
pre-existing row on a collision, the delta id is discarded, so community membership
lists carried over from the delta run must be repointed at the surviving ids.
"""

import pandas as pd
from graphrag.index.update.communities import _update_and_merge_communities
from graphrag.index.update.relationships import _update_and_merge_relationships


def _community_row(
    community: int,
    entity_ids: list[str],
    relationship_ids: list[str],
    community_id: str = "c1",
    level: int = 0,
    parent: int = -1,
) -> dict:
    """Build a community row matching COMMUNITIES_FINAL_COLUMNS shape."""
    return {
        "id": community_id,
        "human_readable_id": community,
        "community": community,
        "level": level,
        "parent": parent,
        "children": [],
        "title": f"Community {community}",
        "entity_ids": entity_ids,
        "relationship_ids": relationship_ids,
        "text_unit_ids": ["tu1"],
        "period": "2026-01-01",
        "size": len(entity_ids),
    }


def _relationship_row(
    source: str,
    target: str,
    relationship_id: str,
    human_readable_id: int = 0,
) -> dict:
    """Build a relationship row matching RELATIONSHIPS_FINAL_COLUMNS shape."""
    return {
        "id": relationship_id,
        "human_readable_id": human_readable_id,
        "source": source,
        "target": target,
        "description": "desc",
        "weight": 1.0,
        "combined_degree": 2,
        "text_unit_ids": ["tu1"],
    }


class TestRelationshipIdMapping:
    """_update_and_merge_relationships reports which delta ids were superseded."""

    def test_matching_pair_is_mapped(self):
        old = pd.DataFrame([_relationship_row("A", "B", "r-old")])
        delta = pd.DataFrame([_relationship_row("A", "B", "r-delta")])

        _, id_mapping = _update_and_merge_relationships(old, delta)

        assert id_mapping == {"r-delta": "r-old"}

    def test_distinct_pairs_are_not_mapped(self):
        old = pd.DataFrame([_relationship_row("A", "B", "r-old")])
        delta = pd.DataFrame([_relationship_row("C", "D", "r-delta")])

        _, id_mapping = _update_and_merge_relationships(old, delta)

        assert id_mapping == {}


class TestCommunityIdRemapping:
    """Delta community membership is repointed at surviving ids."""

    def test_entity_ids_are_remapped(self):
        old = pd.DataFrame([_community_row(0, ["e-1", "e-2"], ["r-1"])])
        delta = pd.DataFrame([
            _community_row(0, ["d-1", "d-2"], ["r-d1"], community_id="c2")
        ])

        merged, _ = _update_and_merge_communities(
            old,
            delta,
            entity_id_mapping={"d-1": "e-1"},
            relationship_id_mapping={},
        )

        delta_row = merged[merged["community"] == 1].iloc[0]
        assert list(delta_row["entity_ids"]) == ["e-1", "d-2"]

    def test_relationship_ids_are_remapped(self):
        old = pd.DataFrame([_community_row(0, ["e-1"], ["r-1"])])
        delta = pd.DataFrame([
            _community_row(0, ["d-1"], ["r-d1", "r-d2"], community_id="c2")
        ])

        merged, _ = _update_and_merge_communities(
            old,
            delta,
            entity_id_mapping={},
            relationship_id_mapping={"r-d1": "r-1"},
        )

        delta_row = merged[merged["community"] == 1].iloc[0]
        assert list(delta_row["relationship_ids"]) == ["r-1", "r-d2"]

    def test_old_communities_are_untouched(self):
        old = pd.DataFrame([_community_row(0, ["e-1", "d-1"], ["r-1"])])
        delta = pd.DataFrame([_community_row(0, ["d-1"], ["r-d1"], community_id="c2")])

        merged, _ = _update_and_merge_communities(
            old,
            delta,
            entity_id_mapping={"d-1": "e-9"},
            relationship_id_mapping={},
        )

        old_row = merged[merged["community"] == 0].iloc[0]
        assert list(old_row["entity_ids"]) == ["e-1", "d-1"]

    def test_no_mapping_leaves_membership_unchanged(self):
        old = pd.DataFrame([_community_row(0, ["e-1"], ["r-1"])])
        delta = pd.DataFrame([_community_row(0, ["d-1"], ["r-d1"], community_id="c2")])

        merged, _ = _update_and_merge_communities(old, delta)

        delta_row = merged[merged["community"] == 1].iloc[0]
        assert list(delta_row["entity_ids"]) == ["d-1"]
        assert list(delta_row["relationship_ids"]) == ["r-d1"]

    def test_community_numbering_still_offset(self):
        old = pd.DataFrame([_community_row(7, ["e-1"], ["r-1"])])
        delta = pd.DataFrame([_community_row(0, ["d-1"], ["r-d1"], community_id="c2")])

        merged, community_id_mapping = _update_and_merge_communities(
            old, delta, entity_id_mapping={"d-1": "e-1"}
        )

        assert community_id_mapping[0] == 8
        assert set(merged["community"]) == {7, 8}
