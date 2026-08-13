import torch

from speculators.train.long_context import (
    build_anchor_candidate_mask,
    stretched_position_ids,
)


def test_anchor_candidate_requires_complete_safe_block():
    tokens = torch.tensor([1, 2, 3, 99, 5, 6, 7, 8, 9, 10])
    loss = torch.ones(10, dtype=torch.bool)
    candidates = build_anchor_candidate_mask(
        tokens, loss, block_size=3, excluded_token_ids=(99,)
    )
    assert candidates.tolist() == [
        True, False, False, False, True, True, True, True, False, False
    ]


def test_anchor_candidate_respects_synthetic_boundary():
    tokens = torch.arange(12)
    loss = torch.ones(12, dtype=torch.bool)
    candidates = build_anchor_candidate_mask(
        tokens, loss, block_size=4, allowed_start=6, allowed_end=11
    )
    assert torch.nonzero(candidates).view(-1).tolist() == [6, 7]


def test_distance_stretch_preserves_local_relative_distances():
    positions = stretched_position_ids(
        seq_len=12000,
        response_start=10000,
        target_anchor_position=59000,
        near_window=4096,
    )
    assert positions[10000].item() == 59000
    assert torch.equal(
        positions[5904:] - positions[5904], torch.arange(12000 - 5904)
    )
    assert torch.all(positions[1:] >= positions[:-1])
