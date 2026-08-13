"""Unit tests for get_base_indices_for_anchored_blocks and select_anchors."""

import pytest
import torch

from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)


class TestGetBaseIndicesForAnchoredBlocks:
    def test_single_anchor(self):
        anchor_positions = torch.tensor([[3]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=4)
        expected = torch.tensor([3, 4, 5, 6])
        assert torch.equal(result, expected)

    def test_multiple_anchors(self):
        anchor_positions = torch.tensor([[0, 5, 10]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=3)
        expected = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12])
        assert torch.equal(result, expected)

    def test_block_size_one(self):
        anchor_positions = torch.tensor([[2, 7, 9]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=1)
        expected = torch.tensor([2, 7, 9])
        assert torch.equal(result, expected)

    def test_1d_input(self):
        anchor_positions = torch.tensor([1, 4])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        expected = torch.tensor([1, 2, 4, 5])
        assert torch.equal(result, expected)

    def test_output_shape(self):
        num_anchors = 5
        block_size = 4
        anchor_positions = torch.tensor([[0, 3, 6, 9, 12]])
        result = get_base_indices_for_anchored_blocks(
            anchor_positions, block_size=block_size
        )
        assert result.shape == (num_anchors * block_size,)

    def test_output_dtype_is_long(self):
        anchor_positions = torch.tensor([[2.0, 5.0]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        assert result.dtype == torch.long


class TestSelectAnchors:
    def test_explicit_candidate_mask_overrides_individual_loss_tokens(self):
        loss_mask = torch.ones(1, 16)
        candidate_mask = torch.zeros(1, 16, dtype=torch.bool)
        candidate_mask[:, [3, 9]] = True
        anchors, anchor_valid = select_anchors(
            loss_mask,
            num_anchors=8,
            block_size=2,
            anchor_candidate_mask=candidate_mask,
        )
        assert set(anchors[anchor_valid].tolist()) == {3, 9}

    def test_sampled_anchors_are_sorted(self):
        # Anchors are returned sorted by position so the draft blocks form
        # contiguous flex-attention blocks (fast path) instead of scattered ones.
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 64)
        anchors, anchor_valid = select_anchors(loss_mask, num_anchors=8, block_size=4)
        selected = anchors[anchor_valid]
        assert torch.equal(selected, torch.sort(selected).values)

    def test_default_uniform_fixed_seed_is_unchanged(self):
        loss_mask = torch.tensor(
            [[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        valid_mask = loss_mask.bool().clone()
        valid_mask[:, -3:] = False
        valid_indices = torch.nonzero(
            valid_mask.squeeze(0), as_tuple=False
        ).squeeze(-1)

        torch.manual_seed(1234)
        expected_perm = torch.randperm(valid_indices.numel())
        expected = torch.sort(valid_indices[expected_perm[:5]]).values
        torch.manual_seed(1234)
        anchors, anchor_valid = select_anchors(
            loss_mask, num_anchors=5, block_size=3
        )

        assert torch.equal(anchors[anchor_valid], expected)

    def test_long_context_mix_is_sorted_unique_and_backfills(self):
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 32)
        position_ids = torch.arange(32).unsqueeze(0)
        anchors, anchor_valid = select_anchors(
            loss_mask,
            num_anchors=20,
            block_size=4,
            position_ids=position_ids,
            sampling="long-context-mix",
            tail_fraction=0.5,
            position_boundaries=[24],
            position_weights=[1.0, 100.0],
        )
        selected = anchors[anchor_valid]

        assert selected.numel() == 20
        assert torch.equal(selected, torch.sort(selected).values)
        assert torch.unique(selected).numel() == selected.numel()
        assert int(selected.max()) < 28

    def test_long_context_mix_uses_per_document_positions(self):
        # Two packed short documents have identical 0..15 positions. The second
        # document must not be mistaken for a 16..31 long-context suffix.
        loss_mask = torch.ones(1, 32)
        position_ids = torch.arange(16).repeat(2).unsqueeze(0)
        second_document_count = 0
        total = 0
        for seed in range(100):
            torch.manual_seed(seed)
            anchors, anchor_valid = select_anchors(
                loss_mask,
                num_anchors=8,
                block_size=1,
                position_ids=position_ids,
                sampling="long-context-mix",
                tail_fraction=1.0,
                position_boundaries=[16],
                position_weights=[1.0, 100.0],
            )
            selected = anchors[anchor_valid]
            second_document_count += int((selected >= 16).sum())
            total += selected.numel()

        second_document_fraction = second_document_count / total
        assert 0.4 < second_document_fraction < 0.6

    def test_long_context_mix_masks_invalid_and_pads_shortfall(self):
        torch.manual_seed(1)
        loss_mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 1]], dtype=torch.float32)
        anchors, anchor_valid = select_anchors(
            loss_mask,
            num_anchors=8,
            block_size=2,
            position_ids=torch.arange(8).unsqueeze(0),
            sampling="long-context-mix",
        )

        selected = anchors[anchor_valid]
        assert set(selected.tolist()) == {0, 2, 3, 5}
        assert not anchor_valid[4:].any()

    def test_long_context_mix_rejects_missing_position_ids(self):
        with pytest.raises(ValueError, match="requires position_ids"):
            select_anchors(
                torch.ones(1, 8),
                num_anchors=2,
                block_size=1,
                sampling="long-context-mix",
            )
