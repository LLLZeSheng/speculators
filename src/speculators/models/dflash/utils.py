"""Utility functions for DFlash draft model."""

from collections.abc import Sequence

import torch


def _validate_long_context_sampling(
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    tail_fraction: float,
    position_boundaries: Sequence[int],
    position_weights: Sequence[float],
) -> torch.Tensor:
    if position_ids is None or position_ids.shape != loss_mask.shape:
        shape = None if position_ids is None else tuple(position_ids.shape)
        raise ValueError(
            "long-context-mix requires position_ids with the same shape as "
            f"loss_mask; got {shape} and {tuple(loss_mask.shape)}"
        )
    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in [0, 1], got {tail_fraction}")
    if len(position_weights) != len(position_boundaries) + 1:
        raise ValueError(
            "position_weights must contain exactly one more value than "
            "position_boundaries"
        )
    if any(boundary < 0 for boundary in position_boundaries) or any(
        right <= left
        for left, right in zip(
            position_boundaries, position_boundaries[1:], strict=False
        )
    ):
        raise ValueError("position_boundaries must be non-negative and increasing")
    if any(weight <= 0 for weight in position_weights):
        raise ValueError("position_weights must all be positive")
    return position_ids


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
    position_ids: torch.Tensor | None = None,
    sampling: str = "uniform",
    tail_fraction: float = 0.5,
    position_boundaries: Sequence[int] = (8192, 16384, 24576),
    position_weights: Sequence[float] = (1.0, 2.0, 6.0, 12.0),
    anchor_candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select anchor positions from valid tokens in a packed sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [1, total_seq_len]
        num_anchors: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded)
        position_ids: Per-document positions. Required by ``long-context-mix``.
        sampling: ``uniform`` (the historical behavior) or ``long-context-mix``.
        tail_fraction: Fraction drawn from the position-weighted distribution.
        position_boundaries: Increasing boundaries for the position buckets.
        position_weights: Positive sampling weights, one per position bucket.

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [num_anchors]
            - anchor_valid: Boolean mask for valid anchors [num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    if anchor_candidate_mask is not None:
        if anchor_candidate_mask.shape != loss_mask.shape:
            raise ValueError(
                "anchor_candidate_mask must have the same shape as loss_mask; "
                f"got {tuple(anchor_candidate_mask.shape)} and {tuple(loss_mask.shape)}"
            )
        valid_mask = anchor_candidate_mask.bool().clone()
    else:
        valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size:] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())

    # Constrain value of k for torch dynamo
    torch._check(k <= valid_indices.numel())  # noqa: SLF001
    torch._check(k >= 0)  # noqa: SLF001

    if sampling == "uniform":
        # Keep the historical path byte-for-byte equivalent: among other things,
        # this preserves fixed-seed behavior for existing training recipes.
        perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
        # Contiguous anchors let flex attention use dense (fast) blocks instead of
        # scattered all-partial (slow) ones; the order never affects the loss.
        anchors[:k] = torch.sort(torch.gather(valid_indices, 0, perm[:k])).values
        anchor_valid[:k] = True

        return anchors, anchor_valid

    if sampling != "long-context-mix":
        raise ValueError(
            "anchor sampling must be 'uniform' or 'long-context-mix', "
            f"got {sampling!r}"
        )
    position_ids = _validate_long_context_sampling(
        loss_mask,
        position_ids,
        tail_fraction,
        position_boundaries,
        position_weights,
    )

    # Draw the uniform portion first, then draw the weighted portion from the
    # remaining candidates. This makes the combined sample duplicate-free and
    # naturally backfills from other buckets when late-context tokens are scarce.
    weighted_count = int(k * tail_fraction)
    uniform_count = k - weighted_count
    perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
    uniform_anchors = torch.gather(valid_indices, 0, perm[:uniform_count])
    remaining_indices = torch.gather(valid_indices, 0, perm[uniform_count:])

    boundaries = torch.tensor(
        position_boundaries, dtype=position_ids.dtype, device=device
    )
    if weighted_count:
        bucket_ids = torch.bucketize(
            position_ids[0, remaining_indices], boundaries, right=True
        )
        bucket_weights = torch.tensor(
            position_weights, dtype=torch.float32, device=device
        )
        candidate_weights = bucket_weights[bucket_ids]
        weighted_choice = torch.multinomial(
            candidate_weights, weighted_count, replacement=False
        )
        weighted_anchors = torch.gather(remaining_indices, 0, weighted_choice)
    else:
        weighted_anchors = remaining_indices[:0]

    selected = torch.cat([uniform_anchors, weighted_anchors])
    # Sorted anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    anchors[:k] = torch.sort(selected).values
    anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]
