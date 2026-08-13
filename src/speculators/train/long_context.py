"""Lightweight, on-the-fly long-context training augmentation.

The index contains only row references and boundaries. Hidden states remain in
their original cache and are concatenated by the DataLoader for the current
sample only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path


REAL = 0
SYNTHETIC_PREFIX = 1
DISTANCE_STRETCH = 2


@dataclass(frozen=True)
class LongContextRecord:
    kind: int
    donor1: int
    donor2: int
    prefix_len: int
    response_start: int
    response_end: int
    target_anchor_position: int


class LongContextIndex:
    """Memory-mapped-style compact augmentation instructions per dataset row."""

    _FIELDS = (
        "kind",
        "donor1",
        "donor2",
        "prefix_len",
        "response_start",
        "response_end",
        "target_anchor_position",
    )

    def __init__(self, path: str | Path, expected_rows: int):
        payload = np.load(path, allow_pickle=False)
        missing = [name for name in self._FIELDS if name not in payload]
        if missing:
            raise ValueError(f"Long-context index is missing fields: {missing}")
        self.arrays = {name: payload[name] for name in self._FIELDS}
        sizes = {len(value) for value in self.arrays.values()}
        if sizes != {expected_rows}:
            raise ValueError(
                "Long-context index must have one record per full dataset row; "
                f"expected {expected_rows}, got field sizes {sorted(sizes)}"
            )

    def __getitem__(self, index: int) -> LongContextRecord:
        return LongContextRecord(
            **{name: int(values[index]) for name, values in self.arrays.items()}
        )


def build_anchor_candidate_mask(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    excluded_token_ids: tuple[int, ...] = (),
    allowed_start: int = 0,
    allowed_end: int | None = None,
) -> torch.Tensor:
    """Return starts whose complete draft block is safe to supervise.

    This is deliberately separate from ``loss_mask``: an individually valid
    token is not necessarily a valid start for an eight-token draft block.
    """
    if input_ids.ndim != 1 or loss_mask.ndim != 1 or input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids and loss_mask must be equal-length 1-D tensors")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    seq_len = input_ids.numel()
    allowed_end = seq_len if allowed_end is None else min(allowed_end, seq_len)
    token_ok = loss_mask.bool().clone()
    for token_id in excluded_token_ids:
        token_ok &= input_ids != token_id

    candidates = torch.zeros(seq_len, dtype=torch.bool)
    last_start = allowed_end - block_size
    if last_start < allowed_start:
        return candidates
    # A start is safe only if every target slot remains supervised and avoids
    # EOS/padding/boundary tokens. conv1d computes the rolling AND efficiently.
    rolling = torch.nn.functional.conv1d(
        token_ok.float().view(1, 1, -1),
        torch.ones(1, 1, block_size),
    ).view(-1)
    candidates[allowed_start : last_start + 1] = (
        rolling[allowed_start : last_start + 1] == block_size
    )
    return candidates


def stretched_position_ids(
    seq_len: int,
    response_start: int,
    target_anchor_position: int,
    near_window: int = 8192,
) -> torch.Tensor:
    """Stretch remote relative distances while preserving the local tail exactly."""
    if not (0 <= response_start < seq_len):
        raise ValueError("response_start must be inside the sequence")
    if target_anchor_position < response_start:
        raise ValueError("target anchor position cannot precede response_start")
    local_start = max(0, response_start - near_window)
    desired_local_start = max(
        0, target_anchor_position - (response_start - local_start)
    )
    positions = torch.arange(seq_len, dtype=torch.long)
    if local_start:
        # Monotone interpolation of the remote prefix. Endpoint is strictly
        # before the unchanged local region.
        positions[:local_start] = torch.linspace(
            0, max(0, desired_local_start - 1), local_start
        ).round().long()
    positions[local_start:] += desired_local_start - local_start
    return positions


def concatenate_samples(
    parts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate tensor fields without changing hidden-state/token alignment."""
    keys = ("hidden_states", "input_ids", "verifier_last_hidden_states", "loss_mask")
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
