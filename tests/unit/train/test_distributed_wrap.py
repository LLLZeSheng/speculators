"""Tests for memory-bounded FSDP wrapping policy."""

from unittest.mock import patch

import torch
from torch import nn

from speculators.train.distributed import apply_fully_sharded


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.large_projection = nn.Linear(16, 16, bias=False)
        self.small_projection = nn.Linear(2, 2, bias=False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([_Layer()])
        self.lm_head = nn.Linear(16, 32, bias=False)


def test_memory_efficient_wraps_large_children_before_layer_and_root():
    model = _Model()
    calls: list[nn.Module] = []

    with patch(
        "speculators.train.distributed.fully_shard",
        side_effect=lambda module, **_kwargs: calls.append(module),
    ):
        apply_fully_sharded(
            model,
            param_dtype=torch.bfloat16,
            wrap_policy="memory_efficient",
            min_numel=128,
        )

    assert model.layers[0].large_projection in calls
    assert model.embed_tokens in calls
    assert model.lm_head not in calls
    assert model.layers[0].small_projection not in calls
    assert calls.index(model.layers[0].large_projection) < calls.index(model.layers[0])
    assert calls.index(model.layers[0]) < calls.index(model)


def test_layer_wrap_policy_preserves_original_granularity():
    model = _Model()
    calls: list[nn.Module] = []

    with patch(
        "speculators.train.distributed.fully_shard",
        side_effect=lambda module, **_kwargs: calls.append(module),
    ):
        apply_fully_sharded(model, wrap_policy="layer")

    assert calls == [model.layers[0], model]
