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
        self.lm_head.weight.requires_grad_(False)


def test_memory_efficient_wraps_large_children_before_layer_and_root():
    model = _Model()
    calls: list[tuple[nn.Module, dict]] = []

    with patch(
        "speculators.train.distributed.fully_shard",
        side_effect=lambda module, **kwargs: calls.append((module, kwargs)),
    ):
        apply_fully_sharded(
            model,
            param_dtype=torch.bfloat16,
            wrap_policy="memory_efficient",
            min_numel=128,
        )

    modules = [module for module, _kwargs in calls]
    assert model.layers[0].large_projection in modules
    assert model.embed_tokens in modules
    assert model.lm_head in modules
    assert model.layers[0].small_projection not in modules
    head_kwargs = next(kwargs for module, kwargs in calls if module is model.lm_head)
    assert head_kwargs["reshard_after_forward"] is False
    assert modules.index(model.layers[0].large_projection) < modules.index(
        model.layers[0]
    )
    assert modules.index(model.layers[0]) < modules.index(model)


def test_layer_wrap_policy_preserves_original_granularity():
    model = _Model()
    calls: list[tuple[nn.Module, dict]] = []

    with patch(
        "speculators.train.distributed.fully_shard",
        side_effect=lambda module, **kwargs: calls.append((module, kwargs)),
    ):
        apply_fully_sharded(model, wrap_policy="layer")

    modules = [module for module, _kwargs in calls]
    assert modules == [model.lm_head, model.layers[0], model]
    assert calls[0][1]["reshard_after_forward"] is False
