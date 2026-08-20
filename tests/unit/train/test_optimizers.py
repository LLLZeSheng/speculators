from types import SimpleNamespace

import pytest
from torch import nn

from speculators.train.optimizers import build_optimizers


def _adamw_config():
    return SimpleNamespace(optimizer="adamw", lr=1e-3, weight_decay=0.01)


def test_adamw_excludes_frozen_parameters():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[1].weight.requires_grad_(False)
    model[1].bias.requires_grad_(False)

    (optimizer,) = build_optimizers(model, _adamw_config())
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_adamw_rejects_model_without_trainable_parameters():
    model = nn.Linear(4, 4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="No trainable parameters"):
        build_optimizers(model, _adamw_config())
