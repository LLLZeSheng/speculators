"""Single source of truth for distributed training topology.

Stores local_rank, SP/DP sizes and ranks, and process groups.
All other modules should import getters from here rather than
maintaining their own distributed state.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

logger = logging.getLogger("speculators")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_local_rank: int = 0
_rank: int = 0
_world_size: int = 1
_is_distributed: bool = False

_sp_size: int = 1
_sp_rank: int = 0
_dp_size: int = 1
_dp_rank: int = 0

_sp_group: ProcessGroup | None = None
_dp_group: ProcessGroup | None = None


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------


def get_local_rank() -> int:
    return _local_rank


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def is_distributed() -> bool:
    return _is_distributed


def get_sp_group() -> ProcessGroup | None:
    return _sp_group


def get_dp_group() -> ProcessGroup | None:
    return _dp_group


def get_sp_size() -> int:
    return _sp_size


def get_sp_rank() -> int:
    return _sp_rank


def get_dp_size() -> int:
    return _dp_size


def get_dp_rank() -> int:
    return _dp_rank


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _init_sp_process_groups(rank: int, world_size: int, sp_size: int) -> None:
    """Initialize sequence-parallel and data-parallel process groups.

    SP groups use contiguous ranks (e.g. sp_size=2, world_size=4: {0,1}, {2,3}).
    DP groups use strided ranks (e.g. sp_size=2, world_size=4: {0,2}, {1,3}).
    """
    global _sp_group, _dp_group, _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603

    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    sp_group = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        pg = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            sp_group = pg

    dp_group = None
    for i in range(sp_size):
        dp_ranks = list(range(i, world_size, sp_size))
        pg = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = pg

    if sp_group is None or dp_group is None:
        raise RuntimeError("Failed to initialize SP/DP process groups")

    _sp_group = sp_group
    _dp_group = dp_group
    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _dp_size = dp_size
    _dp_rank = rank // sp_size


def maybe_setup_distributed(sp_size: int = 1) -> None:
    """Set up distributed training if launched with ``torchrun``.

    Always populates the module-level topology state so that callers
    can use the getter functions regardless of whether SP is enabled.
    Process groups are always created when distributed — with
    ``sp_size == 1`` the DP group spans all ranks and each SP group
    contains a single rank.
    """
    global _local_rank, _rank, _is_distributed, _world_size  # noqa: PLW0603

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "LOCAL_RANK" in os.environ

    _local_rank = local_rank
    _is_distributed = distributed

    if not distributed:
        return

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    _rank = dist.get_rank()
    _world_size = dist.get_world_size()

    _init_sp_process_groups(_rank, _world_size, sp_size)

    logger.info(
        f"Started distributed with local_rank={local_rank}, "
        f"dp_size={_dp_size}, sp_size={_sp_size}",
        extra={"override_rank0_filter": True},
    )


def maybe_destroy_distributed() -> None:
    """Destroy the distributed process group if using distributed training."""
    global _is_distributed, _local_rank, _rank, _world_size  # noqa: PLW0603
    global _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603
    global _sp_group, _dp_group  # noqa: PLW0603

    if not _is_distributed:
        return

    dist.destroy_process_group()
    logger.info(
        "Destroyed distributed process group",
        extra={"override_rank0_filter": True},
    )

    _is_distributed = False
    _local_rank = 0
    _rank = 0
    _world_size = 1
    _sp_size = 1
    _sp_rank = 0
    _dp_size = 1
    _dp_rank = 0
    _sp_group = None
    _dp_group = None


def apply_fully_sharded(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    wrap_policy: Literal["layer", "memory_efficient"] = "layer",
    min_numel: int = 8_000_000,
):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )

    if wrap_policy not in {"layer", "memory_efficient"}:
        raise ValueError(f"Unsupported FSDP wrap policy: {wrap_policy}")
    if min_numel <= 0:
        raise ValueError(f"min_numel must be positive, got {min_numel}")

    wrapped: set[int] = set()

    def shard(
        module: torch.nn.Module, *, reshard_after_forward: bool = True
    ) -> None:
        module_id = id(module)
        if module_id in wrapped:
            return
        fully_shard(
            module,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        wrapped.add(module_id)

    # MTP computes its frozen vocabulary projection inside a non-reentrant
    # activation checkpoint so that full-vocabulary logits are discarded per
    # sequence chunk.  The checkpoint body is called again during backward. If
    # lm_head belongs to the root group (or to a normal resharding child group),
    # its weight may already be a sharded DTensor at that point while the saved
    # hidden-state input is an ordinary Tensor.  F.linear rejects that mixture.
    # Give the head its own group and retain its gathered ordinary Tensor from
    # the first chunk through backward recomputation. It is frozen, so retaining
    # it does not retain gradients or optimizer state.
    lm_head = getattr(model, "lm_head", None)
    if (
        isinstance(lm_head, torch.nn.Module)
        and hasattr(lm_head, "weight")
        and not lm_head.weight.requires_grad
    ):
        logger.info(
            "FSDP checkpoint-safe wrap: module=lm_head "
            "reshard_after_forward=false parameters=%d",
            sum(parameter.numel() for parameter in lm_head.parameters()),
        )
        shard(lm_head, reshard_after_forward=False)

    if wrap_policy == "memory_efficient":
        # FSDP all-gathers one wrapping unit at a time. GLM-MoE layers contain
        # several multi-GiB projections; wrapping only the decoder layer makes
        # all of them resident concurrently. Wrap large parameter-owning
        # submodules bottom-up so attention, routed experts, shared experts,
        # and embeddings can reshard independently.
        #
        # lm_head was assigned above to a non-resharding child group, so it is
        # skipped here and excluded automatically from the root group.
        layers = set(map(id, model.layers))  # type: ignore[union-attr]
        candidates: list[tuple[str, torch.nn.Module, int]] = []
        for name, module in model.named_modules():
            if module is model or id(module) in layers:
                continue
            if name == "lm_head" or name.endswith(".lm_head"):
                continue
            direct_numel = sum(
                parameter.numel() for parameter in module.parameters(recurse=False)
            )
            if direct_numel >= min_numel:
                candidates.append((name, module, direct_numel))

        # Descendants must be wrapped before their ancestors.
        candidates.sort(key=lambda item: item[0].count("."), reverse=True)
        for name, module, direct_numel in candidates:
            logger.info(
                "FSDP memory-efficient wrap: module=%s parameters=%d",
                name,
                direct_numel,
            )
            shard(module)

    for layer in model.layers:  # type: ignore[union-attr]
        shard(layer)

    shard(model)

    return model
