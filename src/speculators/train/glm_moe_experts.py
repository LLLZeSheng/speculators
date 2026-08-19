"""Memory-bounded GLM packed-expert execution for FSDP training."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import nn

logger = logging.getLogger("speculators")


class GlmMoeExpertChunk(nn.Module):
    """A contiguous subset of GLM routed experts.

    GLM stores all experts in two packed 3-D parameters. Keeping a small
    contiguous expert range in its own module gives FSDP a bounded all-gather
    unit while preserving the original top-k routing calculation.
    """

    def __init__(
        self,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        *,
        expert_start: int,
        act_fn,
        requires_grad: bool,
    ) -> None:
        super().__init__()
        self.expert_start = expert_start
        self.num_experts = gate_up_proj.shape[0]
        # Transformers activation objects are often parameter-free Modules.
        # Keep the callable without registering the same Module under every
        # expert chunk, which would create a shared child in the FSDP tree.
        object.__setattr__(self, "act_fn", act_fn)
        # Each chunk must own storage whose offset starts at zero. A contiguous
        # dim-0 slice is still a view into the original packed parameter and
        # retains both its full backing storage and a growing storage_offset.
        # FSDP2 later calls storage.resize_() while allocating all-gather
        # outputs; keeping those views would make the requested allocation grow
        # with the chunk index (for example, the eighth 384-MiB gate/up slice
        # requests 3 GiB). Clone here so the physical allocation is bounded by
        # the advertised per-chunk FSDP group size.
        self.gate_up_proj = nn.Parameter(
            gate_up_proj.detach().clone(), requires_grad=requires_grad
        )
        self.down_proj = nn.Parameter(
            down_proj.detach().clone(), requires_grad=requires_grad
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        expert_end = self.expert_start + self.num_experts

        # Iterate only over experts owned by this FSDP unit. This is equivalent
        # to the Transformers eager expert implementation, except that global
        # router IDs are translated to this chunk's local parameter indices.
        for global_expert in range(self.expert_start, expert_end):
            token_idx, top_k_pos = torch.where(top_k_index == global_expert)
            if token_idx.numel() == 0:
                continue
            local_expert = global_expert - self.expert_start
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[local_expert]
            ).chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = nn.functional.linear(
                current_hidden, self.down_proj[local_expert]
            )
            current_hidden = current_hidden * top_k_weights[
                token_idx, top_k_pos, None
            ]
            output.index_add_(0, token_idx, current_hidden.to(dtype=output.dtype))
        return output


class ChunkedGlmMoeExperts(nn.Module):
    """Drop-in replacement that evaluates packed GLM experts in small units."""

    def __init__(self, original: nn.Module, experts_per_unit: int) -> None:
        super().__init__()
        gate_up_proj = getattr(original, "gate_up_proj")
        down_proj = getattr(original, "down_proj")
        if not isinstance(gate_up_proj, nn.Parameter) or not isinstance(
            down_proj, nn.Parameter
        ):
            raise TypeError("GLM packed expert weights must be Parameters")
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("GLM packed expert weights must be 3-D")
        if gate_up_proj.shape[0] != down_proj.shape[0]:
            raise ValueError("GLM packed expert weights disagree on expert count")
        if experts_per_unit <= 0:
            raise ValueError("experts_per_unit must be positive")

        self.num_experts = gate_up_proj.shape[0]
        self.hidden_dim = getattr(original, "hidden_dim", gate_up_proj.shape[-1])
        self.intermediate_dim = getattr(
            original, "intermediate_dim", down_proj.shape[-1]
        )
        act_fn = getattr(original, "act_fn")
        requires_grad = gate_up_proj.requires_grad or down_proj.requires_grad
        self.chunks = nn.ModuleList()
        for start in range(0, self.num_experts, experts_per_unit):
            end = min(start + experts_per_unit, self.num_experts)
            self.chunks.append(
                GlmMoeExpertChunk(
                    gate_up_proj[start:end],
                    down_proj[start:end],
                    expert_start=start,
                    act_fn=act_fn,
                    requires_grad=requires_grad,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for chunk in self.chunks:
            output = output + chunk(hidden_states, top_k_index, top_k_weights)
        return output


def _packed_glm_experts(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        gate_up_proj = getattr(module, "gate_up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if (
            isinstance(gate_up_proj, nn.Parameter)
            and isinstance(down_proj, nn.Parameter)
            and gate_up_proj.ndim == 3
            and down_proj.ndim == 3
            and gate_up_proj.shape[0] == down_proj.shape[0]
        ):
            yield name, module


def chunk_glm_moe_experts(model: nn.Module, experts_per_unit: int) -> int:
    """Replace packed GLM expert modules with FSDP-friendly expert chunks."""

    replacements = list(_packed_glm_experts(model))
    for name, original in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = ChunkedGlmMoeExperts(original, experts_per_unit)
        setattr(parent, child_name, replacement)
        largest_numel = max(
            sum(parameter.numel() for parameter in chunk.parameters())
            for chunk in replacement.chunks
        )
        logger.info(
            "FSDP GLM expert chunking: module=%s experts=%d units=%d "
            "experts_per_unit=%d largest_unit_parameters=%d",
            name,
            replacement.num_experts,
            len(replacement.chunks),
            experts_per_unit,
            largest_numel,
        )
    return len(replacements)
