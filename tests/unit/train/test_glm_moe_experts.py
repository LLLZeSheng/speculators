"""Tests for memory-bounded packed GLM expert execution."""

import copy

import torch
from torch import nn

from speculators.train.glm_moe_experts import ChunkedGlmMoeExperts


class _PackedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim)
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert)
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current, self.gate_up_proj[expert]
            ).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = nn.functional.linear(current, self.down_proj[expert])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, current)
        return output


def test_chunked_experts_match_packed_forward_and_backward():
    torch.manual_seed(7)
    packed = _PackedExperts(num_experts=4, hidden_dim=6, intermediate_dim=3)
    chunk_source = copy.deepcopy(packed)
    chunked = ChunkedGlmMoeExperts(chunk_source, experts_per_unit=2)

    top_k_index = torch.tensor([[0, 3], [1, 2], [3, 1], [2, 0]])
    top_k_weights = torch.rand(4, 2)
    packed_input = torch.randn(4, 6, requires_grad=True)
    chunked_input = packed_input.detach().clone().requires_grad_(True)

    packed_output = packed(packed_input, top_k_index, top_k_weights)
    chunked_output = chunked(chunked_input, top_k_index, top_k_weights)
    torch.testing.assert_close(chunked_output, packed_output)

    packed_output.square().sum().backward()
    chunked_output.square().sum().backward()
    torch.testing.assert_close(chunked_input.grad, packed_input.grad)
    torch.testing.assert_close(
        torch.cat([chunk.gate_up_proj.grad for chunk in chunked.chunks]),
        packed.gate_up_proj.grad,
    )
    torch.testing.assert_close(
        torch.cat([chunk.down_proj.grad for chunk in chunked.chunks]),
        packed.down_proj.grad,
    )


def test_chunked_experts_backward_with_one_expert_per_unit():
    """Fine-grained units must support repeated in-place accumulation."""
    torch.manual_seed(11)
    packed = _PackedExperts(num_experts=8, hidden_dim=6, intermediate_dim=3)
    chunked = ChunkedGlmMoeExperts(copy.deepcopy(packed), experts_per_unit=1)

    top_k_index = torch.tensor([[0, 7], [1, 6], [2, 5], [3, 4]])
    top_k_weights = torch.rand(4, 2)
    packed_input = torch.randn(4, 6, requires_grad=True)
    chunked_input = packed_input.detach().clone().requires_grad_(True)

    packed_output = packed(packed_input, top_k_index, top_k_weights)
    chunked_output = chunked(chunked_input, top_k_index, top_k_weights)
    torch.testing.assert_close(chunked_output, packed_output)

    packed_output.sum().backward()
    chunked_output.sum().backward()
    torch.testing.assert_close(chunked_input.grad, packed_input.grad)
    torch.testing.assert_close(
        torch.cat([chunk.gate_up_proj.grad for chunk in chunked.chunks]),
        packed.gate_up_proj.grad,
    )
    torch.testing.assert_close(
        torch.cat([chunk.down_proj.grad for chunk in chunked.chunks]),
        packed.down_proj.grad,
    )


def test_chunked_experts_bound_each_parameter_group():
    packed = _PackedExperts(num_experts=7, hidden_dim=6, intermediate_dim=3)
    chunked = ChunkedGlmMoeExperts(packed, experts_per_unit=3)

    assert [chunk.num_experts for chunk in chunked.chunks] == [3, 3, 1]
    assert max(
        sum(parameter.numel() for parameter in chunk.parameters())
        for chunk in chunked.chunks
    ) == 3 * (2 * 3 * 6 + 6 * 3)


def test_chunked_experts_use_independent_zero_offset_storage():
    packed = _PackedExperts(num_experts=7, hidden_dim=6, intermediate_dim=3)
    chunked = ChunkedGlmMoeExperts(packed, experts_per_unit=3)

    gate_storages = []
    down_storages = []
    for chunk in chunked.chunks:
        assert chunk.gate_up_proj.storage_offset() == 0
        assert chunk.down_proj.storage_offset() == 0
        assert chunk.gate_up_proj.untyped_storage().nbytes() == (
            chunk.gate_up_proj.numel() * chunk.gate_up_proj.element_size()
        )
        assert chunk.down_proj.untyped_storage().nbytes() == (
            chunk.down_proj.numel() * chunk.down_proj.element_size()
        )
        gate_storages.append(chunk.gate_up_proj.untyped_storage().data_ptr())
        down_storages.append(chunk.down_proj.untyped_storage().data_ptr())

    assert len(set(gate_storages)) == len(gate_storages)
    assert len(set(down_storages)) == len(down_storages)
