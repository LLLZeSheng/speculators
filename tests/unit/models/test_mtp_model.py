"""Unit tests for MTPDraftModel forward pass."""

import math

import pytest
import torch
from torch import nn

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig

BATCH = 1
SEQ_LEN = 10


# ===== Forward output structure =====


def _counted_metric(metrics, name):
    return metrics[f"{name}_sum"] / metrics[f"{name}_total"]


class _SequentialPredictions(nn.Module):
    def __init__(self, predictions: list[torch.Tensor], vocab_size: int):
        super().__init__()
        self.predictions = predictions
        self.vocab_size = vocab_size
        self.call_index = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pred_ids = self.predictions[self.call_index].to(hidden_states.device)
        self.call_index += 1
        return 10 * nn.functional.one_hot(pred_ids, num_classes=self.vocab_size).to(
            hidden_states.dtype
        )


def test_forward_output_structure(mtp_model, seed):
    """Verify logit shapes, loss, and per-step metrics in a single forward pass."""
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
    with torch.no_grad():
        logits_list, total_loss, metrics = mtp_model(
            input_ids=input_ids, hidden_states=hidden_states
        )

    assert len(logits_list) == num_steps
    expected_len = SEQ_LEN - num_steps - 1
    for step in range(num_steps):
        assert logits_list[step].shape == (BATCH, expected_len, vocab_size)

    assert total_loss.dim() == 0
    assert torch.isfinite(total_loss)
    assert total_loss >= 0

    expected_keys = (
        {
            f"{name}_{suffix}"
            for name in (
                "full_acc",
                "accepted_draft_len",
                "accept_len",
                *(f"position_{k}_acc" for k in range(num_steps)),
                *(f"conditional_position_{k}_acc" for k in range(num_steps)),
            )
            for suffix in ("sum", "total")
        }
        | {f"loss_step_{k}" for k in range(num_steps)}
        | {
            "loss_sum",
            "loss_total",
        }
    )
    assert set(metrics.keys()) == expected_keys
    for key in expected_keys:
        assert math.isfinite(metrics[key])


def test_forward_reports_greedy_prefix_acceptance_metrics(mtp_model, seed):
    """Acceptance length counts consecutive correct recursive MTP predictions."""
    num_steps = mtp_model.config.num_speculative_steps
    assert num_steps == 3
    vocab_size = mtp_model.config.vocab_size
    hidden_size = mtp_model.config.hidden_size
    valid_len = SEQ_LEN - num_steps - 1
    input_ids = torch.arange(SEQ_LEN).unsqueeze(0) % vocab_size
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)

    targets = [
        input_ids[:, step + 2 : step + 2 + valid_len].clone()
        for step in range(num_steps)
    ]
    predictions = [target.clone() for target in targets]
    # Step 0: 6/6 correct. Step 1: 3/6 correct. Step 2: 5/6 correct,
    # but only 2/3 are correct among anchors whose first two tokens matched.
    predictions[1][:, 3:] = (predictions[1][:, 3:] + 1) % vocab_size
    predictions[2][:, 2] = (predictions[2][:, 2] + 1) % vocab_size
    mtp_model.lm_head = _SequentialPredictions(predictions, vocab_size)

    with torch.no_grad():
        _, _, metrics = mtp_model(
            input_ids=input_ids,
            hidden_states=hidden_states,
        )

    expected = {
        "position_0_acc": 1.0,
        "position_1_acc": 0.5,
        "position_2_acc": 5 / 6,
        "conditional_position_0_acc": 1.0,
        "conditional_position_1_acc": 0.5,
        "conditional_position_2_acc": 2 / 3,
        "full_acc": 7 / 9,
        "accepted_draft_len": 11 / 6,
        "accept_len": 17 / 6,
    }
    for name, value in expected.items():
        assert _counted_metric(metrics, name).item() == pytest.approx(value)


# ===== Loss masking =====


class TestLossMasking:
    def test_zero_mask_ignores_all_targets(self, mtp_model, seed):
        """All-zero loss_mask sets every target to -100. Loss returns 0.0
        (not NaN) because the denominator is clamped to min=1."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        loss_mask = torch.zeros(BATCH, SEQ_LEN)
        with torch.no_grad():
            _, total_loss, _ = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
        assert total_loss == 0.0

    def test_partial_mask_changes_loss(self, mtp_model, seed):
        """Masking some positions should change the loss vs no mask."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, loss_no_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states
            )
            mask = torch.ones(BATCH, SEQ_LEN)
            mask[:, -3:] = 0
            _, loss_partial_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=mask
            )
        assert loss_no_mask != loss_partial_mask


# ===== Step weights =====


class TestStepWeights:
    def test_zero_weight_zeroes_step_loss(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, _, metrics = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                step_weights=[1.0, 0.0, 0.0],
            )
        assert metrics["loss_step_0"] > 0
        assert metrics["loss_step_1"] == 0.0
        assert metrics["loss_step_2"] == 0.0


# ===== Short sequence truncation =====


def test_short_sequence_fewer_logits(mtp_model, seed):
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    short_len = 3
    input_ids = torch.randint(0, vocab_size, (BATCH, short_len))
    hidden_states = torch.randn(BATCH, short_len, hidden_size)
    with torch.no_grad():
        logits_list, _, _ = mtp_model(input_ids=input_ids, hidden_states=hidden_states)
    assert len(logits_list) < num_steps
    assert len(logits_list) == 1


def test_glm_moe_dsa_forward_uses_native_sparse_layer(seed, monkeypatch, tmp_path):
    glm_config = pytest.importorskip(
        "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa",
    )
    GlmMoeDsaConfig = glm_config.GlmMoeDsaConfig
    original_empty = torch.empty

    def nan_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.is_floating_point() or tensor.is_complex():
            tensor.fill_(torch.nan)
        return tensor

    # GLM experts and router state are raw Parameters rather than Linear
    # modules. Poison allocations so missing architecture-specific init is
    # deterministic instead of depending on allocator contents.
    monkeypatch.setattr(torch, "empty", nan_empty)
    transformer_config = GlmMoeDsaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=4,
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "full", "shared"],
    )
    config = MTPSpeculatorConfig(
        transformer_layer_config=transformer_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["GlmMoeDsaForCausalLM"],
            ),
        ),
    )
    model = MTPDraftModel(config)
    assert model.config.transformer_layer_config._attn_implementation == "sdpa"
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    hidden_states = torch.randn(1, 6, config.hidden_size)
    with torch.no_grad():
        logits, loss, _ = model(input_ids=input_ids, hidden_states=hidden_states)

    assert len(logits) == 3
    assert logits[0].shape == (1, 2, config.vocab_size)
    assert torch.isfinite(loss)
    assert hasattr(model.mtp_layers[0].mlp, "experts")
    assert torch.isfinite(model.mtp_layers[0].mlp.experts.gate_up_proj).all()
    assert torch.isfinite(model.mtp_layers[0].mlp.experts.down_proj).all()
    assert model.mtp_layers[0].self_attn.indexer is not None
    assert not any(
        parameter.requires_grad
        for parameter in model.mtp_layers[0].self_attn.indexer.parameters()
    )

    model.save_pretrained(tmp_path)
    reloaded = MTPDraftModel.from_pretrained(tmp_path)
    assert not any(
        parameter.requires_grad
        for parameter in reloaded.mtp_layers[0].self_attn.indexer.parameters()
    )
