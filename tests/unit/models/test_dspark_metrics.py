"""Unit tests for the DSpark loss and metrics."""

import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; with sample_from_anchor=False, position 0 is the anchor
        # (masked) and position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the anchor token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2

    def test_perfect_draft_anchor_sampled_includes_slot0(self):
        # sample_from_anchor=True (default): slot 0 is the first real prediction,
        # so every position is supervised and accept_len counts all draft slots.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        assert torch.isfinite(loss)
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # Two draft slots per block accepted w.p. ~1, plus the anchor token -> ~3.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 3.0) < 1e-2

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        # sample_from_anchor=False: position 0 is the anchor (masked).
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}'),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}'),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())

    def test_context_bucket_acceptance_and_anchor_fractions(self):
        # Four two-token blocks, one anchor at the start of each 8K context bucket.
        target_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        draft_ids = torch.tensor(
            [
                [
                    0,
                    1,  # 0-8K: both accepted -> length 3
                    8,
                    3,  # 8-16K: first rejected -> length 1
                    4,
                    8,  # 16-24K: first accepted -> length 2
                    6,
                    7,  # 24-32K: both accepted -> length 3
                ]
            ]
        )
        logits = _ids_to_logits(draft_ids, 16)
        targets = _ids_to_logits(target_ids, 16)
        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            torch.ones(1, 8),
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            anchor_context_positions=torch.tensor([0, 8192, 16384, 24576]),
        )

        expected_lengths = {
            "0_8k": 3.0,
            "8_16k": 1.0,
            "16_24k": 2.0,
            "24_32k": 3.0,
        }
        for label, expected in expected_lengths.items():
            accept_len = (
                metrics[f"accept_len_ctx_{label}_sum"]
                / metrics[f"accept_len_ctx_{label}_total"]
            )
            fraction = (
                metrics[f"anchor_fraction_ctx_{label}_sum"]
                / metrics[f"anchor_fraction_ctx_{label}_total"]
            )
            assert abs(float(accept_len) - expected) < 1e-2
            assert abs(float(fraction) - 0.25) < 1e-6

        fraction_totals = [
            metrics[f"anchor_fraction_ctx_{label}_total"]
            for label in expected_lengths
        ]
        assert len({id(total) for total in fraction_totals}) == len(fraction_totals)

    def test_context_metrics_exclude_padded_anchor_blocks(self):
        ids = torch.tensor([[0, 1, 0, 0]])
        logits = _ids_to_logits(ids, 8)
        _, metrics = compute_metrics(
            logits,
            logits.clone(),
            None,
            torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            anchor_context_positions=torch.tensor([100, 20000]),
        )

        assert float(metrics["anchor_fraction_ctx_0_8k_sum"]) == 1.0
        assert float(metrics["anchor_fraction_ctx_0_8k_total"]) == 1.0
        assert float(metrics["anchor_fraction_ctx_16_24k_sum"]) == 0.0
        assert float(metrics["accept_len_ctx_16_24k_total"]) == 0.0
