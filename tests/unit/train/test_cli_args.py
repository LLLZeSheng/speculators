"""Tests for CLI arguments."""

import argparse

import pytest

from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss, tv_loss_fused_or_eager
from speculators.models.peagle.core import PEagleDraftModel
from speculators.train.config import TrainConfig


def _parse(monkeypatch, extra: list[str]) -> argparse.Namespace:
    cfg = TrainConfig.resolve(["--verifier-name-or-path", "dummy", *extra])
    return argparse.Namespace(**cfg.flatten())


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_decay_gamma(monkeypatch):
    args = _parse(monkeypatch, ["--dflash-decay-gamma", "7.0"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["gamma"] == 7.0
    assert val_kw["gamma"] == 7.0


def test_dflash_decay_gamma_falls_back_when_omitted():
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_long_context_anchor_args_apply_only_to_training(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--anchor-sampling",
            "long-context-mix",
            "--anchor-tail-fraction",
            "0.6",
            "--anchor-position-boundaries",
            "4096",
            "8192",
            "--anchor-position-weights",
            "1",
            "3",
            "9",
        ],
    )
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))

    assert train_kw["anchor_sampling"] == "long-context-mix"
    assert train_kw["anchor_tail_fraction"] == 0.6
    assert train_kw["anchor_position_boundaries"] == [4096, 8192]
    assert train_kw["anchor_position_weights"] == [1.0, 3.0, 9.0]
    assert val_kw["anchor_sampling"] == "uniform"


def test_dflash_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_dspark_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["confidence_head_alpha"] == 1.0
    assert val_kw["confidence_head_alpha"] == 1.0


def test_dspark_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["tv"][0] is tv_loss_fused_or_eager
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_dspark_confidence_head_alpha(monkeypatch):
    args = _parse(monkeypatch, ["--confidence-head-alpha", "0.5"])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["confidence_head_alpha"] == 0.5
    assert val_kw["confidence_head_alpha"] == 0.5


# ---------------------------------------------------------------------------
# Per-speculator-type defaults for draft_arch, norm_before_fc, norm_output
# ---------------------------------------------------------------------------


def test_eagle3_defaults_to_llama_arch(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.draft_arch == "llama"


def test_eagle3_defaults_norm_before_fc_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_before_fc is True


def test_eagle3_defaults_norm_output_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_output is True


def test_dflash_defaults_to_qwen3_arch(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.draft_arch == "qwen3"


def test_dflash_defaults_norm_before_fc_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_before_fc is False


def test_dflash_defaults_norm_output_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_output is False


def test_no_norm_before_fc_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-before-fc"])
    assert args.norm_before_fc is False


def test_no_norm_output_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-output"])
    assert args.norm_output is False


def test_cached_hidden_states_finite_validation_can_be_disabled(monkeypatch):
    args = _parse(monkeypatch, ["--no-validate-cached-hidden-states-finite"])
    assert args.validate_cached_hidden_states_finite is False


def test_generation_model_defaults_to_none(monkeypatch):
    args = _parse(monkeypatch, [])

    assert args.generation_model_name_or_path is None


def test_generation_model_can_differ_from_verifier(monkeypatch):
    args = _parse(
        monkeypatch,
        ["--generation-model-name-or-path", "glm52-w4a8-verifier"],
    )

    assert args.generation_model_name_or_path == "glm52-w4a8-verifier"


def test_online_generation_safety_flags(monkeypatch):
    args = _parse(
        monkeypatch,
        ["--force-generate", "--on-generation-error", "raise"],
    )

    assert args.force_generate is True
    assert args.on_generation_error == "raise"


# ---------------------------------------------------------------------------
# --max-steps
# ---------------------------------------------------------------------------


def test_max_steps_default_is_none(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.max_steps is None


def test_max_steps_explicit(monkeypatch):
    args = _parse(monkeypatch, ["--max-steps", "15"])
    assert args.max_steps == 15


def test_max_steps_rejects_non_positive(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--max-steps", "0"])
