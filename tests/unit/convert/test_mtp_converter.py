"""Unit tests for MTPConverter key remapping, MoE expert fusion, and config building."""

from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from speculators.convert.mtp import converter as mtp_converter
from speculators.convert.mtp.converter import MTPConverter
from tests.conftest import requires_transformers_version


class TestRemapKey:
    """Test _remap_key static method across all remap table entries."""

    @pytest.mark.parametrize(
        ("input_key", "expected"),
        [
            # Exact remaps
            ("mtp.fc.weight", "mtp_layers.0.input_proj.weight"),
            ("mtp.norm.weight", "mtp_layers.0.final_layernorm.weight"),
            # Prefix remaps
            (
                "mtp.pre_fc_norm_hidden.weight",
                "mtp_layers.0.hidden_layernorm.weight",
            ),
            (
                "mtp.pre_fc_norm_embedding.weight",
                "mtp_layers.0.token_layernorm.weight",
            ),
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                "mtp_layers.0.self_attn.q_proj.weight",
            ),
            # Unknown key passthrough
            ("some.random.key", "some.random.key"),
        ],
        ids=[
            "exact_fc",
            "exact_norm",
            "prefix_hidden_layernorm",
            "prefix_token_layernorm",
            "prefix_mtp_layer",
            "unknown_passthrough",
        ],
    )
    def test_remap_key(self, input_key, expected):
        assert MTPConverter._remap_key(input_key) == expected

    @pytest.mark.parametrize(
        ("input_key", "expected"),
        [
            (
                "model.layers.78.eh_proj.weight",
                "mtp_layers.0.input_proj.weight",
            ),
            (
                "model.layers.78.hnorm.weight",
                "mtp_layers.0.hidden_layernorm.weight",
            ),
            (
                "model.layers.78.enorm.weight",
                "mtp_layers.0.token_layernorm.weight",
            ),
            (
                "model.layers.78.shared_head.norm.weight",
                "mtp_layers.0.final_layernorm.weight",
            ),
            (
                "model.layers.78.self_attn.q_a_proj.weight",
                "mtp_layers.0.self_attn.q_a_proj.weight",
            ),
            (
                "model.layers.78.mlp.experts.3.gate_proj.weight",
                "mtp_layers.0.mlp.experts.3.gate_proj.weight",
            ),
        ],
    )
    def test_remap_glm_extra_layer_key(self, input_key, expected):
        assert (
            MTPConverter._remap_key(input_key, source_prefix="model.layers.78.")
            == expected
        )

    @pytest.mark.parametrize(
        ("input_key", "expected"),
        [
            (
                "mtp_layers.0.input_proj.weight",
                "model.layers.78.eh_proj.weight",
            ),
            (
                "mtp_layers.0.final_layernorm.weight",
                "model.layers.78.shared_head.norm.weight",
            ),
            (
                "mtp_layers.0.self_attn.o_proj.weight",
                "model.layers.78.self_attn.o_proj.weight",
            ),
        ],
    )
    def test_reverse_remap_glm_key(self, input_key, expected):
        assert (
            mtp_converter.remap_mtp_key_to_native(input_key, "glm_moe_dsa", 78)
            == expected
        )


class TestResolveMtpPrefix:
    def test_glm_uses_extra_layer_after_verifier_layers(self):
        keys = [
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.77.self_attn.q_a_proj.weight",
            "model.layers.78.eh_proj.weight",
        ]
        config = {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
        }

        assert MTPConverter._resolve_mtp_prefix(keys, config) == "model.layers.78."

    def test_glm_rejects_missing_extra_layer(self):
        keys = ["model.layers.77.self_attn.q_a_proj.weight"]
        config = {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
        }

        with pytest.raises(ValueError, match="model.layers.78"):
            MTPConverter._resolve_mtp_prefix(keys, config)


class TestNormalizeConfigFromWeights:
    def test_repairs_glm_rope_dimension_from_kv_projection(self):
        config = {
            "model_type": "glm_moe_dsa",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 192,
        }
        weights = {
            "mtp_layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.empty(
                576, 6144
            )
        }

        normalized = MTPConverter._normalize_config_from_weights(config, weights)

        assert normalized["qk_rope_head_dim"] == 64
        assert config["qk_rope_head_dim"] == 192

    def test_leaves_non_glm_config_unchanged(self):
        config = {"model_type": "qwen3", "qk_rope_head_dim": 192}

        normalized = MTPConverter._normalize_config_from_weights(config, {})

        assert normalized == config


class TestModelSlimExtraction:
    def test_extracts_native_mtp_from_auxiliary_file(self, tmp_path):
        native_key = "model.layers.78.eh_proj.weight"
        tensor = torch.randn(4, 8, dtype=torch.bfloat16)
        save_file({native_key: tensor}, tmp_path / "mtp.safetensors")

        converter = MTPConverter()
        weights = converter._extract_weights(
            tmp_path, [native_key], "model.layers.78."
        )

        torch.testing.assert_close(weights["mtp_layers.0.input_proj.weight"], tensor)

    def test_rejects_quantized_trainable_mtp_weights(self):
        with pytest.raises(ValueError, match="must be floating point"):
            MTPConverter._validate_float_weights(
                {"mtp_layers.0.input_proj.weight": torch.ones(4, 8, dtype=torch.int8)}
            )


class TestFuseMoeExperts:
    """Test _fuse_moe_experts static method."""

    @pytest.fixture
    def two_expert_weights(self, seed):
        """Two experts with gate_proj, up_proj, down_proj (4x4 matrices)."""
        return {
            "layer.experts.0.gate_proj.weight": torch.randn(4, 4),
            "layer.experts.0.up_proj.weight": torch.randn(4, 4),
            "layer.experts.0.down_proj.weight": torch.randn(4, 4),
            "layer.experts.1.gate_proj.weight": torch.randn(4, 4),
            "layer.experts.1.up_proj.weight": torch.randn(4, 4),
            "layer.experts.1.down_proj.weight": torch.randn(4, 4),
        }

    def test_happy_path_shapes(self, two_expert_weights):
        result = MTPConverter._fuse_moe_experts(two_expert_weights)

        assert "layer.experts.gate_up_proj" in result
        assert "layer.experts.down_proj" in result
        assert result["layer.experts.gate_up_proj"].shape == (2, 8, 4)
        assert result["layer.experts.down_proj"].shape == (2, 4, 4)

    def test_happy_path_values(self, two_expert_weights):
        result = MTPConverter._fuse_moe_experts(two_expert_weights)

        gate_0 = two_expert_weights["layer.experts.0.gate_proj.weight"]
        up_0 = two_expert_weights["layer.experts.0.up_proj.weight"]
        expected_gate_up_0 = torch.cat([gate_0, up_0], dim=0)
        torch.testing.assert_close(
            result["layer.experts.gate_up_proj"][0], expected_gate_up_0
        )

        down_1 = two_expert_weights["layer.experts.1.down_proj.weight"]
        torch.testing.assert_close(result["layer.experts.down_proj"][1], down_1)

    def test_non_contiguous_indices(self, seed):
        weights = {
            "layer.experts.0.gate_proj.weight": torch.randn(4, 4),
            "layer.experts.0.up_proj.weight": torch.randn(4, 4),
            "layer.experts.0.down_proj.weight": torch.randn(4, 4),
            "layer.experts.2.gate_proj.weight": torch.randn(4, 4),
            "layer.experts.2.up_proj.weight": torch.randn(4, 4),
            "layer.experts.2.down_proj.weight": torch.randn(4, 4),
        }

        with pytest.raises(ValueError, match="Non-contiguous expert indices"):
            MTPConverter._fuse_moe_experts(weights)

    def test_missing_projection(self, seed):
        weights = {
            "layer.experts.0.gate_proj.weight": torch.randn(4, 4),
            "layer.experts.0.up_proj.weight": torch.randn(4, 4),
        }

        with pytest.raises(ValueError, match="missing down_proj weight"):
            MTPConverter._fuse_moe_experts(weights)

    def test_no_experts_passthrough(self):
        weights = {
            "some.weight": torch.randn(4, 4),
            "another.weight": torch.randn(4, 4),
        }
        original_keys = set(weights.keys())

        result = MTPConverter._fuse_moe_experts(weights)

        assert result is weights
        assert set(result.keys()) == original_keys


class TestBuildConfig:
    """Test _build_config dimension validation."""

    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_hidden_size_mismatch_raises(self, mock_get_config):
        mock_get_config.return_value = (
            {"hidden_size": 4096, "architectures": ["Qwen3ForCausalLM"]},
            None,
        )
        source_config = {"hidden_size": 2048}
        converter = MTPConverter()

        with pytest.raises(ValueError, match="Architecture mismatch"):
            converter._build_config(
                source_config, "some/model", num_speculative_steps=3
            )

    @requires_transformers_version("5.2.0")
    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_matching_hidden_size_succeeds(self, mock_get_config):
        mock_get_config.return_value = (
            {
                "hidden_size": 1024,
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5_text",
                "vocab_size": 248320,
                "intermediate_size": 3584,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 24,
            },
            None,
        )
        source_config = {
            "hidden_size": 1024,
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "intermediate_size": 3584,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_hidden_layers": 24,
        }
        converter = MTPConverter()

        config = converter._build_config(
            source_config, "Qwen/Qwen3.5-0.8B", num_speculative_steps=3
        )

        assert config.speculators_config.algorithm == "mtp"
        assert config.speculators_config.verifier.name_or_path == "Qwen/Qwen3.5-0.8B"
        assert config.speculators_config.default_proposal_method == "greedy"


class TestValidateLoadResult:
    def test_rejects_missing_mtp_weight(self):
        with pytest.raises(ValueError, match="Critical MTP layer weights missing"):
            MTPConverter.validate_load_result(
                ["mtp_layers.0.self_attn.indexer.wk.weight"], []
            )

    def test_rejects_unexpected_weight(self):
        with pytest.raises(ValueError, match="Unexpected keys"):
            MTPConverter.validate_load_result([], ["mtp_layers.0.renamed.weight"])

    def test_allows_runtime_verifier_weights(self):
        MTPConverter.validate_load_result(
            ["embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight"],
            [],
        )
