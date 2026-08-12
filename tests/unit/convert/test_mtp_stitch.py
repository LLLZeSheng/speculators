"""Unit tests for stitching Speculators MTP weights into GLM checkpoints."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.stitch_mtp import _remap_key, stitch


def test_remap_key_uses_glm_extra_layer_layout():
    assert (
        _remap_key("mtp_layers.0.input_proj.weight", "glm_moe_dsa", 78)
        == "model.layers.78.eh_proj.weight"
    )
    assert (
        _remap_key("mtp_layers.0.mlp.gate.weight", "glm_moe_dsa", 78)
        == "model.layers.78.mlp.gate.weight"
    )


def test_stitch_replaces_glm_extra_layer_weights(tmp_path):
    verifier = tmp_path / "verifier"
    finetuned = tmp_path / "finetuned"
    output = tmp_path / "stitched"
    verifier.mkdir()
    finetuned.mkdir()

    (verifier / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
            }
        )
    )
    save_file(
        {
            "model.layers.2.eh_proj.weight": torch.zeros(2, 4),
            "model.layers.2.self_attn.o_proj.weight": torch.zeros(2, 2),
        },
        verifier / "model.safetensors",
    )
    save_file(
        {
            "mtp_layers.0.input_proj.weight": torch.ones(2, 4),
            "mtp_layers.0.self_attn.o_proj.weight": torch.full((2, 2), 2.0),
        },
        finetuned / "model.safetensors",
    )

    stitch(finetuned, verifier, output)

    with safe_open(output / "model.safetensors", framework="pt") as handle:
        torch.testing.assert_close(
            handle.get_tensor("model.layers.2.eh_proj.weight"), torch.ones(2, 4)
        )
        torch.testing.assert_close(
            handle.get_tensor("model.layers.2.self_attn.o_proj.weight"),
            torch.full((2, 2), 2.0),
        )
        assert "mtp_layers.0.input_proj.weight" not in handle.keys()  # noqa: SIM118
