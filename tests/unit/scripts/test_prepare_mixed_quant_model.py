import json
from pathlib import Path

import pytest

from scripts.prepare_mixed_quant_model import (
    QuantConfigError,
    prepare_runtime_model,
    split_mixed_num_bits,
)


def _mixed_config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": {
                        "dynamic": True,
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                    },
                    "output_activations": None,
                    "weights": {
                        "dynamic": False,
                        "group_size": None,
                        "num_bits": {
                            "self_attn.q_a_proj": 8,
                            "mlp.shared_experts": 8,
                            "mlp.experts": 4,
                        },
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "int",
                    },
                }
            },
        },
    }


def test_split_mixed_num_bits_creates_standard_w4_and_w8_groups():
    normalized, changed = split_mixed_num_bits(_mixed_config())

    assert changed is True
    groups = normalized["quantization_config"]["config_groups"]
    assert set(groups) == {"group_0_w4", "group_0_w8"}
    assert groups["group_0_w4"]["weights"]["num_bits"] == 4
    assert groups["group_0_w4"]["targets"] == [r"re:.*mlp\.experts(?:\..*)?$"]
    assert groups["group_0_w8"]["weights"]["num_bits"] == 8
    assert groups["group_0_w8"]["targets"] == [
        r"re:.*mlp\.shared_experts(?:\..*)?$",
        r"re:.*self_attn\.q_a_proj(?:\..*)?$",
    ]
    assert _mixed_config()["quantization_config"]["config_groups"]["group_0"][
        "weights"
    ]["num_bits"]["mlp.experts"] == 4


def test_prepare_runtime_model_links_weights_and_preserves_source(tmp_path: Path):
    source = tmp_path / "model"
    source.mkdir()
    original = _mixed_config()
    (source / "config.json").write_text(json.dumps(original), encoding="utf-8")
    (source / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")

    runtime, method, changed = prepare_runtime_model(source, tmp_path / "runtime")

    assert method == "compressed-tensors"
    assert changed is True
    assert runtime != source
    assert (runtime / ".ready").is_file()
    assert (runtime / "model-00001-of-00001.safetensors").is_symlink()
    assert (runtime / "tokenizer.json").is_symlink()
    runtime_config = json.loads((runtime / "config.json").read_text())
    assert set(runtime_config["quantization_config"]["config_groups"]) == {
        "group_0_w4",
        "group_0_w8",
    }
    assert json.loads((source / "config.json").read_text()) == original

    repeated, _, repeated_changed = prepare_runtime_model(
        source, tmp_path / "runtime"
    )
    assert repeated == runtime
    assert repeated_changed is True


def test_standard_compressed_tensors_config_uses_source_directly(tmp_path: Path):
    source = tmp_path / "model"
    source.mkdir()
    config = _mixed_config()
    config["quantization_config"]["config_groups"]["group_0"]["weights"][
        "num_bits"
    ] = 8
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    runtime, method, changed = prepare_runtime_model(source, tmp_path / "runtime")

    assert runtime == source.resolve()
    assert method == "compressed-tensors"
    assert changed is False
    assert not (tmp_path / "runtime").exists()


def test_ambiguous_mixed_group_is_rejected():
    config = _mixed_config()
    config["quantization_config"]["config_groups"]["group_0"]["targets"] = [
        "GlmMoeDsaMLP"
    ]

    with pytest.raises(QuantConfigError, match="target exactly"):
        split_mixed_num_bits(config)
