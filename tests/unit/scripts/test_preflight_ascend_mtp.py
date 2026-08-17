import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

import scripts.preflight_ascend_mtp as preflight


def _write_config(path: Path, **updates) -> Path:
    path.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "vocab_size": 128,
        "num_hidden_layers": 3,
        "n_routed_experts": 1,
    }
    config.update(updates)
    (path / "config.json").write_text(json.dumps(config))
    return path


def _native_mtp_tensors(layer_idx: int = 3) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."
    suffixes = {
        "eh_proj.weight",
        "hnorm.weight",
        "enorm.weight",
        "shared_head.norm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.experts.0.gate_proj.weight",
        "mlp.experts.0.up_proj.weight",
        "mlp.experts.0.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    return {prefix + suffix: torch.ones(1) for suffix in suffixes}


def test_compare_model_configs_ignores_quantization_metadata(tmp_path):
    bf16 = _write_config(tmp_path / "bf16")
    w4a8 = _write_config(
        tmp_path / "w4a8",
        quantization_config={"quant_method": "compressed-tensors"},
    )

    assert preflight.compare_model_configs(bf16, w4a8)["hidden_size"] == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3"),
        ("hidden_size", 96),
        ("vocab_size", 256),
        ("num_hidden_layers", 4),
    ],
)
def test_compare_model_configs_rejects_structural_mismatch(tmp_path, field, value):
    bf16 = _write_config(tmp_path / "bf16")
    w4a8 = _write_config(tmp_path / "w4a8", **{field: value})

    with pytest.raises(preflight.PreflightError, match=field):
        preflight.compare_model_configs(bf16, w4a8)


def test_compare_model_configs_unwraps_text_config(tmp_path):
    bf16 = _write_config(tmp_path / "bf16")
    nested = tmp_path / "nested"
    nested.mkdir()
    base = json.loads((bf16 / "config.json").read_text())
    (nested / "config.json").write_text(json.dumps({"text_config": base}))

    assert preflight.compare_model_configs(bf16, nested)["model_type"] == (
        "glm_moe_dsa"
    )


def test_compare_model_configs_requires_glm_moe_dsa(tmp_path):
    bf16 = _write_config(tmp_path / "bf16", model_type="qwen3")
    w4a8 = _write_config(tmp_path / "w4a8", model_type="qwen3")

    with pytest.raises(preflight.PreflightError, match="glm_moe_dsa"):
        preflight.compare_model_configs(bf16, w4a8)


def test_compare_tokenizers_accepts_identical_assets(tmp_path):
    bf16 = _write_config(tmp_path / "bf16")
    w4a8 = _write_config(tmp_path / "w4a8")
    for path in (bf16, w4a8):
        (path / "tokenizer.json").write_text('{"model":"same"}')
        (path / "tokenizer_config.json").write_text('{"eos_token":"<eos>"}')

    assert preflight.compare_tokenizers(bf16, w4a8) == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]


def test_compare_tokenizers_rejects_mismatch(tmp_path):
    bf16 = _write_config(tmp_path / "bf16")
    w4a8 = _write_config(tmp_path / "w4a8")
    (bf16 / "tokenizer.json").write_text('{"model":"bf16"}')
    (w4a8 / "tokenizer.json").write_text('{"model":"w4a8"}')

    with pytest.raises(preflight.PreflightError, match="tokenizer.json"):
        preflight.compare_tokenizers(bf16, w4a8)


def test_validate_native_mtp_weights_accepts_index(tmp_path):
    model = _write_config(tmp_path / "model")
    shard = model / "part-2.safetensors"
    tensors = _native_mtp_tensors()
    save_file(tensors, shard)
    index = {"weight_map": dict.fromkeys(tensors, "part-2.safetensors")}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))

    assert preflight.validate_native_mtp_weights(model, 3) == len(tensors)


def test_validate_native_mtp_weights_accepts_single_file(tmp_path):
    model = _write_config(tmp_path / "model")
    tensors = _native_mtp_tensors()
    save_file(tensors, model / "model.safetensors")

    assert preflight.validate_native_mtp_weights(model, 3) == len(tensors)


def test_validate_native_mtp_weights_accepts_modelslim_layout(tmp_path):
    model = _write_config(tmp_path / "model")
    tensors = _native_mtp_tensors()
    save_file(tensors, model / "mtp.safetensors")
    (model / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "base.safetensors"}})
    )

    assert preflight.validate_native_mtp_weights(model, 3) == len(tensors)


def test_validate_native_mtp_weights_rejects_integer_tensors(tmp_path):
    model = _write_config(tmp_path / "model")
    tensors = {
        key: tensor.to(torch.int8) for key, tensor in _native_mtp_tensors().items()
    }
    save_file(tensors, model / "mtp.safetensors")

    with pytest.raises(preflight.PreflightError, match="must be floating point"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_native_mtp_weights_rejects_missing_layer(tmp_path):
    model = _write_config(tmp_path / "model")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.2.x": "part.safetensors"}})
    )

    with pytest.raises(preflight.PreflightError, match="model.layers.3"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_native_mtp_weights_rejects_missing_critical_key(tmp_path):
    model = _write_config(tmp_path / "model")
    save_file(
        {"model.layers.3.eh_proj.weight": torch.ones(1)},
        model / "model.safetensors",
    )

    with pytest.raises(preflight.PreflightError, match="critical"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_native_mtp_weights_rejects_missing_shard(tmp_path):
    model = _write_config(tmp_path / "model")
    keys = dict.fromkeys(_native_mtp_tensors(), "missing.safetensors")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": keys})
    )

    with pytest.raises(preflight.PreflightError, match="shard"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_native_mtp_weights_rejects_index_key_absent_from_shard(tmp_path):
    model = _write_config(tmp_path / "model")
    tensors = _native_mtp_tensors()
    missing_key = next(iter(tensors))
    save_file(
        {key: tensor for key, tensor in tensors.items() if key != missing_key},
        model / "part.safetensors",
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "part.safetensors")})
    )

    with pytest.raises(preflight.PreflightError, match="absent from shard"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_native_mtp_weights_rejects_unexpected_key(tmp_path):
    model = _write_config(tmp_path / "model")
    tensors = _native_mtp_tensors()
    tensors["model.layers.3.unsupported.weight"] = torch.ones(1)
    save_file(tensors, model / "model.safetensors")

    with pytest.raises(preflight.PreflightError, match="unexpected"):
        preflight.validate_native_mtp_weights(model, 3)


def test_validate_dataset_accepts_required_columns(tmp_path):
    dataset_path = tmp_path / "dataset"
    Dataset.from_dict(
        {"input_ids": [[1]], "loss_mask": [[1]], "seq_len": [1]}
    ).save_to_disk(dataset_path)

    assert preflight.validate_dataset(dataset_path) == 1


def test_validate_dataset_rejects_missing_column(tmp_path):
    dataset_path = tmp_path / "dataset"
    Dataset.from_dict({"input_ids": [[1]], "seq_len": [1]}).save_to_disk(dataset_path)

    with pytest.raises(preflight.PreflightError, match="loss_mask"):
        preflight.validate_dataset(dataset_path)


def test_parse_args_allows_verifier_to_skip_dataset_check():
    args = preflight.parse_args(
        [
            "--mtp-model",
            "/model",
            "--verifier-model",
            "/model",
            "--data-path",
            "/dataset-still-building",
            "--hidden-states-path",
            "/hidden-states",
            "--skip-dataset-check",
        ]
    )

    assert args.skip_dataset_check is True


def test_validate_shared_directory_writes_probe(tmp_path):
    shared = tmp_path / "hidden_states"

    preflight.validate_shared_directory(shared)

    assert shared.is_dir()
    assert list(shared.iterdir()) == []


def test_validate_shared_directory_rejects_file(tmp_path):
    shared = tmp_path / "not-a-directory"
    shared.write_text("x")

    with pytest.raises(preflight.PreflightError, match="directory"):
        preflight.validate_shared_directory(shared)


class _FakeAccelerator:
    type = "npu"


def _fake_accelerator():
    return _FakeAccelerator()


def test_validate_npu_runtime_accepts_npu(monkeypatch):
    monkeypatch.setattr(
        preflight.torch.accelerator,
        "current_accelerator",
        _fake_accelerator,
    )
    monkeypatch.setattr(preflight.torch.accelerator, "device_count", lambda: 16)
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda name: {"torch-npu": "2.10.0.post2"}[name],
    )
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())

    result = preflight.validate_npu_runtime(16)

    assert result["device_type"] == "npu"
    assert result["device_count"] == 16


def test_validate_npu_runtime_rejects_wrong_device(monkeypatch):
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda name: {"torch-npu": "2.10.0.post2"}[name],
    )
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        preflight.torch.accelerator,
        "current_accelerator",
        lambda: None,
    )

    with pytest.raises(preflight.PreflightError, match="Ascend NPU"):
        preflight.validate_npu_runtime(16)


def test_validate_npu_runtime_can_be_skipped():
    assert preflight.validate_npu_runtime(16, skip=True) == {"skipped": True}


def test_validate_npu_runtime_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda _name: "2.9.0",
    )
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())

    with pytest.raises(preflight.PreflightError, match="torch-npu"):
        preflight.validate_npu_runtime(16)


def test_validate_npu_runtime_rejects_vllm_version_mismatch(monkeypatch):
    versions = {
        "torch-npu": "2.10.0.post2",
        "vllm": "0.22.0",
        "vllm-ascend": "0.23.0rc1",
    }
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        versions.__getitem__,
    )
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        preflight.torch.accelerator,
        "current_accelerator",
        _fake_accelerator,
    )
    monkeypatch.setattr(preflight.torch.accelerator, "device_count", lambda: 16)

    with pytest.raises(preflight.PreflightError, match="vllm version mismatch"):
        preflight.validate_npu_runtime(16, require_vllm=True)


def test_validate_endpoint_uses_health_url(monkeypatch):
    seen = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def _open(request, timeout):
        seen.append((request.full_url, timeout))
        return _Response()

    monkeypatch.setattr(preflight.urllib.request, "urlopen", _open)

    preflight.validate_endpoint("http://verifier:8000/v1", timeout=3.0)

    assert seen == [("http://verifier:8000/health", 3.0)]
