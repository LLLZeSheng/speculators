import argparse
import json
from pathlib import Path

from scripts.check_glm52_mtp_mapping import audit, rewrite_mtp_name


def _write_fixture(tmp_path: Path, *, omit_description: bool = False):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"num_hidden_layers": 78, "num_nextn_predict_layers": 1}),
        encoding="utf-8",
    )
    prefix = "model.layers.78."
    keys = [
        prefix + "enorm.weight",
        prefix + "hnorm.weight",
        prefix + "eh_proj.weight",
        prefix + "shared_head.norm.weight",
        prefix + "shared_head.head.weight",
        prefix + "embed_tokens.weight",
        prefix + "self_attn.q_a_proj.weight",
        prefix + "mlp.experts.0.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "part.safetensors")}),
        encoding="utf-8",
    )
    description_keys = keys[:]
    if omit_description:
        description_keys.remove(prefix + "self_attn.q_a_proj.weight")
    (model / "quant_model_description.json").write_text(
        json.dumps({key: "FLOAT" for key in description_keys}),
        encoding="utf-8",
    )
    patch = tmp_path / "patch_deepseek_mtp.py"
    patch.write_text("shared_head = embed_tokens\n", encoding="utf-8")
    args = argparse.Namespace(
        model=model,
        ascend_patch=patch,
        check_values=False,
        json_output=None,
    )
    return args


def test_rewrite_mtp_names_matches_upstream_structure():
    assert rewrite_mtp_name(
        "model.layers.78.self_attn.q_a_proj.weight", 78
    ) == "model.layers.78.mtp_block.self_attn.q_a_proj.weight"
    assert rewrite_mtp_name(
        "model.layers.78.embed_tokens.weight", 78
    ) == "model.embed_tokens.weight"
    assert rewrite_mtp_name(
        "model.layers.78.shared_head.head.weight", 78
    ) == "model.layers.78.shared_head.head.weight"


def test_audit_accepts_complete_float_mtp_description(tmp_path: Path):
    report, status = audit(_write_fixture(tmp_path))

    assert status == 0
    assert report["verdict"] == "PASS"
    assert report["layer_reports"][0]["tensor_count"] == 8


def test_audit_rejects_mtp_weight_missing_from_description(tmp_path: Path):
    report, status = audit(_write_fixture(tmp_path, omit_description=True))

    assert status == 1
    assert report["verdict"] == "FAIL"
    assert any("absent from" in error for error in report["errors"])
