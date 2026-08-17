#!/usr/bin/env python3
"""Audit GLM-5.2 native MTP checkpoint and Ascend ModelSlim metadata.

This is a static audit. It proves that checkpoint tensors and ModelSlim
metadata are internally consistent, and highlights shared embedding/head
weights that still require an explicit runtime binding. It does not claim that
a running vLLM process has loaded the values; use the emitted loader findings
and startup log checks for that final step.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(
    "/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4"
)
DEFAULT_ASCEND_PATCH = Path(
    "/vllm-workspace/vllm-ascend/"
    "vllm_ascend/patch/worker/patch_deepseek_mtp.py"
)

SPECIAL_MTP_NAMES = ("embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head")
QUANT_AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_offset",
    ".scale_bias",
    ".input_scale",
    ".input_offset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--ascend-patch",
        type=Path,
        default=DEFAULT_ASCEND_PATCH,
        help="installed vLLM-Ascend patch_deepseek_mtp.py",
    )
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="sample representative tensors with safetensors and check finiteness",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the machine-readable report to this path",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_weight_map(model: Path) -> tuple[dict[str, str], list[str]]:
    candidates = sorted(glob.glob(str(model / "*.safetensors.index.json")))
    if not candidates:
        raise FileNotFoundError(f"no *.safetensors.index.json under {model}")
    weight_map: dict[str, str] = {}
    for candidate in candidates:
        current = load_json(Path(candidate)).get("weight_map", {})
        if not isinstance(current, dict):
            raise ValueError(f"invalid weight_map: {candidate}")
        duplicates = set(weight_map).intersection(current)
        if duplicates:
            raise ValueError(f"duplicate tensor in indexes: {next(iter(duplicates))}")
        weight_map.update({str(key): str(value) for key, value in current.items()})
    return weight_map, candidates


def rewrite_mtp_name(name: str, layer: int) -> str:
    """Apply the upstream DeepSeek MTP structural name rewrite."""
    prefix = f"model.layers.{layer}."
    if "embed_tokens" in name:
        return name.replace(prefix, "model.")
    if any(token in name for token in SPECIAL_MTP_NAMES):
        return name
    return name.replace(prefix, f"model.layers.{layer}.mtp_block.")


def representative_keys(keys: list[str]) -> list[str]:
    patterns = (
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head.norm.weight",
        "shared_head.head.weight",
        "embed_tokens.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "mlp.experts.0.gate_proj.weight",
        "mlp.experts.0.down_proj.weight",
    )
    selected = [
        key for key in keys if any(key.endswith(pattern) for pattern in patterns)
    ]
    return selected[:20]


def check_tensor_values(
    model: Path, weight_map: dict[str, str], keys: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        return [], [f"--check-values requires torch and safetensors: {error}"]

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for key in representative_keys(keys):
        shard = model / weight_map[key]
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                tensor_slice = handle.get_slice(key)
                shape = list(tensor_slice.get_shape())
                if len(shape) == 1:
                    sample = tensor_slice[: min(shape[0], 4096)]
                else:
                    slices = tuple(slice(0, min(size, 16)) for size in shape)
                    sample = tensor_slice[slices]
                sample = sample.float()
            finite = bool(torch.isfinite(sample).all().item())
            results.append(
                {
                    "name": key,
                    "shape": shape,
                    "finite": finite,
                    "mean": float(sample.mean().item()),
                    "std": float(sample.std().item()),
                }
            )
            if not finite:
                errors.append(f"non-finite tensor sample: {key}")
        except Exception as error:  # noqa: BLE001 - audit should report all failures
            errors.append(f"failed to sample {key}: {error}")
    return results, errors


def inspect_ascend_patch(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "findings": [
                "Ascend patch file not found; runtime mapping was not audited"
            ],
        }
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []
    if "AutoWeightsLoader" in text:
        findings.append("uses AutoWeightsLoader")
    if "loader.load_weights(weights)" in text:
        findings.append("delegates to generic loader.load_weights(weights)")
    if "_rewrite_spec_layer_name" in text:
        findings.append("contains upstream-style MTP name rewrite")
    if re.search(r"shared_head|embed_tokens", text):
        findings.append("mentions shared_head/embed_tokens")
    else:
        findings.append("does not visibly bind shared_head/embed_tokens")
    return {"path": str(path), "present": True, "findings": findings}


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model = args.model.resolve()
    config_path = model / "config.json"
    description_path = model / "quant_model_description.json"
    config = load_json(config_path)
    description = load_json(description_path)
    weight_map, indexes = load_weight_map(model)

    start = int(config["num_hidden_layers"])
    count = int(config.get("num_nextn_predict_layers", 0))
    if count <= 0:
        raise ValueError("config has no positive num_nextn_predict_layers")

    errors: list[str] = []
    warnings: list[str] = []
    layer_reports: list[dict[str, Any]] = []
    all_mtp_keys: list[str] = []

    for layer in range(start, start + count):
        prefix = f"model.layers.{layer}."
        keys = sorted(key for key in weight_map if key.startswith(prefix))
        all_mtp_keys.extend(keys)
        weights = [key for key in keys if key.endswith(".weight")]
        if not keys:
            errors.append(f"checkpoint has no tensors for MTP layer {layer}")

        critical = {
            "enorm": prefix + "enorm.weight",
            "hnorm": prefix + "hnorm.weight",
            "eh_proj": prefix + "eh_proj.weight",
            "shared_head_norm": prefix + "shared_head.norm.weight",
            "shared_head_head": prefix + "shared_head.head.weight",
            "layer_embed_tokens": prefix + "embed_tokens.weight",
        }
        presence = {name: key in weight_map for name, key in critical.items()}
        for name in ("enorm", "hnorm", "eh_proj", "shared_head_norm"):
            if not presence[name]:
                errors.append(f"MTP layer {layer} lacks required {critical[name]}")

        # These two parameters are shared by design, but the installed loader must
        # explicitly bind them when duplicate layer-local tensors are absent.
        for name in ("shared_head_head", "layer_embed_tokens"):
            if not presence[name]:
                warnings.append(
                    f"MTP layer {layer} lacks {critical[name]}; runtime must bind "
                    "the corresponding main-model shared weight"
                )

        no_description = [key for key in weights if key not in description]
        non_float = [key for key in weights if description.get(key) != "FLOAT"]
        quant_aux = [key for key in keys if key.endswith(QUANT_AUX_SUFFIXES)]
        if no_description:
            errors.append(
                f"MTP layer {layer} has {len(no_description)} weights absent from "
                "quant_model_description.json"
            )
        if non_float:
            errors.append(
                f"MTP layer {layer} has {len(non_float)} weights not marked FLOAT"
            )
        if quant_aux:
            warnings.append(
                f"MTP layer {layer} contains {len(quant_aux)} quantization auxiliaries"
            )

        mapped = {
            key: rewrite_mtp_name(key, layer) for key in representative_keys(keys)
        }
        layer_reports.append(
            {
                "layer": layer,
                "tensor_count": len(keys),
                "weight_count": len(weights),
                "shards": dict(Counter(weight_map[key] for key in keys)),
                "critical_presence": presence,
                "missing_description": no_description,
                "non_float_description": non_float,
                "quant_auxiliaries": quant_aux,
                "representative_mapping": mapped,
            }
        )

    main_shared = {
        name: weight_map.get(name)
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
        if name in weight_map
    }
    patch_report = inspect_ascend_patch(args.ascend_patch)
    missing_shared = any(
        not report["critical_presence"][name]
        for report in layer_reports
        for name in ("shared_head_head", "layer_embed_tokens")
    )
    patch_findings = " ".join(patch_report["findings"])
    if missing_shared and "does not visibly bind" in patch_findings:
        errors.append(
            "layer-local shared weights are absent and the Ascend patch has no "
            "visible shared_head/embed_tokens binding"
        )
    elif missing_shared:
        warnings.append(
            "static files cannot prove the runtime shared-weight binding; inspect "
            "the loader return set or enforce a runtime missing-parameter check"
        )

    value_results: list[dict[str, Any]] = []
    if args.check_values:
        value_results, value_errors = check_tensor_values(
            model, weight_map, all_mtp_keys
        )
        errors.extend(value_errors)

    report = {
        "model": str(model),
        "indexes": indexes,
        "num_hidden_layers": start,
        "num_nextn_predict_layers": count,
        "layer_reports": layer_reports,
        "main_shared_weights": main_shared,
        "ascend_patch": patch_report,
        "sampled_values": value_results,
        "warnings": warnings,
        "errors": errors,
        "verdict": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
    }
    return report, 1 if errors else 0


def print_report(report: dict[str, Any]) -> None:
    print(f"Model: {report['model']}")
    print(
        "MTP range: "
        f"[{report['num_hidden_layers']}, "
        f"{report['num_hidden_layers'] + report['num_nextn_predict_layers']})"
    )
    for layer in report["layer_reports"]:
        print(
            f"\nLayer {layer['layer']}: tensors={layer['tensor_count']} "
            f"weights={layer['weight_count']}"
        )
        for name, present in layer["critical_presence"].items():
            print(f"  {'PRESENT' if present else 'MISSING'} {name}")
        print("  Representative checkpoint -> runtime mapping:")
        for source, destination in layer["representative_mapping"].items():
            print(f"    {source} -> {destination}")
    print("\nMain shared weights:")
    for name, shard in report["main_shared_weights"].items():
        print(f"  {name} -> {shard}")
    print("\nAscend loader patch:")
    print(f"  {report['ascend_patch']['path']}")
    for finding in report["ascend_patch"]["findings"]:
        print(f"  - {finding}")
    if report["sampled_values"]:
        print("\nSampled tensor values:")
        for item in report["sampled_values"]:
            print(
                f"  {item['name']}: shape={item['shape']} finite={item['finite']} "
                f"mean={item['mean']:.6g} std={item['std']:.6g}"
            )
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print(f"  - {error}")
    print(f"\nVERDICT={report['verdict']}")


def main() -> int:
    args = parse_args()
    try:
        report, status = audit(args)
    except Exception as error:  # noqa: BLE001 - command boundary
        print(f"AUDIT_ERROR: {error}", file=sys.stderr)
        return 2
    print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
