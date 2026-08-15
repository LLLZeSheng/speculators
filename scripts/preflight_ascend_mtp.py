#!/usr/bin/env python3
"""Cheap preflight checks for GLM-5.2 online MTP training on Ascend."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from safetensors import safe_open


class PreflightError(RuntimeError):
    """Raised when an environment prerequisite is not satisfied."""


_STRUCTURAL_FIELDS = (
    "model_type",
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
_HTTP_OK = 200
_EXPECTED_VERSIONS = {
    "torch-npu": os.environ.get("EXPECTED_TORCH_NPU_VERSION", "2.10.0.post2"),
    "vllm": os.environ.get("EXPECTED_VLLM_VERSION", "0.23.0"),
    "vllm-ascend": os.environ.get(
        "EXPECTED_VLLM_ASCEND_VERSION", "0.23.0rc1"
    ),
}
_GLM_MTP_NON_EXPERT_SUFFIXES = (
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
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def _load_model_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise PreflightError(f"missing model config: {path}")
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read model config {path}: {exc}") from exc
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def compare_model_configs(
    bf16_model_path: str | Path,
    verifier_model_path: str | Path,
) -> dict[str, Any]:
    """Require matching GLM structural fields while ignoring quantization data."""
    bf16 = _load_model_config(bf16_model_path)
    verifier = _load_model_config(verifier_model_path)
    if bf16.get("model_type") != "glm_moe_dsa":
        raise PreflightError(
            "BF16 checkpoint model_type must be glm_moe_dsa, got "
            f"{bf16.get('model_type')!r}"
        )
    for field in _STRUCTURAL_FIELDS:
        left = bf16.get(field)
        right = verifier.get(field)
        if left is None or right is None:
            raise PreflightError(f"model config field {field!r} is missing")
        if left != right:
            raise PreflightError(
                f"model config mismatch for {field}: BF16={left!r}, W4A8={right!r}"
            )
    return {field: bf16[field] for field in _STRUCTURAL_FIELDS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tokenizers(
    bf16_model_path: str | Path,
    verifier_model_path: str | Path,
) -> list[str]:
    """Compare local tokenizer assets byte-for-byte without loading model code."""
    bf16_root = Path(bf16_model_path)
    verifier_root = Path(verifier_model_path)
    compared: list[str] = []
    for filename in _TOKENIZER_FILES:
        left = bf16_root / filename
        right = verifier_root / filename
        if left.is_file() != right.is_file():
            raise PreflightError(
                f"tokenizer asset {filename} exists in only one checkpoint"
            )
        if left.is_file():
            if _sha256(left) != _sha256(right):
                raise PreflightError(f"tokenizer asset mismatch: {filename}")
            compared.append(filename)
    if not any(name in compared for name in ("tokenizer.json", "tokenizer.model")):
        raise PreflightError(
            "no tokenizer.json or tokenizer.model found in both checkpoints"
        )
    return compared


def _load_checkpoint_keys(root: Path) -> tuple[list[str], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
            if not isinstance(weight_map, dict):
                raise TypeError("weight_map must be an object")
            return list(weight_map), weight_map
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PreflightError(
                f"invalid safetensors index {index_path}: {exc}"
            ) from exc
    single_path = root / "model.safetensors"
    if not single_path.is_file():
        raise PreflightError(f"no safetensors checkpoint found under {root}")
    try:
        with safe_open(str(single_path), framework="pt") as handle:
            return list(handle.keys()), None
    except (OSError, ValueError) as exc:
        raise PreflightError(f"invalid safetensors file {single_path}: {exc}") from exc


def validate_native_mtp_weights(model_path: str | Path, layer_idx: int) -> int:
    """Count native GLM MTP keys without loading any tensor payloads."""
    root = Path(model_path)
    keys, weight_map = _load_checkpoint_keys(root)

    prefix = f"model.layers.{layer_idx}."
    mtp_keys = [key for key in keys if key.startswith(prefix)]
    count = len(mtp_keys)
    if count == 0:
        raise PreflightError(f"no native MTP weights found with prefix {prefix}")
    config = _load_model_config(root)
    num_experts = config.get("n_routed_experts")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise PreflightError("model config n_routed_experts must be a positive integer")
    required = {prefix + suffix for suffix in _GLM_MTP_NON_EXPERT_SUFFIXES}
    required.update(
        prefix + f"mlp.experts.{expert}.{projection}_proj.weight"
        for expert in range(num_experts)
        for projection in ("gate", "up", "down")
    )
    missing = sorted(required - set(mtp_keys))
    if missing:
        raise PreflightError(f"native MTP critical tensors are missing: {missing}")
    unexpected = sorted(set(mtp_keys) - required)
    if unexpected:
        raise PreflightError(f"native MTP tensors are unexpected: {unexpected}")
    if weight_map is not None:
        _validate_indexed_tensors(root, weight_map, mtp_keys)
    return count


def _validate_indexed_tensors(
    root: Path,
    weight_map: dict[str, str],
    keys: list[str],
) -> None:
    shard_to_keys: dict[str, set[str]] = {}
    for key in keys:
        shard = weight_map[key]
        if not isinstance(shard, str) or not (root / shard).is_file():
            raise PreflightError(f"native MTP index references missing shard: {shard}")
        shard_to_keys.setdefault(shard, set()).add(key)

    for shard, expected in shard_to_keys.items():
        try:
            with safe_open(str(root / shard), framework="pt") as handle:
                absent = sorted(expected - set(handle.keys()))
        except Exception as exc:
            raise PreflightError(
                f"cannot open native MTP shard {shard}: {exc}"
            ) from exc
        if absent:
            raise PreflightError(
                f"native MTP tensors absent from shard {shard}: {absent}"
            )


def validate_dataset(data_path: str | Path) -> int:
    """Validate the token dataset schema and return its row count."""
    try:
        dataset = load_from_disk(str(data_path))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise PreflightError(f"cannot load dataset {data_path}: {exc}") from exc
    columns = set(getattr(dataset, "column_names", []))
    required = {"input_ids", "loss_mask", "seq_len"}
    missing = sorted(required - columns)
    if missing:
        raise PreflightError(f"dataset is missing required columns: {missing}")
    if len(dataset) == 0:
        raise PreflightError("dataset is empty")
    return len(dataset)


def validate_shared_directory(path: str | Path) -> None:
    """Create, write, fsync, and remove a probe file in shared storage."""
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise PreflightError(f"shared path is not a directory: {directory}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".preflight-{os.getpid()}-{uuid.uuid4().hex}"
        with probe.open("w") as stream:
            stream.write("ok\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
    except OSError as exc:
        raise PreflightError(
            f"shared directory is not writable: {directory}: {exc}"
        ) from exc


def validate_npu_runtime(
    required_devices: int,
    *,
    skip: bool = False,
    require_vllm: bool = False,
) -> dict[str, Any]:
    """Validate torch_npu registration, device count, and optional vLLM packages."""
    if skip:
        return {"skipped": True}
    try:
        torch_npu_version = importlib.metadata.version("torch-npu")
        importlib.import_module("torch_npu")
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        raise PreflightError(
            f"torch_npu is not installed or importable: {exc}"
        ) from exc

    expected_torch_npu = _EXPECTED_VERSIONS["torch-npu"]
    if torch_npu_version != expected_torch_npu:
        raise PreflightError(
            f"torch-npu version mismatch: expected {expected_torch_npu}, "
            f"got {torch_npu_version}"
        )

    accelerator = torch.accelerator.current_accelerator()
    device_type = getattr(accelerator, "type", None)
    if device_type not in {"npu", "privateuseone"}:
        raise PreflightError(
            f"Ascend NPU accelerator is not active (detected {device_type!r})"
        )
    device_count = torch.accelerator.device_count()
    if device_count < required_devices:
        raise PreflightError(
            f"visible NPU count {device_count} is lower than required "
            f"{required_devices}"
        )

    result: dict[str, Any] = {
        "device_type": "npu" if device_type == "privateuseone" else device_type,
        "device_count": device_count,
        "torch_npu": torch_npu_version,
    }
    if require_vllm:
        for distribution in ("vllm", "vllm-ascend"):
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise PreflightError(
                    f"required package is missing: {distribution}"
                ) from exc
            expected = _EXPECTED_VERSIONS[distribution]
            if installed != expected:
                raise PreflightError(
                    f"{distribution} version mismatch: expected {expected}, "
                    f"got {installed}"
                )
            result[distribution] = installed
    return result


def validate_endpoint(endpoint: str, *, timeout: float = 5.0) -> None:
    """Require a successful vLLM health response."""
    parsed = urllib.parse.urlsplit(endpoint.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PreflightError(f"endpoint must be an HTTP(S) URL: {endpoint}")
    path = parsed.path
    if path.endswith("/v1"):
        path = path[:-3]
    health_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path.rstrip('/')}/health", "", "")
    )
    request = urllib.request.Request(health_url, method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if response.status != _HTTP_OK:
                raise PreflightError(
                    f"verifier health endpoint returned HTTP {response.status}"
                )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PreflightError(
            f"verifier health endpoint is unreachable: {health_url}: {exc}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GLM-5.2 Ascend online MTP training prerequisites"
    )
    parser.add_argument("--bf16-model", required=True)
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--required-devices", type=int, default=16)
    parser.add_argument("--endpoint")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--skip-device-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = compare_model_configs(args.bf16_model, args.verifier_model)
        tokenizer_files = compare_tokenizers(args.bf16_model, args.verifier_model)
        mtp_tensors = validate_native_mtp_weights(
            args.bf16_model,
            config["num_hidden_layers"],
        )
        rows = validate_dataset(args.data_path)
        validate_shared_directory(args.hidden_states_path)
        runtime = validate_npu_runtime(
            args.required_devices,
            skip=args.skip_device_check,
            require_vllm=args.require_vllm,
        )
        if args.endpoint:
            validate_endpoint(args.endpoint)
    except PreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1

    print("Preflight passed")
    print(f"  architecture: {config['model_type']}")
    print(f"  hidden size: {config['hidden_size']}")
    print(f"  native MTP tensors: {mtp_tensors}")
    print(f"  tokenizer assets: {', '.join(tokenizer_files)}")
    print(f"  dataset rows: {rows}")
    print(f"  runtime: {json.dumps(runtime, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
