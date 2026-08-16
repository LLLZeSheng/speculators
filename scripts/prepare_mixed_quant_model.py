#!/usr/bin/env python3
"""Create a non-invasive model view for mixed-bit compressed-tensors weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


class QuantConfigError(ValueError):
    """Raised when mixed-bit quantization metadata cannot be normalized safely."""


def _normalized_quant_method(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _target_for_suffix(suffix: str) -> str:
    """Match a named module itself and packed/fused children below it."""
    escaped = re.escape(suffix.strip("."))
    return rf"re:.*{escaped}(?:\..*)?$"


def split_mixed_num_bits(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Split dictionary-valued ``num_bits`` into standard config groups."""
    result = copy.deepcopy(config)
    quant_config = result.get("quantization_config")
    if not isinstance(quant_config, dict):
        return result, False

    quant_method = _normalized_quant_method(quant_config.get("quant_method"))
    if quant_method != "compressed-tensors":
        return result, False

    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise QuantConfigError(
            "compressed-tensors config_groups must be a non-empty object"
        )

    normalized_groups: dict[str, Any] = {}
    changed = False
    for group_name, scheme in groups.items():
        if not isinstance(scheme, dict):
            raise QuantConfigError(f"config group {group_name!r} must be an object")
        weights = scheme.get("weights")
        if not isinstance(weights, dict):
            normalized_groups[group_name] = scheme
            continue
        num_bits = weights.get("num_bits")
        if not isinstance(num_bits, dict):
            normalized_groups[group_name] = scheme
            continue

        if scheme.get("targets") != ["Linear"]:
            raise QuantConfigError(
                f"mixed-bit group {group_name!r} must target exactly ['Linear']; "
                "refusing an ambiguous rewrite"
            )
        if not num_bits:
            raise QuantConfigError(
                f"mixed-bit group {group_name!r} has an empty num_bits map"
            )

        suffixes_by_bits: dict[int, list[str]] = defaultdict(list)
        for suffix, bits in num_bits.items():
            if not isinstance(suffix, str) or not suffix.strip():
                raise QuantConfigError(
                    f"mixed-bit group {group_name!r} contains an invalid module suffix"
                )
            if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
                raise QuantConfigError(
                    f"mixed-bit group {group_name!r} has invalid bit width {bits!r} "
                    f"for {suffix!r}"
                )
            suffixes_by_bits[bits].append(suffix)

        for bits in sorted(suffixes_by_bits):
            split_scheme = copy.deepcopy(scheme)
            split_scheme["weights"]["num_bits"] = bits
            split_scheme["targets"] = [
                _target_for_suffix(suffix)
                for suffix in sorted(suffixes_by_bits[bits])
            ]
            split_name = f"{group_name}_w{bits}"
            if split_name in groups or split_name in normalized_groups:
                raise QuantConfigError(
                    f"generated config group name collides: {split_name}"
                )
            normalized_groups[split_name] = split_scheme
        changed = True

    quant_config["config_groups"] = normalized_groups
    return result, changed


def _model_fingerprint(source: Path, config_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"mixed-quant-config-v1\0")
    digest.update(str(source).encode())
    digest.update(b"\0")
    digest.update(config_bytes)
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        if not child.is_file() or not (
            child.name.endswith(".safetensors")
            or child.name.endswith(".safetensors.index.json")
        ):
            continue
        stat = child.stat()
        metadata = f"{child.name}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
        digest.update(metadata.encode())
    return digest.hexdigest()[:16]


def prepare_runtime_model(source: Path, runtime_root: Path) -> tuple[Path, str, bool]:
    """Return a model path with standard metadata, linking all original assets."""
    source = source.resolve(strict=True)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise QuantConfigError(f"model config is missing: {config_path}")

    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as exc:
        raise QuantConfigError(f"invalid JSON in {config_path}: {exc}") from exc

    quant_config = config.get("quantization_config")
    quant_method = _normalized_quant_method(
        quant_config.get("quant_method") if isinstance(quant_config, dict) else None
    )
    normalized, changed = split_mixed_num_bits(config)
    if not changed:
        return source, quant_method, False

    fingerprint = _model_fingerprint(source, config_bytes)
    destination = runtime_root / f"{source.name}-ct-split-{fingerprint}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    ready = destination / ".ready"
    if ready.is_file():
        return destination, quant_method, True
    if destination.exists():
        raise QuantConfigError(
            f"incomplete runtime model exists without .ready: {destination}"
        )

    temporary = runtime_root / (
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        for child in source.iterdir():
            reserved_names = {
                "config.json",
                ".ready",
                "runtime_model_manifest.json",
            }
            if child.name in reserved_names:
                continue
            os.symlink(
                child,
                temporary / child.name,
                target_is_directory=child.is_dir(),
            )
        (temporary / "config.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "runtime_model_manifest.json").write_text(
            json.dumps(
                {
                    "source_model": str(source),
                    "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "normalization": "split dictionary-valued num_bits by bit width",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / ".ready").touch()
        try:
            temporary.rename(destination)
        except FileExistsError:
            if not ready.is_file():
                raise QuantConfigError(
                    "concurrent preparation left an incomplete runtime model: "
                    f"{destination}"
                )
            shutil.rmtree(temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination, quant_method, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model_path, quant_method, changed = prepare_runtime_model(
            args.model, args.runtime_root
        )
    except (OSError, QuantConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(model_path)
    print(quant_method)
    print("normalized" if changed else "unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
