#!/usr/bin/env python3
"""Validate the flat Ascend MTP cluster YAML and render shell variables.

The parser intentionally supports only the small, dependency-free YAML subset
used by the checked-in template: top-level scalars and top-level string lists.
This keeps cluster bootstrap independent of PyYAML on all eight hosts.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


SCALAR_MAP = {
    "container_image": "IMAGE",
    "container_name_prefix": "CONTAINER_NAME_PREFIX",
    "container_repo_path": "CONTAINER_REPO_PATH",
    "repo_path": "REMOTE_REPO_PATH",
    "shared_root": "SHARED_ROOT",
    "nic_name": "NIC_NAME",
    "verifier_model_path": "VERIFIER_MODEL_PATH",
    "verifier_source_model_path": "VERIFIER_SOURCE_MODEL_PATH",
    "verifier_quantization_mode": "VERIFIER_QUANTIZATION_MODE",
    "mtp_init_model_path": "MTP_INIT_MODEL_PATH",
    "data_path": "DATA_PATH",
    "hidden_states_path": "HIDDEN_STATES_PATH",
    "output_path": "OUTPUT_PATH",
    "log_root": "LOG_ROOT",
    "verifier_port": "VERIFIER_PORT",
    "verifier_dp_size": "VERIFIER_DP_SIZE",
    "verifier_tp_size": "VERIFIER_TP_SIZE",
    "verifier_max_model_len": "VERIFIER_MAX_MODEL_LEN",
    "verifier_max_num_seqs": "VERIFIER_MAX_NUM_SEQS",
    "verifier_max_batched_tokens": "VERIFIER_MAX_BATCHED_TOKENS",
    "total_seq_len": "TOTAL_SEQ_LEN",
    "request_timeout": "REQUEST_TIMEOUT",
    "max_retries": "MAX_RETRIES",
    "epochs": "EPOCHS",
    "trainer_mode": "TRAINER_MODE",
    "trainer_data_mode": "TRAINER_DATA_MODE",
    "smoke_run_id": "SMOKE_RUN_ID",
    "install_speculators_verifier": "INSTALL_SPECULATORS_VERIFIER",
    "install_speculators_trainer": "INSTALL_SPECULATORS_TRAINER",
}


def parse_scalar(value: str, line_number: int) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[0] in "\"'":
        if value[0] == '"':
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid quoted value") from exc
        else:
            if not value.endswith("'"):
                raise ValueError(f"line {line_number}: unterminated quoted value")
            parsed = value[1:-1].replace("''", "'")
        if not isinstance(parsed, str):
            raise ValueError(f"line {line_number}: expected a string")
        return parsed
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def load_flat_yaml(path: Path) -> dict[str, str | list[str]]:
    result: dict[str, str | list[str]] = {}
    active_list: str | None = None
    key_pattern = re.compile(r"^([a-z][a-z0-9_]*):(?:\s*(.*))?$")
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith((" ", "\t")):
            stripped = raw_line.strip()
            if active_list is None or not stripped.startswith("- "):
                raise ValueError(
                    f"line {line_number}: only top-level lists are supported"
                )
            item = parse_scalar(stripped[2:], line_number)
            assert isinstance(result[active_list], list)
            result[active_list].append(item)  # type: ignore[union-attr]
            continue
        match = key_pattern.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"line {line_number}: invalid top-level YAML entry")
        key, raw_value = match.groups()
        if key in result:
            raise ValueError(f"line {line_number}: duplicate key {key!r}")
        if raw_value is None or not raw_value.strip():
            result[key] = []
            active_list = key
        else:
            result[key] = parse_scalar(raw_value, line_number)
            active_list = None
    return result


def boolean_as_flag(value: str, key: str) -> str:
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return "1"
    if normalized in {"false", "no", "0"}:
        return "0"
    raise ValueError(f"{key} must be true or false")


def validate(config: dict[str, str | list[str]]) -> None:
    allowed = {"version", "cluster_name", "verifier_ips", "trainer_ips"} | set(
        SCALAR_MAP
    )
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown YAML keys: {unknown}")
    for key in ("verifier_ips", "trainer_ips"):
        values = config.get(key)
        if not isinstance(values, list) or len(values) != 4 or not all(values):
            raise ValueError(f"{key} must contain exactly four non-empty addresses")
    all_ips = [*config["verifier_ips"], *config["trainer_ips"]]  # type: ignore[misc]
    if len(set(all_ips)) != 8:
        raise ValueError("all verifier and trainer addresses must be unique")
    for key in ("container_image", "container_name_prefix", "repo_path"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} is required")


def render(config: dict[str, str | list[str]]) -> str:
    lines = []
    for yaml_key, shell_key in SCALAR_MAP.items():
        value = config.get(yaml_key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{yaml_key} must be a scalar")
        if yaml_key.startswith("install_speculators_"):
            value = boolean_as_flag(value, yaml_key)
        # Preserve explicit runtime overrides, matching the former shell config.
        lines.append(f"{shell_key}=${{{shell_key}:-{shlex.quote(value)}}}")
    for yaml_key, shell_key in (
        ("verifier_ips", "CLUSTER_VERIFIER_IPS"),
        ("trainer_ips", "CLUSTER_TRAINER_IPS"),
    ):
        values = config[yaml_key]
        assert isinstance(values, list)
        lines.append(f"{shell_key}=({' '.join(shlex.quote(item) for item in values)})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_flat_yaml(Path(args.config))
    validate(config)
    print(render(config), end="")


if __name__ == "__main__":
    main()
