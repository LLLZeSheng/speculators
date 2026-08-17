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
    "container_mode": "CONTAINER_MODE",
    "existing_container_name": "EXISTING_CONTAINER_NAME",
    "container_name_prefix": "CONTAINER_NAME_PREFIX",
    "container_repo_path": "CONTAINER_REPO_PATH",
    "container_shm_size": "CONTAINER_SHM_SIZE",
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
    allowed = {
        "version",
        "cluster_name",
        "verifier_ips",
        "trainer_ips",
        "container_mounts",
    } | set(SCALAR_MAP)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown YAML keys: {unknown}")
    if config.get("version") != "1":
        raise ValueError("version must be 1")
    for key in ("verifier_ips", "trainer_ips"):
        values = config.get(key)
        if not isinstance(values, list) or len(values) != 4 or not all(values):
            raise ValueError(f"{key} must contain exactly four non-empty addresses")
    all_ips = [*config["verifier_ips"], *config["trainer_ips"]]  # type: ignore[misc]
    if len(set(all_ips)) != 8:
        raise ValueError("all verifier and trainer addresses must be unique")
    required_scalars = (
        "container_image",
        "container_mode",
        "container_name_prefix",
        "container_repo_path",
        "repo_path",
        "shared_root",
        "nic_name",
        "verifier_model_path",
        "verifier_source_model_path",
        "verifier_quantization_mode",
        "mtp_init_model_path",
        "data_path",
        "hidden_states_path",
        "output_path",
        "log_root",
        "trainer_mode",
        "trainer_data_mode",
        "smoke_run_id",
        "install_speculators_verifier",
        "install_speculators_trainer",
    )
    for key in required_scalars:
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} is required")
    for key, value in config.items():
        values = value if isinstance(value, list) else [value]
        if any("FILL_" in item for item in values):
            raise ValueError(f"{key} still contains a FILL_ placeholder")
    prefix = config["container_name_prefix"]
    assert isinstance(prefix, str)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix) is None:
        raise ValueError("container_name_prefix is not a valid Docker name prefix")
    if config.get("container_mode") not in {"create", "existing"}:
        raise ValueError("container_mode must be create or existing")
    existing_name = config.get("existing_container_name")
    if config.get("container_mode") == "existing":
        if not isinstance(existing_name, str) or not existing_name:
            raise ValueError(
                "existing_container_name is required when container_mode=existing"
            )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", existing_name) is None:
            raise ValueError("existing_container_name is not a valid Docker name")
    mounts = config.get("container_mounts")
    if not isinstance(mounts, list) or not mounts:
        raise ValueError("container_mounts must contain at least one mount")
    for mount in mounts:
        parts = mount.split(":", 2)
        if (
            len(parts) < 2
            or not parts[0].startswith("/")
            or not parts[1].startswith("/")
        ):
            raise ValueError(
                "container_mounts entries must use host_path:container_path[:options]"
            )
    if config.get("container_mode") == "create":
        repo_mount = f"{config['repo_path']}:{config['container_repo_path']}"
        if not any(
            mount == repo_mount or mount.startswith(repo_mount + ":")
            for mount in mounts
        ):
            raise ValueError(
                "container_mounts must map repo_path to container_repo_path"
            )

    integer_keys = (
        "verifier_port",
        "verifier_dp_size",
        "verifier_tp_size",
        "verifier_max_model_len",
        "verifier_max_num_seqs",
        "verifier_max_batched_tokens",
        "total_seq_len",
        "request_timeout",
        "max_retries",
        "epochs",
    )
    numbers = {}
    for key in integer_keys:
        value = config.get(key)
        if not isinstance(value, str) or not value.isdigit():
            raise ValueError(f"{key} must be a non-negative integer")
        numbers[key] = int(value)
    for key in integer_keys:
        if key != "max_retries" and numbers[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if numbers["verifier_dp_size"] * numbers["verifier_tp_size"] != 16:
        raise ValueError("verifier_dp_size * verifier_tp_size must equal 16")
    if numbers["verifier_max_model_len"] <= numbers["total_seq_len"]:
        raise ValueError("verifier_max_model_len must exceed total_seq_len")
    if (
        numbers["verifier_max_batched_tokens"]
        < numbers["verifier_max_model_len"]
    ):
        raise ValueError(
            "verifier_max_batched_tokens must be at least verifier_max_model_len"
        )
    if config.get("trainer_mode") not in {"smoke", "trainer"}:
        raise ValueError("trainer_mode must be smoke or trainer")
    if config.get("trainer_data_mode") not in {"online-cache", "offline"}:
        raise ValueError("trainer_data_mode must be online-cache or offline")


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
        ("container_mounts", "CONTAINER_MOUNTS"),
    ):
        values = config[yaml_key]
        assert isinstance(values, list)
        lines.append(f"{shell_key}=({' '.join(shlex.quote(item) for item in values)})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = load_flat_yaml(Path(args.config))
        validate(config)
        output = render(config)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(output, end="")


if __name__ == "__main__":
    main()
