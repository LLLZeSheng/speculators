import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from safetensors import safe_open

_WEIGHT_ALIASES: dict[str, list[str]] = {
    "embed_tokens.weight": ["tok_embeddings.weight"],
    "lm_head.weight": ["output.weight"],
    "model.norm.weight": ["norm.weight"],
}

_SAFETENSORS_INDEX_NAMES = (
    "model.safetensors.index.json",
    "quant_model_weights.safetensors.index.json",
)


def find_local_safetensors_index(checkpoint_dir: str | Path) -> Path | None:
    """Find a supported local safetensors shard index.

    Ascend ModelSlim exports use ``quant_model_weights.safetensors.index.json``
    instead of the Hugging Face default. Prefer the standard name when both are
    present so existing checkpoints retain their original behavior.
    """
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return None
    return next(
        (
            directory / name
            for name in _SAFETENSORS_INDEX_NAMES
            if (directory / name).is_file()
        ),
        None,
    )


def _resolve_key(name: str, weight_map: dict[str, str]) -> str | None:
    """Try exact match, then suffix match, then known aliases."""
    for candidate in [name, *_WEIGHT_ALIASES.get(name, [])]:
        if candidate in weight_map:
            return candidate
        matched = next((k for k in weight_map if k.endswith(candidate)), None)
        if matched:
            return matched
    return None


def is_config_only_dir(path: str | Path) -> bool:
    """Return True if ``path`` is a local directory with a ``config.json`` but no
    weight files (``*.safetensors`` / ``*.bin``).

    Used to distinguish a saved speculator *config* (from which a fresh draft is
    initialized) from a full checkpoint whose weights should be loaded.

    :param path: A local directory path. Hub ids and non-directories return False.
    :return: True when the directory holds a config but no weights.
    """
    directory = Path(path)
    if not directory.is_dir():
        return False
    has_config = (directory / "config.json").is_file()
    # Weight files, plus sharded-checkpoint index files (e.g.
    # model.safetensors.index.json) -- the latter end in .json and would not match
    # the *.safetensors / *.bin globs, so a shard manifest must be checked explicitly
    # to avoid treating an incomplete sharded checkpoint as config-only.
    has_weights = (
        any(directory.glob("*.safetensors"))
        or any(directory.glob("*.bin"))
        or any(directory.glob("*.safetensors.index.json"))
        or any(directory.glob("*.bin.index.json"))
    )
    return has_config and not has_weights


def list_checkpoint_keys(checkpoint_dir: str | Path) -> list[str]:
    """List all tensor keys in a checkpoint without loading weights.

    Supports sharded safetensors (via index) and single safetensors formats.

    :param checkpoint_dir: Path to a local checkpoint directory.
    :return: List of tensor key names present in the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)

    keys: dict[str, None] = {}
    index_path = find_local_safetensors_index(checkpoint_dir)
    if index_path is not None:
        with index_path.open() as f:
            keys.update(dict.fromkeys(json.load(f)["weight_map"].keys()))

    # Some ModelSlim runtime views keep native, unquantized MTP tensors in an
    # auxiliary file which may not be represented by an older shard index.
    mtp_file = checkpoint_dir / "mtp.safetensors"
    if mtp_file.exists():
        with safe_open(str(mtp_file), framework="pt") as f:
            keys.update(dict.fromkeys(f.keys()))

    if keys:
        return list(keys)

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return list(f.keys())

    raise FileNotFoundError(
        f"No safetensors checkpoint found at {checkpoint_dir}. "
        "Expected model.safetensors.index.json, "
        "quant_model_weights.safetensors.index.json, model.safetensors, "
        "or mtp.safetensors."
    )


def load_model_layers(
    layer_names: list[str], model_path: str
) -> dict[str, torch.Tensor]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Supports both exact keys and suffix pattern matching.

    :param layer_names: list of tensor names or suffix patterns to load, e.g.
    ["model.embed_tokens.weight", "lm_head.weight"]
    :param model_path: either a local directory of huggingface model
    containing model.safetensors.index
    :return: dict mapping input names/patterns to loaded tensors
    """
    # download the index file or build weight map for single-file models
    local_index = find_local_safetensors_index(model_path)
    try:
        index_file = (
            local_index
            if local_index is not None
            else _resolve_file(model_path, "model.safetensors.index.json")
        )
        with Path(index_file).open() as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        logger.warning(
            "No supported safetensors index file found. "
            "Checking for `model.safetensors` instead."
        )
        model_file = _resolve_file(model_path, "model.safetensors")
        # Build virtual weight map for single-file models
        with safe_open(model_file, framework="pt", device="cpu") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")

    # Resolve names: try exact match, then suffix match, then known aliases
    name_to_key = {}  # Maps input name to actual checkpoint key
    for name in layer_names:
        key = _resolve_key(name, weight_map)
        if key:
            name_to_key[name] = key
        else:
            logger.warning(f"Tensor '{name}' not found in weight_map.")

    # group requested names by shard filename
    shard_to_names: dict[str, list[tuple[str, str]]] = {}
    for name, key in name_to_key.items():
        shard = weight_map[key]
        shard_to_names.setdefault(shard, []).append((name, key))

    if not shard_to_names:
        raise ValueError("None of the requested tensor names were found in the index.")

    # fetch each required shard and extract only the requested tensors
    out: dict[str, Any] = {}
    for shard_file, name_key_pairs in shard_to_names.items():
        shard_path = _resolve_file(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name, key in name_key_pairs:
                out[name] = f.get_tensor(key)
    return out


def _resolve_file(model_path: str, file_name: str) -> Path:
    """
    If model_path is a local directory, return path/<filename> if it exists.
    Otherwise treat model_path as a HF repo_id and download with hf_hub_download.

    :param model_path: local directory or HF repo_id
    :param file_name: filename to look for or download
    :return: local path to the resolved file
    """
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        logger.info("Loading from local directory: {}", model_path)
        p = model_path_obj / file_name
        if not p.exists():
            raise FileNotFoundError(f"Expected local file missing: {p}")
        return p
    # Treat as repo_id on the Hub
    logger.info(f"Loading from huggingface directory: {model_path}: {file_name}")
    return Path(hf_hub_download(repo_id=model_path, filename=file_name))
