#!/usr/bin/env python3
"""Build a trainable Hugging Face dataset from a partial offline cache.

The generated dataset contains only rows whose ``hs_<index>.safetensors`` file
already exists.  A ``source_index`` column preserves the original cache index,
so hidden-state files do not need to be copied or renamed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from datasets import load_from_disk
from safetensors.torch import load_file

from speculators.data_generation.offline import check_hidden_states


FILE_RE = re.compile(r"^hs_(\d+)\.safetensors$")
MANIFEST_NAME = "partial_offline_manifest.json"


def _cache_indices(cache: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for entry in cache.iterdir()
        if (match := FILE_RE.fullmatch(entry.name)) is not None
    )


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count < -1:
        raise ValueError("--validate-samples must be -1 or a non-negative integer")
    if count == 0 or not values:
        return []
    if count == -1 or count >= len(values):
        return values
    return [
        values[(offset * (len(values) - 1)) // max(count - 1, 1)]
        for offset in range(count)
    ]


def build_partial_dataset(
    data_path: Path,
    cache: Path,
    output: Path,
    *,
    validate_samples: int,
) -> dict[str, object]:
    if not data_path.is_dir():
        raise ValueError(f"source dataset does not exist: {data_path}")
    if not cache.is_dir():
        raise ValueError(f"hidden-state directory does not exist: {cache}")
    if output.exists():
        raise ValueError(
            f"output already exists: {output}; choose a new OUTPUT_DATA path"
        )

    dataset = load_from_disk(str(data_path))
    cache_indices = _cache_indices(cache)
    if not cache_indices:
        raise ValueError(f"no hs_<index>.safetensors files found in {cache}")

    invalid_indices = [index for index in cache_indices if index >= len(dataset)]
    if invalid_indices:
        preview = invalid_indices[:20]
        raise ValueError(
            "hidden-state indices exceed the source dataset length; this cache "
            f"does not match --data (dataset rows={len(dataset)}, indices={preview})"
        )

    validated_indices = _evenly_spaced(cache_indices, validate_samples)
    for index in validated_indices:
        try:
            payload = load_file(cache / f"hs_{index}.safetensors")
            check_hidden_states(payload, list(dataset[index]["input_ids"]))
        except Exception as error:  # noqa: BLE001
            raise ValueError(
                f"hidden-state validation failed for source index {index}: {error}"
            ) from error

    filtered = dataset.select(cache_indices)
    if "source_index" in filtered.column_names:
        # Cache filenames are indexed against the dataset passed to this script,
        # even when that dataset was itself derived from an older source.
        filtered = filtered.remove_columns("source_index")
    filtered = filtered.add_column("source_index", cache_indices)

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data": str(data_path.resolve()),
        "hidden_states": str(cache.resolve()),
        "source_rows": len(dataset),
        "selected_rows": len(cache_indices),
        "first_source_index": cache_indices[0],
        "last_source_index": cache_indices[-1],
        "validated_files": len(validated_indices),
        "index_column": "source_index",
        "hidden_states_copied": False,
    }
    if len(validated_indices) <= 100:
        manifest["validated_indices"] = validated_indices
    elif validated_indices:
        manifest["first_validated_index"] = validated_indices[0]
        manifest["last_validated_index"] = validated_indices[-1]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid4().hex}"
    try:
        filtered.save_to_disk(str(staging))
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--hidden-states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=32,
        help="Evenly spaced files to validate; -1 validates every file, 0 skips.",
    )
    args = parser.parse_args()

    try:
        manifest = build_partial_dataset(
            args.data,
            args.hidden_states,
            args.output,
            validate_samples=args.validate_samples,
        )
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"OUTPUT_DATA={args.output}")
    print(f"HIDDEN_STATES={args.hidden_states}")


if __name__ == "__main__":
    main()
