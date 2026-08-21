#!/usr/bin/env python3
"""Check that an offline hidden-state cache completely covers a dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_from_disk
from safetensors.torch import load_file

from speculators.data_generation.offline import check_hidden_states


FILE_RE = re.compile(r"^hs_(\d+)\.safetensors$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Expected prefix length; 0 means the complete dataset.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=8,
        help="Evenly spaced files to validate; -1 validates every file, 0 skips.",
    )
    parser.add_argument("--write-ready-marker", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.data)
    expected_rows = len(dataset)
    if args.max_samples:
        if args.max_samples < 0:
            parser.error("--max-samples must be non-negative")
        expected_rows = min(expected_rows, args.max_samples)

    if "source_index" in dataset.column_names:
        raw_source_indices = dataset.with_format(None)["source_index"]
        source_indices = [
            int(value) for value in raw_source_indices[:expected_rows]
        ]
        if any(index < 0 for index in source_indices):
            parser.error("source_index values must be non-negative")
        if len(set(source_indices)) != len(source_indices):
            parser.error("source_index values must be unique")
        expected_indices = source_indices
        index_mapping = "source_index"
    else:
        expected_indices = list(range(expected_rows))
        index_mapping = "contiguous"
    expected_set = set(expected_indices)

    cache = Path(args.hidden_states)
    if not cache.is_dir():
        parser.error(f"hidden-state directory does not exist: {cache}")

    present: set[int] = set()
    partial = 0
    for entry in cache.iterdir():
        match = FILE_RE.fullmatch(entry.name)
        if match:
            present.add(int(match.group(1)))
        elif entry.name.endswith((".partial", ".lock")):
            if index_mapping == "contiguous":
                partial += 1
            else:
                # A filtered dataset is a snapshot of completed final files.
                # Ignore work in progress for rows outside that snapshot.
                match = re.search(r"hs_(\d+)\.safetensors", entry.name)
                if match and int(match.group(1)) in expected_set:
                    partial += 1

    missing = [index for index in expected_indices if index not in present]
    missing_count = len(missing)
    first_missing = missing[:20]
    out_of_range = len(present - expected_set)
    valid_present = len(present & expected_set)
    status = "complete" if missing_count == 0 and not partial else "incomplete"
    summary = {
        "status": status,
        "dataset_rows": len(dataset),
        "expected_files": expected_rows,
        "present_files": valid_present,
        "missing_files": missing_count,
        "partial_or_lock_files": partial,
        "out_of_range_files": out_of_range,
        "first_missing": first_missing,
        "index_mapping": index_mapping,
    }
    if status != "complete":
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    if args.validate_samples < -1:
        parser.error("--validate-samples must be -1 or a non-negative integer")
    count = (
        expected_rows
        if args.validate_samples == -1
        else min(args.validate_samples, expected_rows)
    )
    if count > 0:
        positions = sorted(
            {
                (offset * (expected_rows - 1)) // max(count - 1, 1)
                for offset in range(count)
            }
        )
        validated_indices = []
        for position in positions:
            file_index = expected_indices[position]
            payload = load_file(cache / f"hs_{file_index}.safetensors")
            check_hidden_states(payload, list(dataset[position]["input_ids"]))
            validated_indices.append(file_index)
        summary["validated_indices"] = validated_indices

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.write_ready_marker:
        marker = cache / ".offline-ready.json"
        temporary = cache / ".offline-ready.json.partial"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(marker)
        print(f"READY_MARKER={marker}")


if __name__ == "__main__":
    main()
