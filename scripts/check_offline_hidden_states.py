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
        help="Number of evenly spaced files to load and validate.",
    )
    parser.add_argument("--write-ready-marker", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.data)
    expected = len(dataset)
    if args.max_samples:
        if args.max_samples < 0:
            parser.error("--max-samples must be non-negative")
        expected = min(expected, args.max_samples)

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
            partial += 1

    missing_count = 0
    first_missing: list[int] = []
    for index in range(expected):
        if index not in present:
            missing_count += 1
            if len(first_missing) < 20:
                first_missing.append(index)
    out_of_range = sum(index >= expected for index in present)
    valid_present = sum(index < expected for index in present)
    status = "complete" if missing_count == 0 and not partial else "incomplete"
    summary = {
        "status": status,
        "dataset_rows": len(dataset),
        "expected_files": expected,
        "present_files": valid_present,
        "missing_files": missing_count,
        "partial_or_lock_files": partial,
        "out_of_range_files": out_of_range,
        "first_missing": first_missing,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "complete":
        raise SystemExit(1)

    count = min(args.validate_samples, expected)
    if count > 0:
        indices = sorted(
            {(offset * (expected - 1)) // max(count - 1, 1) for offset in range(count)}
        )
        for index in indices:
            payload = load_file(cache / f"hs_{index}.safetensors")
            check_hidden_states(payload, list(dataset[index]["input_ids"]))
        summary["validated_indices"] = indices

    if args.write_ready_marker:
        marker = cache / ".offline-ready.json"
        temporary = cache / ".offline-ready.json.partial"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(marker)
        print(f"READY_MARKER={marker}")


if __name__ == "__main__":
    main()
