#!/usr/bin/env python3
"""Check that an offline hidden-state cache completely covers a dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from safetensors.torch import load_file

from speculators.data_generation.offline import check_hidden_states


FILE_RE = re.compile(r"^hs_(\d+)\.safetensors$")
HEARTBEAT_SECONDS = 30.0


def _progress(phase: str, status: str, **details: object) -> None:
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {fields}" if fields else ""
    print(
        f"OFFLINE_CHECK phase={phase} status={status}{suffix}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:  # noqa: C901
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

    _progress("dataset", "started", path=args.data)
    dataset = load_from_disk(args.data)
    expected_rows = len(dataset)
    _progress("dataset", "completed", rows=expected_rows)
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
    _progress(
        "mapping",
        "completed",
        expected_files=expected_rows,
        index_mapping=index_mapping,
    )

    cache = Path(args.hidden_states)
    if not cache.is_dir():
        parser.error(f"hidden-state directory does not exist: {cache}")

    present: set[int] = set()
    partial = 0
    scanned_entries = 0
    scan_started = time.monotonic()
    last_report = scan_started
    _progress("cache_scan", "started", path=cache)
    for entry in cache.iterdir():
        scanned_entries += 1
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
        now = time.monotonic()
        if now - last_report >= HEARTBEAT_SECONDS:
            _progress(
                "cache_scan",
                "heartbeat",
                entries=scanned_entries,
                final_files=len(present),
                elapsed_seconds=f"{now - scan_started:.1f}",
            )
            last_report = now
    scan_elapsed = time.monotonic() - scan_started
    _progress(
        "cache_scan",
        "completed",
        entries=scanned_entries,
        final_files=len(present),
        elapsed_seconds=f"{scan_elapsed:.1f}",
    )

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
    _progress(
        "coverage",
        status,
        expected=expected_rows,
        present=valid_present,
        missing=missing_count,
        extra=out_of_range,
    )
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
        if count == expected_rows:
            positions = range(expected_rows)
        else:
            positions = sorted(
                {
                    (offset * (expected_rows - 1)) // max(count - 1, 1)
                    for offset in range(count)
                }
            )
        validation_total = len(positions)
        validation_started = time.monotonic()
        last_report = validation_started
        validated_indices: list[int] = []
        _progress(
            "content_validation",
            "started",
            files=validation_total,
            mode="all" if args.validate_samples == -1 else "sampled",
        )
        for completed, position in enumerate(positions, 1):
            file_index = expected_indices[position]
            try:
                payload = load_file(cache / f"hs_{file_index}.safetensors")
                check_hidden_states(payload, list(dataset[position]["input_ids"]))
            except Exception as error:  # noqa: BLE001
                raise ValueError(
                    "hidden-state validation failed: "
                    f"dataset_position={position} source_index={file_index}: {error}"
                ) from error
            del payload
            if validation_total <= 100:
                validated_indices.append(file_index)
            now = time.monotonic()
            if now - last_report >= HEARTBEAT_SECONDS:
                elapsed = now - validation_started
                _progress(
                    "content_validation",
                    "heartbeat",
                    completed=completed,
                    total=validation_total,
                    source_index=file_index,
                    elapsed_seconds=f"{elapsed:.1f}",
                    files_per_second=f"{completed / max(elapsed, 0.001):.2f}",
                )
                last_report = now
        validation_elapsed = time.monotonic() - validation_started
        summary["validated_files"] = validation_total
        if validation_total <= 100:
            summary["validated_indices"] = validated_indices
        elif validation_total:
            summary["first_validated_index"] = expected_indices[positions[0]]
            summary["last_validated_index"] = expected_indices[positions[-1]]
        _progress(
            "content_validation",
            "completed",
            files=validation_total,
            elapsed_seconds=f"{validation_elapsed:.1f}",
            files_per_second=(
                f"{validation_total / max(validation_elapsed, 0.001):.2f}"
            ),
        )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.write_ready_marker:
        marker = cache / ".offline-ready.json"
        temporary = cache / ".offline-ready.json.partial"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(marker)
        print(f"READY_MARKER={marker}")


if __name__ == "__main__":
    main()
