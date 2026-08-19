#!/usr/bin/env python3
"""Prepare a selected Nuoya JSONL mixture as a resumable HF dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SOURCES = (
    "/mnt/xds/mtp/spec_train/dataset/raw_conversions/average-2k-nuoya",
    "/mnt/xds/mtp/spec_train/dataset/raw_conversions/average-8k",
)
DEFAULT_OUTPUT = (
    "/mnt/xds/mtp/spec_train/dataset/hf/nuoya-average2k8k-32k"
)
DEFAULT_MODEL = (
    "/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize Nuoya JSONL files independently, resume completed shards, "
            "and atomically publish one Hugging Face dataset."
        )
    )
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument(
        "--source-file-limit",
        action="append",
        type=int,
        dest="source_file_limits",
        help=(
            "Select only the first N sorted JSON/JSONL files from the matching "
            "--source. Repeat once per --source. Omit to use every file."
        ),
    )
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Discover only .jsonl files inside source directories.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("MTP_INIT_MODEL_PATH", DEFAULT_MODEL)
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--staging-dir")
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument("--num-preprocessing-workers", type=int, default=8)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-valid-tokens", type=int, default=1)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Move an existing final output aside before publishing the new dataset.",
    )
    return parser.parse_args()


def discover_files(
    sources: list[str],
    source_file_limits: list[int] | None = None,
    *,
    jsonl_only: bool = False,
) -> list[Path]:
    if source_file_limits is not None and len(source_file_limits) != len(sources):
        raise ValueError(
            "--source-file-limit must be repeated exactly once per --source"
        )
    files: list[Path] = []
    for index, value in enumerate(sources):
        source = Path(value)
        allowed_suffixes = {".jsonl"} if jsonl_only else {".json", ".jsonl"}
        if source.is_file() and source.suffix in allowed_suffixes:
            discovered = [source]
        elif source.is_dir():
            discovered = list(source.rglob("*.jsonl"))
            if not jsonl_only:
                discovered.extend(source.rglob("*.json"))
            discovered.sort()
        else:
            raise FileNotFoundError(f"source does not exist or is not JSON: {source}")
        if source_file_limits is not None:
            limit = source_file_limits[index]
            if limit <= 0:
                raise ValueError("--source-file-limit values must be positive")
            if len(discovered) < limit:
                raise ValueError(
                    f"source {source} has {len(discovered)} JSON/JSONL files, "
                    f"fewer than requested limit {limit}"
                )
            discovered = discovered[:limit]
        files.extend(discovered)
    resolved = [path.resolve() for path in files]
    if not resolved:
        raise ValueError("no JSON/JSONL files were discovered")
    if len(resolved) != len(set(resolved)):
        raise ValueError("the source paths contain duplicate JSON/JSONL files")
    return resolved


def validate_first_row(path: Path) -> None:
    if path.suffix == ".json":
        return
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            supported = (
                "conversations" in row
                or "messages" in row
                or {"input_ids", "loss_mask"} <= set(row)
            )
            if not supported:
                raise ValueError(
                    f"{path}:{line_number} needs conversations, messages, or "
                    "pre-tokenized input_ids/loss_mask"
                )
            return
    raise ValueError(f"empty JSONL file: {path}")


def shard_name(index: int, source: Path) -> str:
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in source.stem)
    return f"{index:03d}-{safe_stem}"


def prepare_shard(
    source: Path,
    destination: Path,
    args: argparse.Namespace,
    prepare_script: Path,
) -> None:
    from datasets import load_from_disk  # noqa: PLC0415

    marker = destination / ".complete"
    if marker.is_file():
        dataset = load_from_disk(str(destination))
        print(f"[reuse] {source} -> {destination} rows={len(dataset)}", flush=True)
        return
    # This directory is exclusively owned by this script.  An absent marker
    # means prepare_data or save_to_disk was interrupted; partial Arrow files
    # must not be mistaken for a reusable dataset on the next run.
    if destination.exists():
        print(f"[restart] removing incomplete shard: {destination}", flush=True)
        shutil.rmtree(destination)

    command = [
        sys.executable,
        str(prepare_script),
        "--model",
        args.model,
        "--data",
        str(source),
        "--output",
        str(destination),
        "--seq-length",
        str(args.seq_length),
        "--num-preprocessing-workers",
        str(args.num_preprocessing_workers),
        "--minimum-valid-tokens",
        str(args.minimum_valid_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    print("[prepare] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    dataset = load_from_disk(str(destination))
    if len(dataset) == 0:
        raise RuntimeError(f"prepared shard is empty: {destination}")
    marker.write_text(f"source={source}\nrows={len(dataset)}\n", encoding="utf-8")


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def collect_stats(dataset) -> dict[str, int | float]:
    lengths = [int(value) for value in dataset.with_format(None)["seq_len"]]
    lengths.sort()
    total_tokens = sum(lengths)
    # One BF16 target layer, GLM-5.2 hidden size 6144.
    hidden_state_bytes = total_tokens * 6144 * 2
    return {
        "rows": len(lengths),
        "total_tokens": total_tokens,
        "mean_seq_len": total_tokens / len(lengths) if lengths else 0.0,
        "p50_seq_len": percentile(lengths, 0.50),
        "p90_seq_len": percentile(lengths, 0.90),
        "p99_seq_len": percentile(lengths, 0.99),
        "max_seq_len": lengths[-1] if lengths else 0,
        "estimated_bf16_hidden_state_bytes_one_layer": hidden_state_bytes,
        "estimated_bf16_hidden_state_tib_one_layer": hidden_state_bytes / (1024**4),
    }


def publish_dataset(
    shard_paths: list[Path], sources: list[Path], args: argparse.Namespace
) -> None:
    from datasets import concatenate_datasets, load_from_disk  # noqa: PLC0415

    from speculators.train.vocab_mapping import (  # noqa: PLC0415
        combine_token_frequency_distributions,
    )

    output = Path(args.output)
    if output.exists():
        if not args.overwrite_output:
            dataset = load_from_disk(str(output))
            print(f"[done] output already exists: {output} rows={len(dataset)}")
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup = output.with_name(f"{output.name}.backup-{timestamp}")
        output.rename(backup)
        print(f"[backup] {output} -> {backup}")

    datasets = [load_from_disk(str(path)) for path in shard_paths]
    combined = concatenate_datasets(datasets).shuffle(seed=args.seed)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    print(f"[merge] shards={len(datasets)} rows={len(combined)} -> {temporary}")
    combined.save_to_disk(
        str(temporary),
        max_shard_size=args.max_shard_size,
        num_proc=args.save_workers,
    )

    token_freq_paths = [path / "token_freq.pt" for path in shard_paths]
    missing_freq = [str(path) for path in token_freq_paths if not path.is_file()]
    if missing_freq:
        raise RuntimeError(f"missing per-shard token frequency files: {missing_freq}")
    combine_token_frequency_distributions(
        token_freq_paths, temporary / "token_freq.pt"
    )
    stats = collect_stats(combined)
    manifest = {
        "sources": [str(path) for path in sources],
        "model": args.model,
        "seq_length": args.seq_length,
        "seed": args.seed,
        "stats": stats,
    }
    (temporary / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary / ".ready").write_text("ok\n", encoding="utf-8")
    temporary.rename(output)
    print(json.dumps({"output": str(output), **stats}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.seq_length <= 0:
        raise ValueError("--seq-length must be positive")
    if args.num_preprocessing_workers <= 0 or args.save_workers <= 0:
        raise ValueError("worker counts must be positive")
    source_values = args.sources or list(DEFAULT_SOURCES)
    if args.source_file_limits and not args.sources:
        raise ValueError("--source-file-limit requires explicit --source values")
    sources = discover_files(
        source_values,
        args.source_file_limits,
        jsonl_only=args.jsonl_only,
    )
    for source in sources:
        validate_first_row(source)
    output = Path(args.output)
    staging = Path(args.staging_dir or f"{output}.staging")
    staging.mkdir(parents=True, exist_ok=True)
    prepare_script = Path(__file__).resolve().with_name("prepare_data.py")
    shard_paths = []
    for index, source in enumerate(sources):
        destination = staging / shard_name(index, source)
        prepare_shard(source, destination, args, prepare_script)
        shard_paths.append(destination)
    publish_dataset(shard_paths, sources, args)


if __name__ == "__main__":
    main()
