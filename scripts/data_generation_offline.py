#!/usr/bin/env python3
"""
Offline Hidden States Generation Pipeline

This script generates hidden states and saves them to disk for offline training.

Usage:
    python data_generation_offline.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import openai
from datasets import load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm

from speculators.data_generation.offline import (
    check_hidden_states,
    get_existing_hidden_state_indices,
    get_indices_to_process,
)
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states_async,
    wait_for_lock_async,
)
from speculators.train.data import build_client_item
from speculators.train.logger import setup_root_logger

logger = logging.getLogger(__name__)


class _FailureTracker:
    """Tracks consecutive sample failures across async workers.

    When the number of consecutive failures (with no successes in between)
    reaches ``threshold``, the tracker signals that the run should abort.
    Because asyncio is single-threaded, no locking is needed.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> bool:
        """Record a failure. Returns True when the threshold is reached."""
        self._consecutive += 1
        return self._consecutive >= self.threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path for target model "
            "(default auto select). For verification purposes only."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=(
            "The address of the vLLM instance to use for hidden states generation "
            "(default: 'http://localhost:8000/v1'). "
            "Note: the vLLM instance must be configured for hidden states extraction."
        ),
    )

    # Data arguments
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        default="./output",
        help="Path to preprocessed dataset (dataset produced by prepare_data.py)"
        " (default: ./output)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory to generated hidden states files "
            "(default args.preprocessed_data / 'hidden_states')"
        ),
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help=(
            "Number of active vLLM requests at a time. "
            "Note: number of async workers set to 2*concurrency"
        ),
    )
    parser.add_argument(
        "--write-concurrency",
        type=int,
        default=1,
        help=(
            "Maximum concurrent publications to the final shared cache. Keep this "
            "small for NFS/SFS; verifier requests continue while writes run in "
            "background threads (default: 1)."
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "Multi-node scheduling policy. static assigns index %% world_size. "
            "dynamic uses atomic shared-directory claims so faster verifiers can "
            "steal unfinished work from slower nodes (default: static)."
        ),
    )
    parser.add_argument(
        "--claim-timeout",
        type=float,
        default=3600.0,
        help="Seconds after which an abandoned dynamic work claim may be recovered.",
    )
    parser.add_argument(
        "--schedule-poll-interval",
        type=float,
        default=30.0,
        help=(
            "Seconds between dynamic rescans when all remaining samples are "
            "currently claimed (default: 30)."
        ),
    )
    parser.add_argument(
        "--validate-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate token ids, shape, and finite hidden-state values before "
            "atomically publishing each file (default: enabled)"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of retry attempts per request on failure "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=(
            "Abort when a request fails after all retries. "
            "By default, failed samples are skipped."
        ),
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=None,
        help=(
            "Abort after this many consecutive sample failures (each sample "
            "already retried --max-retries times). Prevents silently churning "
            "through the entire dataset when the server is down. "
            "Ignored when --fail-on-error is set. "
            "(default: value of --concurrency)"
        ),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help=(
            "World size for multi-node data generation offline. IMPORTANT: this "
            "is the number of nodes (not the number of gpus). Defaults to 1"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help=(
            "Rank for multi-node data generation offline. IMPORTANT: this is "
            "the node index, not an index for a gpu. Must be in range[0, world_size)."
            " Defaults to 0"
        ),
    )
    return parser.parse_args()


def _dynamic_candidates(
    num_samples: int,
    max_samples: int | None,
    existing: set[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Return local-first candidates followed by other ranks' unfinished work."""
    stop = min(num_samples, max_samples) if max_samples is not None else num_samples
    shards = [
        [index for index in range(owner, stop, world_size) if index not in existing]
        for owner in range(world_size)
    ]
    order = [(rank + offset) % world_size for offset in range(world_size)]
    return [index for owner in order for index in shards[owner]]


def _try_claim_index(
    output_dir: Path, index: int, claim_timeout: float
) -> Path | None:
    """Atomically claim one output index using NFS-safe directory creation."""
    target = output_dir / f"hs_{index}.safetensors"
    if target.is_file():
        return None
    claim_root = output_dir / ".claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    claim = claim_root / f"hs_{index}.claim"
    owner_name = f"owner.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}"

    def create_owner_marker() -> Path:
        marker = claim / owner_name
        marker.write_text(f"time={time.time()}\n", encoding="utf-8")
        return marker

    try:
        claim.mkdir()
        marker = create_owner_marker()
        if target.is_file():
            _release_claim(marker)
            return None
        return marker
    except FileExistsError:
        try:
            age = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            return None
        if age <= claim_timeout:
            return None
        # Rename is atomic on a shared filesystem. Exactly one contender wins stale
        # recovery; the others observe FileNotFoundError or a fresh replacement.
        stale = claim.with_name(f"{claim.name}.stale.{uuid.uuid4().hex}")
        try:
            claim.rename(stale)
        except (FileNotFoundError, OSError):
            return None
        shutil.rmtree(stale, ignore_errors=True)
        try:
            claim.mkdir()
            return create_owner_marker()
        except FileExistsError:
            return None


def _release_claim(claim: str | Path | None) -> None:
    """Release only the claim still owned by this worker.

    Stale recovery renames the entire claim directory and creates a new one at
    the old path. A unique marker prevents a late original worker from deleting
    that new owner's claim.
    """
    if claim is None:
        return
    marker = Path(claim)
    try:
        marker.unlink()
    except FileNotFoundError:
        return
    try:
        marker.parent.rmdir()
    except OSError:
        # A new owner marker or a diagnostic file appeared; it is not ours.
        pass


def _validate_and_commit_hidden_states(
    generated_path: str | Path,
    target_path: Path,
    tokens: list[int],
    validate_outputs: bool,
) -> None:
    """Validate a generated file and atomically publish it to its final name.

    The final ``hs_<index>.safetensors`` path is never visible until validation
    and any cross-filesystem copy have completed. Failed inputs and partial
    staging files are removed so the sample remains eligible on the next run.
    """
    source = Path(generated_path)
    staged = target_path.with_name(f".{target_path.name}.partial")
    try:
        if validate_outputs:
            loaded = load_file(source)
            check_hidden_states(loaded, tokens)

        staged.unlink(missing_ok=True)
        shutil.move(str(source), str(staged))
        os.replace(staged, target_path)
    except BaseException:
        staged.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
        Path(f"{source}.lock").unlink(missing_ok=True)
        raise


async def worker(
    client,
    model: str,
    queue: "asyncio.Queue[dict[str, Any]]",
    pbar: tqdm,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    hidden_states_output_dir: Path,
    validate_outputs: bool,
    request_timeout: float | None,
    max_retries: int,
    fail_on_error: bool,
    skipped_indices: list[int],
    cancel_event: asyncio.Event,
    failure_tracker: _FailureTracker | None,
    saved_indices: list[int] | None = None,
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        claim_path = item.get("claim_path")

        # Drain remaining items quickly after cancellation
        if cancel_event.is_set():
            _release_claim(claim_path)
            queue.task_done()
            continue

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        request_started = time.monotonic()
        request_seconds = 0.0
        publish_seconds = 0.0
        try:
            async with vllm_semaphore:  # Limit number of active generate calls
                hidden_states_path = await generate_hidden_states_async(
                    client,
                    model,
                    item,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
            request_seconds = time.monotonic() - request_started
            lock_path = hidden_states_path + ".lock"
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(
                    lock_path,
                    timeout=request_timeout if request_timeout is not None else 600,
                )

            claim_was_recovered = (
                claim_path is not None
                and not Path(claim_path).is_file()  # noqa: ASYNC240
            )
            if claim_was_recovered:
                # This request exceeded claim_timeout and another collector
                # recovered it. Do not let the late original overwrite the new
                # owner's publication.
                Path(hidden_states_path).unlink(missing_ok=True)  # noqa: ASYNC240
                Path(lock_path).unlink(missing_ok=True)  # noqa: ASYNC240
                logger.warning(
                    "Discarding stale completed request for sample %d after "
                    "claim ownership changed",
                    idx,
                )
                continue

            async with write_semaphore:  # Limit number of active disk writes
                publish_started = time.monotonic()
                await asyncio.to_thread(
                    _validate_and_commit_hidden_states,
                    hidden_states_path,
                    target_hidden_states_path,
                    item["input_ids"],
                    validate_outputs,
                )
                publish_seconds = time.monotonic() - publish_started
        except Exception as e:
            if fail_on_error:
                logger.exception(
                    "Fatal: sample %d aborted with --fail-on-error: %s", idx, e
                )
                # Propagate through the normal asyncio shutdown path so this
                # sample's dynamic claim is released in ``finally``. A hard
                # os._exit left claims behind until their stale timeout and made
                # an otherwise resumable production collection appear hung.
                cancel_event.set()
                raise RuntimeError(
                    f"sample {idx} failed with --fail-on-error"
                ) from e
            logger.warning("Skipping sample %d due to error: %s", idx, e)
            skipped_indices.append(idx)
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from e
        else:
            if failure_tracker is not None:
                failure_tracker.record_success()
            if saved_indices is not None:
                saved_indices.append(idx)
            logger.info(
                "OFFLINE_SAMPLE index=%d tokens=%d request_seconds=%.2f "
                "publish_seconds=%.2f",
                idx,
                len(item["input_ids"]),
                request_seconds,
                publish_seconds,
            )
        finally:
            _release_claim(claim_path)
            pbar.update(1)
            queue.task_done()


async def _feed_queue(
    to_process,
    dataset,
    queue,
    cancel_event,
    *,
    hidden_states_output_dir: Path,
    dynamic_schedule: bool,
    claim_timeout: float,
    schedule_poll_interval: float,
):
    """Feed work, rescanning dynamic claims until the shard is complete."""
    while not cancel_event.is_set():
        pending = False
        claimed = False
        for i in to_process:
            if cancel_event.is_set():
                break

            target_path = hidden_states_output_dir / f"hs_{i}.safetensors"
            if target_path.is_file():  # noqa: ASYNC240
                continue
            pending = True

            claim_path = None
            if dynamic_schedule:
                claim_path = _try_claim_index(
                    hidden_states_output_dir, i, claim_timeout
                )
                if claim_path is None:
                    continue
            claimed = True
            try:
                dataset_item = dataset[i]
                client_item = build_client_item(dataset_item) | {
                    "idx": i,
                    "claim_path": (
                        str(claim_path) if claim_path is not None else None
                    ),
                }
            except BaseException:
                _release_claim(claim_path)
                raise

            # Check cancel_event while waiting for queue space to avoid
            # deadlocking when all workers have died.
            while not cancel_event.is_set():
                try:
                    queue.put_nowait(client_item)
                    break
                except asyncio.QueueFull:
                    await asyncio.sleep(0.1)
            if cancel_event.is_set():
                _release_claim(claim_path)
                break

        if not dynamic_schedule or not pending or cancel_event.is_set():
            break
        if not claimed:
            logger.info(
                "Dynamic scheduler waiting %.1fs for outstanding claims",
                schedule_poll_interval,
            )
        await asyncio.sleep(schedule_poll_interval if not claimed else 0)


async def _shutdown_workers(workers, queue, cancel_event):
    """Shut down workers and propagate the first real exception."""
    logger.info("Waiting for remaining file saves to complete...")
    if cancel_event.is_set():
        # Claims are acquired by the feeder before an item enters the bounded
        # queue. Release queued-but-not-started claims immediately so healthy
        # collectors can steal them instead of waiting for claim_timeout.
        while True:
            try:
                queued_item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(queued_item, dict):
                _release_claim(queued_item.get("claim_path"))
            queue.task_done()
        # Workers may be dead or draining — cancel any that are
        # still alive so we don't deadlock on sentinel puts.
        for w in workers:
            if not w.done():
                w.cancel()
    else:
        # Normal shutdown: send sentinel values so workers exit
        for _ in range(len(workers)):
            await queue.put(None)
    results = await asyncio.gather(*workers, return_exceptions=True)

    # Propagate the first real worker exception (skip CancelledError)
    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise result


async def generate_and_save_hidden_states(args, dataset):
    if args.output is None:
        hidden_states_dir = Path(args.preprocessed_data) / "hidden_states"
    else:
        hidden_states_dir = Path(args.output)
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    existing_file_indices = get_existing_hidden_state_indices(hidden_states_dir)
    num_samples = len(dataset)

    if args.schedule == "dynamic":
        to_process = _dynamic_candidates(
            num_samples,
            args.max_samples,
            set(existing_file_indices),
            args.world_size,
            args.rank,
        )
    else:
        to_process = get_indices_to_process(
            num_samples,
            args.max_samples,
            existing_file_indices,
            args.world_size,
            args.rank,
        )
    if not to_process:
        return

    logger.info(
        "Scanning %d candidate samples with schedule=%s",
        len(to_process),
        args.schedule,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(args.concurrency)
    write_semaphore = asyncio.Semaphore(args.write_concurrency)

    skipped_indices: list[int] = []
    saved_indices: list[int] = []
    cancel_event = asyncio.Event()

    max_consec = args.max_consecutive_errors
    if max_consec is None:
        max_consec = args.concurrency
    failure_tracker = _FailureTracker(max_consec) if not args.fail_on_error else None

    async with openai.AsyncOpenAI(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        list_models = await client.models.list()
        model_id = list_models.data[0].id
        if args.model and args.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({args.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )

        with tqdm(total=len(to_process)) as pbar:
            workers = [
                asyncio.create_task(
                    worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        hidden_states_dir,
                        args.validate_outputs,
                        args.request_timeout,
                        args.max_retries,
                        args.fail_on_error,
                        skipped_indices,
                        cancel_event,
                        failure_tracker,
                        saved_indices,
                    )
                )
                for _ in range(args.concurrency * 2)
            ]

            await _feed_queue(
                to_process,
                dataset,
                queue,
                cancel_event,
                hidden_states_output_dir=hidden_states_dir,
                dynamic_schedule=args.schedule == "dynamic",
                claim_timeout=args.claim_timeout,
                schedule_poll_interval=args.schedule_poll_interval,
            )
            await _shutdown_workers(workers, queue, cancel_event)

    num_saved = len(saved_indices)
    logger.info(f"Saved {num_saved} new data points to {args.output}")
    if skipped_indices:
        logger.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )


def main():
    args = parse_args()
    if int(args.rank) < 0 or int(args.rank) >= int(args.world_size):
        raise ValueError("--rank must be in range [0, world_size)")
    if args.concurrency <= 0 or args.write_concurrency <= 0:
        raise ValueError("--concurrency and --write-concurrency must be positive")
    if args.claim_timeout <= 0 or args.schedule_poll_interval <= 0:
        raise ValueError(
            "--claim-timeout and --schedule-poll-interval must be positive"
        )
    setup_root_logger()

    logger.info("EAGLE Offline Data Generation")

    dataset = load_from_disk(args.preprocessed_data)

    try:
        asyncio.run(generate_and_save_hidden_states(args, dataset))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
