#!/usr/bin/env python3
"""Restore TP-sharded hidden states at the connector's save boundary.

The request metadata contains the complete prompt token ids, while Ascend may
expose only the local sequence shard through the cache-only layer.  Repair the
tensor where both lengths are visible, before a short safetensors file can be
published to an online trainer.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/"
    "example_hidden_states_connector.py"
)
PATCH_MARKER = "SPECULATORS_HS_CONNECTOR_TP_GATHER_V1"
IMPORT_ANCHOR = "from vllm.forward_context import get_forward_context\n"
IMPORT_REPLACEMENT = (
    "from vllm.distributed import tensor_model_parallel_all_gather\n"
    + IMPORT_ANCHOR
)
ORIGINAL = """                hidden_states_gpu = extract_from_kv_cache(
                    kv_layer, req_slot_mapping_gpu, num_tokens
                )
                # Async DtoH copy into pinned host memory.
"""
REPLACEMENT = f"""                hidden_states_gpu = extract_from_kv_cache(
                    kv_layer, req_slot_mapping_gpu, num_tokens
                )
                # {PATCH_MARKER}: request.token_ids is the authoritative full
                # prompt length. Ascend sequence parallelism can leave only a
                # local token shard in the cache-only layer.
                if hidden_states_gpu.shape[0] < num_tokens:
                    local_tokens = hidden_states_gpu.shape[0]
                    hidden_states_gpu = tensor_model_parallel_all_gather(
                        hidden_states_gpu.contiguous(), dim=0
                    )
                    if (
                        hidden_states_gpu.shape[0] > num_tokens
                        and hidden_states_gpu.shape[0] % num_tokens == 0
                    ):
                        copies = hidden_states_gpu.shape[0] // num_tokens
                        full_copies = hidden_states_gpu.reshape(
                            copies, num_tokens, *hidden_states_gpu.shape[1:]
                        )
                        if all(
                            torch.equal(full_copies[0], full_copies[idx])
                            for idx in range(1, copies)
                        ):
                            hidden_states_gpu = full_copies[0]
                        elif num_tokens % local_tokens == 0:
                            unique_shards = num_tokens // local_tokens
                            rank_shards = hidden_states_gpu.reshape(
                                unique_shards,
                                copies,
                                local_tokens,
                                *hidden_states_gpu.shape[1:],
                            )
                            if all(
                                torch.equal(rank_shards[:, 0], rank_shards[:, idx])
                                for idx in range(1, copies)
                            ):
                                hidden_states_gpu = rank_shards[:, 0].reshape(
                                    num_tokens, *hidden_states_gpu.shape[1:]
                                )
                    if hidden_states_gpu.shape[0] != num_tokens:
                        raise RuntimeError(
                            "hidden-state connector TP gather produced an "
                            f"invalid token dimension: local={{local_tokens}}, "
                            f"gathered={{hidden_states_gpu.shape[0]}}, "
                            f"expected={{num_tokens}}"
                        )
                    logger.warning_once(
                        "Restored connector TP-sharded hidden states: "
                        "local_tokens=%d full_tokens=%d",
                        local_tokens,
                        num_tokens,
                    )
                if hidden_states_gpu.shape[0] != num_tokens:
                    raise RuntimeError(
                        "hidden-state connector received an invalid token "
                        f"dimension: actual={{hidden_states_gpu.shape[0]}}, "
                        f"expected={{num_tokens}}"
                    )
                # Async DtoH copy into pinned host memory.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-connector-tp-gather")


def atomic_write(target: Path, text: str) -> None:
    mode = target.stat().st_mode
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def check(target: Path) -> int:
    if not target.is_file():
        print(f"PATCH_STATUS=target-missing\nTARGET={target}")
        return 2
    count = target.read_text(encoding="utf-8").count(PATCH_MARKER)
    status = "applied" if count == 1 else "not-applied" if count == 0 else "partial"
    print(f"PATCH_STATUS={status}\nTARGET={target}\nBACKUP={backup_path(target)}")
    return 0 if count == 1 else 1


def apply(target: Path) -> int:
    if not target.is_file():
        raise FileNotFoundError(target)
    text = target.read_text(encoding="utf-8")
    if text.count(PATCH_MARKER) == 1:
        print(f"PATCH_STATUS=already-applied\nTARGET={target}")
        return 0
    if PATCH_MARKER in text:
        raise RuntimeError("refusing to modify a partially patched source file")
    for name, anchor in (("import", IMPORT_ANCHOR), ("save", ORIGINAL)):
        if text.count(anchor) != 1:
            raise RuntimeError(
                f"unsupported vLLM source: {name} anchor is not unique"
            )
    patched = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(ORIGINAL, REPLACEMENT, 1)
    compile(patched, str(target), "exec")
    backup = backup_path(target)
    if backup.exists():
        if backup.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"existing backup does not match source: {backup}")
    else:
        shutil.copy2(target, backup)
    atomic_write(target, patched)
    print(
        f"PATCH_STATUS=applied\nTARGET={target}\nBACKUP={backup}"
        "\nRESTART_REQUIRED=yes"
    )
    return 0


def restore(target: Path) -> int:
    backup = backup_path(target)
    if not backup.is_file():
        raise FileNotFoundError(f"backup not found: {backup}")
    original = backup.read_text(encoding="utf-8")
    compile(original, str(target), "exec")
    atomic_write(target, original)
    print(f"PATCH_STATUS=restored\nTARGET={target}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            return check(args.target)
        if args.restore:
            return restore(args.target)
        return apply(args.target)
    except Exception as error:  # noqa: BLE001 - command boundary
        print(f"PATCH_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
