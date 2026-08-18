#!/usr/bin/env python3
"""Restore TP-sharded verifier states before Ascend caches them for export.

vLLM-Ascend 0.23 can leave auxiliary hidden states sequence-sharded even when
the extract-hidden-states connector expects one full-token tensor per worker.
The connector then writes a short payload (for example 128 states for a
1024-token prompt).  Patch the proposer boundary to gather only when the
observed token dimension is shorter than the scheduler's global token count.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
)
PATCH_MARKER = "SPECULATORS_ASCEND_EXTRACT_HS_TP_GATHER_V1"
ORIGINAL = """            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            if vllm_version_is("0.23.0"):
"""
REPLACEMENT = f"""            # {PATCH_MARKER}: the Ascend MoE/SP path may return a
            # token-sharded auxiliary tensor even though the cache-only layer
            # and ExampleHiddenStatesConnector require the complete prompt on
            # every TP worker.  Repair only an observed short tensor; keeping
            # the full-size fast path avoids an unnecessary collective.
            target_hidden_states = []
            for aux_state in aux_hidden_states:
                state = aux_state[:num_scheduled_tokens]
                if state.shape[0] < num_scheduled_tokens:
                    local_tokens = state.shape[0]
                    state = tensor_model_parallel_all_gather(
                        state.contiguous(), dim=0
                    )
                    if state.shape[0] != num_scheduled_tokens:
                        raise RuntimeError(
                            "extract_hidden_states TP gather produced an "
                            f"invalid token dimension: local={{local_tokens}}, "
                            f"gathered={{state.shape[0]}}, "
                            f"expected={{num_scheduled_tokens}}"
                        )
                    logger.warning_once(
                        "Restored TP-sharded extract_hidden_states tensor: "
                        "local_tokens=%d full_tokens=%d",
                        local_tokens,
                        num_scheduled_tokens,
                    )
                if state.shape[0] != num_scheduled_tokens:
                    raise RuntimeError(
                        "extract_hidden_states received an invalid token "
                        f"dimension: actual={{state.shape[0]}}, "
                        f"expected={{num_scheduled_tokens}}"
                    )
                target_hidden_states.append(state)

            if vllm_version_is("0.23.0"):
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-extract-hs-tp-gather")


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
    if text.count(ORIGINAL) != 1:
        raise RuntimeError(
            "unsupported vLLM-Ascend source: extract-hidden-states anchor "
            "is not unique"
        )
    patched = text.replace(ORIGINAL, REPLACEMENT, 1)
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
