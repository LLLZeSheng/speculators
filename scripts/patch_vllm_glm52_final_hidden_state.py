#!/usr/bin/env python3
"""Teach vLLM's DeepSeek/GLM model to expose the final normalized state.

``launch_vllm.py`` represents the final verifier state with layer id
``num_hidden_layers``.  The vLLM 0.23 DeepSeekV2 implementation only samples
states before decoder blocks, whose ids stop at ``num_hidden_layers - 1``.
Consequently GLM-5.2 returns a plain tensor while the runner expects
``(hidden_states, aux_hidden_states)``.

This patch is intentionally narrow and idempotent.  It validates the exact
source fragment, creates a backup, compiles the result, and supports checking
and restoring the installed source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py"
)
PATCH_MARKER = "GLM_FINAL_AUX_HIDDEN_STATE_FIX_V1"
ORIGINAL = """        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
REPLACEMENT = f"""        hidden_states, _ = self.norm(hidden_states, residual)
        # {PATCH_MARKER}: layer id ``end_layer`` denotes the normalized
        # output after the final decoder block.  Decoder-loop ids only reach
        # ``end_layer - 1``, so collect this state explicitly.
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-glm-final-aux-hidden-state-fix")


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
            "unsupported vLLM source: final hidden-state insertion point "
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
