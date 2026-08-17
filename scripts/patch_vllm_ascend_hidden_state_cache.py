#!/usr/bin/env python3
"""Make Ascend MLA cache merging aware of HiddenStateCacheSpec.

vLLM's extraction cache is an MLAAttentionSpec marker subclass but does not
carry vLLM-Ascend's extra quantized-cache layout fields.  The Ascend merge
override must therefore keep extraction caches in a separate homogeneous
group instead of reading ``scale_dim`` from them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/vllm-workspace/vllm-ascend/vllm_ascend/core/kv_cache_interface.py"
)
PATCH_MARKER = "ASCEND_HIDDEN_STATE_CACHE_MERGE_FIX_V1"
METHOD_ANCHOR = """    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
"""
METHOD_REPLACEMENT = f"""    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # {PATCH_MARKER}: HiddenStateCacheSpec is an upstream MLA marker but
        # has none of AscendMLAAttentionSpec's scale/sparse layout fields.
        # A mixed collection must form separate cache groups; a homogeneous
        # extraction-cache collection can safely retain its exact spec.
        hidden_specs = [
            spec for spec in specs
            if type(spec).__name__ == "HiddenStateCacheSpec"
        ]
        if hidden_specs:
            assert len(hidden_specs) == len(specs), (
                "Hidden-state and attention caches must use separate groups."
            )
            first = hidden_specs[0]
            assert all(spec == first for spec in hidden_specs), (
                "All hidden-state layers in one cache group must use the "
                "same layout."
            )
            return first

        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-hidden-state-cache-merge-fix")


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
    if text.count(METHOD_ANCHOR) != 1:
        raise RuntimeError(
            "unsupported vLLM-Ascend source: MLA merge insertion point "
            "is not unique"
        )
    patched = text.replace(METHOD_ANCHOR, METHOD_REPLACEMENT, 1)
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
