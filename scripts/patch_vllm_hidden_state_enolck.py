#!/usr/bin/env python3
"""Fall back to synchronous hidden-state writes when flock is unavailable.

Some shared filesystems return ENOLCK for ``flock``.  vLLM's example hidden-
state connector treats that as fatal, which kills EngineCore.  Merely disabling
the lock is unsafe for online consumers because the response can expose the
path before the asynchronous safetensors write finishes.  This patch switches
the connector to synchronous writes after the first ENOLCK, so the returned
path always names a complete file.
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
PATCH_MARKER = "SPECULATORS_HS_ENOLCK_SYNC_FALLBACK_V1"
IMPORT_ANCHOR = "import fcntl\nimport os\n"
IMPORT_REPLACEMENT = "import errno\nimport fcntl\nimport os\n"
INIT_ANCHOR = """        self.use_lock = self._kv_transfer_config.get_from_extra_config(
            "use_synchronization_lock", True
        )
"""
INIT_REPLACEMENT = INIT_ANCHOR + f"""        # {PATCH_MARKER}
        self._synchronous_write_fallback = False
"""
FLOCK_ANCHOR = "                fcntl.flock(lock_fd, fcntl.LOCK_EX)\n"
FUTURE_ANCHOR = "            future = self._executor.submit(\n"
USE_LOCK_ANCHOR = "            if self.use_lock:\n"
USE_LOCK_REPLACEMENT = (
    "            if self.use_lock and not self._synchronous_write_fallback:\n"
)
FLOCK_REPLACEMENT = """                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except OSError as error:
                    if error.errno != errno.ENOLCK:
                        raise
                    os.close(lock_fd)
                    lock_fd = None
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass
                    self._synchronous_write_fallback = True
                    logger.warning(
                        "Filesystem locking is unavailable for %s; "
                        "falling back to synchronous hidden-state writes.",
                        self._storage_path,
                    )
"""
FUTURE_REPLACEMENT = """            if self._synchronous_write_fallback:
                self._write_tensors(tensors, event, filename, None)
                self._req_copy_events[req_id] = event
                continue
            future = self._executor.submit(
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-enolck-sync-fallback")


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
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("initialization", INIT_ANCHOR),
        ("use-lock", USE_LOCK_ANCHOR),
        ("flock", FLOCK_ANCHOR),
        ("future", FUTURE_ANCHOR),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"unsupported vLLM source: {name} anchor is not unique")
    patched = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(INIT_ANCHOR, INIT_REPLACEMENT, 1)
    patched = patched.replace(USE_LOCK_ANCHOR, USE_LOCK_REPLACEMENT, 1)
    patched = patched.replace(FLOCK_ANCHOR, FLOCK_REPLACEMENT, 1)
    patched = patched.replace(FUTURE_ANCHOR, FUTURE_REPLACEMENT, 1)
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
