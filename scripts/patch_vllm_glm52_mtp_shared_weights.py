#!/usr/bin/env python3
"""Patch vLLM's native MTP loader to load GLM shared embedding/head weights.

The GLM-5.2 checkpoint stores the shared tensors at the target-model level as
``model.embed_tokens.weight`` and ``lm_head.weight``.  DeepSeekMTP.load_weights
normally filters out weights which are not under an MTP layer before applying
its name rewrite, leaving the drafter embedding/head randomly initialized.

This patch is idempotent, creates a byte-for-byte backup, validates the exact
upstream insertion points, and compiles the result before replacing the file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/vllm-workspace/vllm/vllm/model_executor/models/deepseek_mtp.py"
)
PATCH_MARKER = "GLM_MTP_SHARED_WEIGHT_FIX_V1"

LOOP_ANCHOR = """        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
"""

LOOP_REPLACEMENT = f"""        for name, loaded_weight in weights:
            # {PATCH_MARKER}: GLM checkpoints keep the MTP shared embedding
            # and logits head at target-model level.  Route them before the
            # spec-layer filter, which would otherwise silently skip both.
            shared_mtp_name = None
            if name == "model.embed_tokens.weight":
                shared_mtp_name = "model.embed_tokens.weight"
            elif name == "lm_head.weight":
                shared_mtp_name = (
                    f"model.layers.{{self.model.mtp_start_layer_idx}}."
                    "shared_head.head.weight"
                )
            if shared_mtp_name is not None:
                if shared_mtp_name not in params_dict:
                    raise ValueError(
                        "MTP shared-weight destination is missing: "
                        f"{{shared_mtp_name}}"
                    )
                param = params_dict[shared_mtp_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(shared_mtp_name)
                continue

            if "rotary_emb.inv_freq" in name:
"""

VALIDATION_ANCHOR = (
    "        # Validate that weights were loaded for each expected MTP layer.\n"
    "        loaded_layers: set[int] = set()\n"
)

VALIDATION_REPLACEMENT = f"""        # {PATCH_MARKER}: fail on missing shared weights.
        # continue with a random embedding or logits head.
        required_shared_params = {{
            "model.embed_tokens.weight",
            (
                f"model.layers.{{self.model.mtp_start_layer_idx}}."
                "shared_head.head.weight"
            ),
        }}
        missing_shared_params = (
            required_shared_params.intersection(params_dict) - loaded_params
        )
        if missing_shared_params:
            raise ValueError(
                "MTP shared weights were not initialized from checkpoint: "
                f"{{sorted(missing_shared_params)}}"
            )

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report patch status")
    mode.add_argument("--restore", action="store_true", help="restore the backup")
    return parser.parse_args()


def backup_path(target: Path) -> Path:
    return target.with_name(target.name + ".before-glm-mtp-shared-weight-fix")


def validate_original(text: str) -> None:
    if text.count(LOOP_ANCHOR) != 1:
        raise RuntimeError(
            "unsupported vLLM source: MTP load loop insertion point is not unique"
        )
    if text.count(VALIDATION_ANCHOR) != 1:
        raise RuntimeError(
            "unsupported vLLM source: MTP validation insertion point is not unique"
        )


def build_patched(text: str, filename: str) -> str:
    validate_original(text)
    patched = text.replace(LOOP_ANCHOR, LOOP_REPLACEMENT, 1)
    patched = patched.replace(VALIDATION_ANCHOR, VALIDATION_REPLACEMENT, 1)
    if patched.count(PATCH_MARKER) != 2:
        raise RuntimeError("internal error: patch markers were not inserted twice")
    compile(patched, filename, "exec")
    return patched


def atomic_write(target: Path, text: str) -> None:
    original_mode = target.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def check(target: Path) -> int:
    if not target.is_file():
        print(f"PATCH_STATUS=target-missing\nTARGET={target}")
        return 2
    text = target.read_text(encoding="utf-8")
    count = text.count(PATCH_MARKER)
    status = "applied" if count == 2 else "not-applied" if count == 0 else "partial"
    print(f"PATCH_STATUS={status}")
    print(f"TARGET={target}")
    print(f"BACKUP={backup_path(target)}")
    return 0 if count == 2 else 1


def apply(target: Path) -> int:
    if not target.is_file():
        raise FileNotFoundError(target)
    text = target.read_text(encoding="utf-8")
    if text.count(PATCH_MARKER) == 2:
        print(f"PATCH_STATUS=already-applied\nTARGET={target}")
        return 0
    if PATCH_MARKER in text:
        raise RuntimeError("refusing to modify a partially patched source file")

    patched = build_patched(text, str(target))
    backup = backup_path(target)
    if backup.exists():
        if backup.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"existing backup does not match current source: {backup}"
            )
    else:
        shutil.copy2(target, backup)
    atomic_write(target, patched)
    print("PATCH_STATUS=applied")
    print(f"TARGET={target}")
    print(f"BACKUP={backup}")
    print("RESTART_REQUIRED=yes")
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
