"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import errno
import fcntl
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


_STALE_READ_ATTEMPTS = 5
_GENERATED_FILE_APPEAR_TIMEOUT = 120.0
_GENERATED_FILE_POLL_INTERVAL = 0.1


def _is_stale_file_handle(error: BaseException) -> bool:
    return (
        isinstance(error, OSError) and error.errno == errno.ESTALE
    ) or "stale file handle" in str(error).lower()


def _is_incomplete_publication(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, (FileNotFoundError, EOFError)) or any(
        marker in message
        for marker in (
            "incomplete metadata",
            "invalid header",
            "header too small",
            "metadata incomplete",
        )
    )


def _load_hs_file(
    file_path: Path,
    *,
    appearance_timeout: float = 0.0,
) -> dict[str, torch.Tensor] | None:
    """Load a complete payload and detach it from the shared-file mmap.

    safetensors uses mmap-backed CPU tensors.  Deleting or moving the generated
    file immediately after ``load_file`` is safe on a local POSIX filesystem,
    but distributed filesystems may invalidate the still-lazy mapping with
    ESTALE. Clone every tensor before returning and retry short-lived ESTALE
    failures caused by metadata propagation on the shared filesystem. Freshly
    generated handles may additionally wait for asynchronous publication; cache
    misses remain immediate.
    """
    lock_path = str(file_path) + ".lock"
    appearance_deadline = time.monotonic() + appearance_timeout
    stale_attempt = 0
    while True:
        try:
            if Path(lock_path).exists():
                # The connector creates the lock before dispatching its async
                # safetensors save. For a freshly generated handle, allow the
                # writer the same bounded publication window as file appearance.
                remaining = appearance_deadline - time.monotonic()
                wait_for_lock(
                    lock_path,
                    timeout=max(10.0, remaining) if appearance_timeout else 10.0,
                )

            if not file_path.exists():
                # vLLM returns the request handle before a remote filesystem is
                # required to expose the newly-created lock/file to another
                # host. Cached reads must remain non-blocking, but generated
                # handles get a bounded metadata-propagation window.
                if appearance_timeout and time.monotonic() < appearance_deadline:
                    time.sleep(_GENERATED_FILE_POLL_INTERVAL)
                    continue
                return None

            mmap_tensors = load_file(file_path)
            return {name: tensor.clone() for name, tensor in mmap_tensors.items()}
        except Exception as error:  # noqa: BLE001 - filesystem boundary
            if (
                appearance_timeout
                and time.monotonic() < appearance_deadline
                and _is_incomplete_publication(error)
            ):
                time.sleep(_GENERATED_FILE_POLL_INTERVAL)
                continue
            if not _is_stale_file_handle(error):
                raise
            stale_attempt += 1
            if stale_attempt == _STALE_READ_ATTEMPTS:
                raise
            time.sleep(0.1 * stale_attempt)


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(
            Path(handle),
            appearance_timeout=_GENERATED_FILE_APPEAR_TIMEOUT,
        )

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }
