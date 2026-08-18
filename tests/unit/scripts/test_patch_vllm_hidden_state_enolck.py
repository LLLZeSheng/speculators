from pathlib import Path

from scripts.patch_vllm_hidden_state_enolck import PATCH_MARKER, apply, check, restore


SOURCE = '''import fcntl
import os

class Connector:
    def __init__(self):
        self.use_lock = self._kv_transfer_config.get_from_extra_config(
            "use_synchronization_lock", True
        )

    def wait_for_save(self):
        for tensors, event, filename, req_id in self._pending_copies:
            lock_fd = None
            if self.use_lock:
                lock_path = filename + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            future = self._executor.submit(
                self._write_tensors, tensors, event, filename, lock_fd
            )
'''


def test_patch_apply_check_restore(tmp_path: Path):
    target = tmp_path / "example_hidden_states_connector.py"
    target.write_text(SOURCE)

    assert apply(target) == 0
    patched = target.read_text()
    assert patched.count(PATCH_MARKER) == 1
    assert "if self.use_lock and not self._synchronous_write_fallback:" in patched
    assert "error.errno != errno.ENOLCK" in patched
    assert "self._write_tensors(tensors, event, filename, None)" in patched
    assert check(target) == 0
    assert apply(target) == 0

    assert restore(target) == 0
    assert target.read_text() == SOURCE
