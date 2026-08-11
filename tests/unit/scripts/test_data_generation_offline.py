import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "data_generation_offline.py"


def load_data_generation_offline_module():
    spec = importlib.util.spec_from_file_location("data_generation_offline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Progress:
    def update(self, _amount):
        pass


class DataGenerationOfflineWorkerTest(unittest.IsolatedAsyncioTestCase):
    def test_validate_and_commit_publishes_only_finite_files(self):
        module = load_data_generation_offline_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            generated = tmp_path / "generated.safetensors"
            target = tmp_path / "hs_7.safetensors"
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 3]),
                    "hidden_states": torch.zeros(3, 2, 4),
                },
                generated,
            )

            module._validate_and_commit_hidden_states(
                generated, target, [1, 2, 3], validate_outputs=True
            )

            assert target.is_file()
            assert not generated.exists()
            assert not (tmp_path / ".hs_7.safetensors.partial").exists()

    def test_validate_and_commit_removes_nonfinite_source_and_partial(self):
        module = load_data_generation_offline_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            generated = tmp_path / "generated.safetensors"
            target = tmp_path / "hs_9.safetensors"
            hidden_states = torch.zeros(3, 2, 4)
            hidden_states[1, 0, 0] = torch.nan
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 3]),
                    "hidden_states": hidden_states,
                },
                generated,
            )

            with pytest.raises(ValueError, match="non-finite"):
                module._validate_and_commit_hidden_states(
                    generated, target, [1, 2, 3], validate_outputs=True
                )

            assert not target.exists()
            assert not generated.exists()
            assert not (tmp_path / ".hs_9.safetensors.partial").exists()

    async def test_worker_uses_request_timeout_while_waiting_for_hidden_state_file(
        self,
    ):
        module = load_data_generation_offline_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            generated = tmp_path / "generated.safetensors"
            generated.touch()
            Path(f"{generated}.lock").touch()  # noqa: ASYNC240
            output_path = tmp_path / "output"
            output_path.mkdir()
            observed = {}

            async def fake_generate(*_args, **_kwargs):
                return str(generated)

            async def fake_wait(lock_path, timeout=None):
                observed["lock_path"] = lock_path
                observed["timeout"] = timeout

            async def run_file_operation_inline(function, *args, **kwargs):
                return function(*args, **kwargs)

            queue = asyncio.Queue()
            queue.put_nowait({"idx": 131, "input_ids": [1, 2, 3]})
            queue.put_nowait(None)

            with (
                patch.object(module, "generate_hidden_states_async", fake_generate),
                patch.object(module, "wait_for_lock_async", fake_wait),
                patch.object(module.asyncio, "to_thread", run_file_operation_inline),
            ):
                await module.worker(
                    client=object(),
                    model="glm-5.2",
                    queue=queue,
                    pbar=_Progress(),
                    vllm_semaphore=asyncio.Semaphore(1),
                    write_semaphore=asyncio.Semaphore(1),
                    hidden_states_output_dir=output_path,
                    validate_outputs=False,
                    request_timeout=600,
                    max_retries=1,
                    fail_on_error=False,
                    skipped_indices=[],
                    cancel_event=asyncio.Event(),
                    failure_tracker=None,
                )

            assert observed == {
                "lock_path": f"{generated}.lock",
                "timeout": 600,
            }
            assert (output_path / "hs_131.safetensors").is_file()


if __name__ == "__main__":
    unittest.main()
