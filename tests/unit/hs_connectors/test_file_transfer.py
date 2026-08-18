import errno
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from hs_connectors import transfer


def test_load_hs_file_clones_mmap_tensors(tmp_path: Path, monkeypatch):
    path = tmp_path / "sample.safetensors"
    path.touch()
    source = torch.arange(8)
    monkeypatch.setattr(transfer, "load_file", lambda _: {"token_ids": source})

    loaded = transfer._load_hs_file(path)

    assert loaded is not None
    assert torch.equal(loaded["token_ids"], source)
    assert loaded["token_ids"].data_ptr() != source.data_ptr()


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ESTALE, "Stale file handle"),
        RuntimeError("Stale file handle (os error 116)"),
    ],
)
def test_load_hs_file_retries_stale_handle(
    tmp_path: Path, monkeypatch, error: Exception
):
    path = tmp_path / "sample.safetensors"
    path.touch()
    load = Mock(side_effect=[error, {"hidden_states": torch.ones(2, 3)}])
    sleep = Mock()
    monkeypatch.setattr(transfer, "load_file", load)
    monkeypatch.setattr(transfer.time, "sleep", sleep)

    loaded = transfer._load_hs_file(path)

    assert loaded is not None
    assert torch.equal(loaded["hidden_states"], torch.ones(2, 3))
    assert load.call_count == 2
    sleep.assert_called_once_with(0.1)


def test_load_hs_file_does_not_retry_unrelated_error(tmp_path: Path, monkeypatch):
    path = tmp_path / "sample.safetensors"
    path.touch()
    load = Mock(side_effect=RuntimeError("invalid safetensors header"))
    monkeypatch.setattr(transfer, "load_file", load)

    with pytest.raises(RuntimeError, match="invalid safetensors header"):
        transfer._load_hs_file(path)

    load.assert_called_once_with(path)
