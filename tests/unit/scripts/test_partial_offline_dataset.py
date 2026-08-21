import json
import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset, load_from_disk
from safetensors.torch import save_file

from scripts.build_partial_offline_dataset import build_partial_dataset
from scripts.check_offline_hidden_states import main as check_offline_main


def _source_dataset(path: Path) -> list[list[int]]:
    input_ids = [[index, index + 10, index + 20] for index in range(5)]
    Dataset.from_dict(
        {
            "input_ids": input_ids,
            "loss_mask": [[1, 1, 1]] * len(input_ids),
            "seq_len": [3] * len(input_ids),
        }
    ).save_to_disk(str(path))
    return input_ids


def _save_hidden_states(path: Path, index: int, tokens: list[int]) -> None:
    save_file(
        {
            "token_ids": torch.tensor(tokens),
            "hidden_states": torch.zeros(len(tokens), 2, 4),
        },
        path / f"hs_{index}.safetensors",
    )


def test_build_partial_dataset_preserves_original_cache_indices(tmp_path: Path):
    data_path = tmp_path / "source"
    cache = tmp_path / "cache"
    output = tmp_path / "filtered"
    tokens = _source_dataset(data_path)
    cache.mkdir()
    _save_hidden_states(cache, 1, tokens[1])
    _save_hidden_states(cache, 4, tokens[4])
    (cache / ".hs_2.safetensors.partial").touch()

    manifest = build_partial_dataset(data_path, cache, output, validate_samples=-1)

    filtered = load_from_disk(str(output)).with_format(None)
    assert list(filtered["input_ids"]) == [tokens[1], tokens[4]]
    assert list(filtered["source_index"]) == [1, 4]
    assert manifest["selected_rows"] == 2
    assert manifest["hidden_states_copied"] is False
    assert not (cache / "hs_0.safetensors").exists()
    saved_manifest = json.loads(
        (output / "partial_offline_manifest.json").read_text()
    )
    assert saved_manifest["validated_indices"] == [1, 4]


def test_checker_accepts_source_index_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    data_path = tmp_path / "source"
    cache = tmp_path / "cache"
    output = tmp_path / "filtered"
    tokens = _source_dataset(data_path)
    cache.mkdir()
    _save_hidden_states(cache, 1, tokens[1])
    _save_hidden_states(cache, 4, tokens[4])
    # Collection may continue for rows outside this completed-file snapshot.
    (cache / ".hs_2.safetensors.partial").touch()
    build_partial_dataset(data_path, cache, output, validate_samples=0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_offline_hidden_states.py",
            "--data",
            str(output),
            "--hidden-states",
            str(cache),
            "--validate-samples",
            "-1",
        ],
    )

    check_offline_main()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "complete"
    assert summary["index_mapping"] == "source_index"
    assert summary["validated_indices"] == [1, 4]
    assert "OFFLINE_CHECK phase=cache_scan status=completed" in captured.err
    assert "OFFLINE_CHECK phase=content_validation status=completed" in captured.err


def test_build_partial_dataset_rejects_cache_from_different_dataset(tmp_path: Path):
    data_path = tmp_path / "source"
    cache = tmp_path / "cache"
    output = tmp_path / "filtered"
    _source_dataset(data_path)
    cache.mkdir()
    (cache / "hs_99.safetensors").touch()

    with pytest.raises(ValueError, match="does not match --data"):
        build_partial_dataset(data_path, cache, output, validate_samples=0)
