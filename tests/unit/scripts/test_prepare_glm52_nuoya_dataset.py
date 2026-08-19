from pathlib import Path

import pytest

from scripts.prepare_glm52_nuoya_32k import discover_files


REPO_ROOT = Path(__file__).resolve().parents[3]
PREPARE_8K = REPO_ROOT / "examples/train/prepare_glm52_nuoya_8k.sh"
PREPARE_4K = REPO_ROOT / "examples/train/prepare_glm52_nuoya_4k.sh"


def _touch_files(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_text("{}\n", encoding="utf-8")


def test_discover_files_selects_sorted_jsonl_limit_per_source(tmp_path: Path):
    nuoya = tmp_path / "nuoya"
    long = tmp_path / "long"
    _touch_files(
        nuoya,
        ["06.jsonl", "02.jsonl", "01.jsonl", "04.jsonl", "03.jsonl", "05.jsonl"],
    )
    _touch_files(long, ["long.jsonl", "ignored.json"])

    selected = discover_files(
        [str(nuoya), str(long)],
        [5, 1],
        jsonl_only=True,
    )

    assert [path.name for path in selected] == [
        "01.jsonl",
        "02.jsonl",
        "03.jsonl",
        "04.jsonl",
        "05.jsonl",
        "long.jsonl",
    ]


def test_discover_files_rejects_limit_larger_than_source(tmp_path: Path):
    source = tmp_path / "source"
    _touch_files(source, ["only.jsonl"])

    with pytest.raises(ValueError, match="fewer than requested limit"):
        discover_files([str(source)], [2], jsonl_only=True)


def test_8k_wrapper_exports_repo_pythonpath():
    text = PREPARE_8K.read_text(encoding="utf-8")

    assert 'export PYTHONPATH="$REPO_ROOT/src:' in text
    assert "$REPO_ROOT/hs_connectors/src:" in text
    assert "--source-file-limit 5" in text
    assert "--source-file-limit 1" in text


def test_4k_wrapper_uses_isolated_output_and_expected_sources():
    text = PREPARE_4K.read_text(encoding="utf-8")

    assert 'export PYTHONPATH="$REPO_ROOT/src:' in text
    assert "nuoya-first5-long1-4k" in text
    assert "--source-file-limit 5" in text
    assert "--source-file-limit 1" in text
    assert "--seq-length 4096" in text
