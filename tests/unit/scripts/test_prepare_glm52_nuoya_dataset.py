from pathlib import Path

import pytest

from scripts.prepare_glm52_nuoya_32k import discover_files


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
