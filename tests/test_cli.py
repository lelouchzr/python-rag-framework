import json
from pathlib import Path

import pytest

from rag.cli import main


def test_cli_version_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip()


def test_cli_inspect_outputs_document_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Topic particles mark contrast.", encoding="utf-8")

    exit_code = main(["inspect", str(path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["chunks"] == 1
    assert payload["metadata"]["filename"] == "notes.txt"


def test_cli_chunk_outputs_chunks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Topic particles mark contrast.", encoding="utf-8")

    exit_code = main(
        [
            "chunk",
            str(path),
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "10",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload[0]["text"] == "Topic particles mark contrast."


def test_cli_search_outputs_ranked_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text(
        "Topic particles mark contrast.\n\nBread needs flour and water.",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "search",
            str(path),
            "topic contrast",
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "10",
            "--limit",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert len(payload) == 1
    assert "Topic particles" in payload[0]["text"]
