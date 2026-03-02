from __future__ import annotations

from pathlib import Path

from c4_training.dataset import import_legacy_file, parse_jsonl_records, parse_semicolon_records


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_jsonl_records(tmp_path: Path):
    src = _write(
        tmp_path / "legacy.jsonl",
        '{"board": [' + ",".join(["0"] * 42) + '], "score": 5, "move score": [1,2,3,4,5,6,7]}\n',
    )
    rows = parse_jsonl_records(src)
    assert len(rows) == 1
    assert rows[0].label == 6


def test_parse_semicolon_records(tmp_path: Path):
    src = _write(
        tmp_path / "legacy_semic.csv",
        '"board": [' + ",".join(["0"] * 42) + ']; "score": -2; "move score": [7,6,5,4,3,2,1]\n',
    )
    rows = parse_semicolon_records(src)
    assert len(rows) == 1
    assert rows[0].label == 0


def test_import_legacy_file_to_csv(tmp_path: Path):
    src = _write(
        tmp_path / "legacy.jsonl",
        '{"board": [' + ",".join(["0"] * 42) + '], "score": 1, "move score": [0,0,0,1,0,0,0]}\n',
    )
    out = tmp_path / "normalized" / "training.csv"
    count, written = import_legacy_file(src, out, file_format="jsonl")
    assert count == 1
    assert written.exists()
    header = written.read_text(encoding="utf-8").splitlines()[0]
    assert "b00" in header
    assert "label" in header
