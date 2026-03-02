"""Legacy dataset import utilities for Connect4 notebook-era files."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LegacyMoveRecord:
    """One board-position sample with move-score annotations."""

    board: list[int]
    move_scores: list[int]
    score: int | None = None
    source: str = ""
    line_no: int = 0

    @property
    def label(self) -> int:
        """Return argmax move index used for supervised label."""

        return int(max(range(len(self.move_scores)), key=lambda idx: self.move_scores[idx]))


def infer_legacy_format(path: str | Path) -> str:
    """Infer parser format (`jsonl` or `semicolon`) from file content."""

    src = Path(path)
    with src.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("{"):
                return "jsonl"
            if '"board"' in line and ";" in line:
                return "semicolon"
    raise ValueError(f"Unable to infer legacy format for file: {src}")


def _validate_record(record: LegacyMoveRecord) -> LegacyMoveRecord:
    if len(record.board) != 42:
        raise ValueError(f"Expected board length 42, got {len(record.board)} at {record.source}:{record.line_no}")
    if len(record.move_scores) != 7:
        raise ValueError(
            f"Expected move_scores length 7, got {len(record.move_scores)} at {record.source}:{record.line_no}"
        )
    return record


def parse_jsonl_records(path: str | Path, *, limit: int | None = None) -> list[LegacyMoveRecord]:
    """Parse JSONL-style lines with `board` and `move score` keys."""

    src = Path(path)
    rows: list[LegacyMoveRecord] = []
    with src.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line[:-1] if line.endswith(";") else line)
            record = LegacyMoveRecord(
                board=[int(v) for v in payload["board"]],
                move_scores=[int(v) for v in payload["move score"]],
                score=int(payload["score"]) if "score" in payload and payload["score"] is not None else None,
                source=str(src),
                line_no=line_no,
            )
            rows.append(_validate_record(record))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


_BOARD_RE = re.compile(r'"board"\s*:\s*(\[[^\]]+\])')
_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+)')
_MOVE_SCORE_RE = re.compile(r'"move score"\s*:\s*(\[[^\]]+\])')


def parse_semicolon_records(path: str | Path, *, limit: int | None = None) -> list[LegacyMoveRecord]:
    """Parse semicolon-delimited legacy lines from `good_move_semic.csv`."""

    src = Path(path)
    rows: list[LegacyMoveRecord] = []
    with src.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            board_match = _BOARD_RE.search(line)
            score_match = _SCORE_RE.search(line)
            move_score_match = _MOVE_SCORE_RE.search(line)
            if not board_match or not move_score_match:
                raise ValueError(f"Malformed semicolon record at {src}:{line_no}")
            board = [int(v) for v in json.loads(board_match.group(1))]
            move_scores = [int(v) for v in json.loads(move_score_match.group(1))]
            score = int(score_match.group(1)) if score_match else None
            record = LegacyMoveRecord(
                board=board,
                move_scores=move_scores,
                score=score,
                source=str(src),
                line_no=line_no,
            )
            rows.append(_validate_record(record))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def records_to_rows(records: list[LegacyMoveRecord]) -> list[dict]:
    """Convert parsed records to flat training rows."""

    rows: list[dict] = []
    for item in records:
        row = {f"b{index:02d}": int(value) for index, value in enumerate(item.board)}
        row["label"] = int(item.label)
        row["score"] = int(item.score) if item.score is not None else ""
        row["move_scores_json"] = json.dumps(item.move_scores)
        rows.append(row)
    return rows


def write_training_csv(records: list[LegacyMoveRecord], output_path: str | Path) -> Path:
    """Write normalized supervised dataset rows to CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(records)
    if not rows:
        raise ValueError("No records to write.")
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def import_legacy_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    file_format: str = "auto",
    limit: int | None = None,
) -> tuple[int, Path]:
    """Import one legacy source file into normalized CSV training format."""

    resolved_format = infer_legacy_format(source_path) if file_format == "auto" else str(file_format).strip().lower()
    if resolved_format == "jsonl":
        records = parse_jsonl_records(source_path, limit=limit)
    elif resolved_format == "semicolon":
        records = parse_semicolon_records(source_path, limit=limit)
    else:
        raise ValueError("file_format must be one of: auto, jsonl, semicolon")
    path = write_training_csv(records, output_path)
    return len(records), path
