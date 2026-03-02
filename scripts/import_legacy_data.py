"""Import notebook-era Connect4 datasets into normalized c4 CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c4_training.dataset import import_legacy_file


def main() -> int:
    """Run one legacy data import job from CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to legacy input file")
    parser.add_argument("--output", required=True, help="Path to output normalized CSV")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "jsonl", "semicolon"],
        help="Legacy input format (default: auto)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to import (0 means all)")
    args = parser.parse_args()

    count, path = import_legacy_file(
        args.source,
        args.output,
        file_format=args.format,
        limit=(None if int(args.limit) <= 0 else int(args.limit)),
    )
    print(f"Imported {count} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
