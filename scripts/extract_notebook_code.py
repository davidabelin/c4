"""CLI utility to extract ``%%writefile`` cells from notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_writefiles(notebook_path: Path, output_dir: Path, overwrite: bool) -> list[Path]:
    """Extract files declared via notebook ``%%writefile`` magics."""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    written: list[Path] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell.get("source", [])).splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if not first.startswith("%%writefile "):
            continue
        target_name = first.replace("%%writefile ", "", 1).strip()
        target_path = output_dir / target_name
        if target_path.exists() and not overwrite:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
        written.append(target_path)
    return written


def main() -> int:
    """Parse arguments and run notebook extraction CLI."""

    parser = argparse.ArgumentParser(description="Extract %%writefile cells from a Jupyter notebook.")
    parser.add_argument("--input", required=True, help="Path to the notebook (.ipynb)")
    parser.add_argument("--output", required=True, help="Directory where extracted files will be written")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files if they already exist")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Notebook not found: {input_path}")
    written = extract_writefiles(input_path, output_dir, overwrite=args.overwrite)
    print(f"Extracted {len(written)} files from {input_path.name}")
    for path in written:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
