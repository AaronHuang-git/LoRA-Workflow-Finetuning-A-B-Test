"""
apply_edits.py — apply a batch of human explanation edits to the dataset.

Reads a JSON file mapping {id: new_explanation}, updates dax_explanations.jsonl,
and sets curated=True + edited=True on each touched row.

    python apply_edits.py _edits.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPLAINED_FILE = Path(__file__).parent / "dataset" / "dax_explanations.jsonl"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python apply_edits.py <edits.json>")
    edits = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    rows = [json.loads(l) for l in EXPLAINED_FILE.open(encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}

    applied = 0
    for rid, new_text in edits.items():
        if rid not in by_id:
            print(f"  ! id not found: {rid}", file=sys.stderr)
            continue
        by_id[rid]["explanation"] = new_text.strip()
        by_id[rid]["curated"] = True
        by_id[rid]["edited"] = True
        applied += 1

    with EXPLAINED_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    edited = sum(1 for r in rows if r.get("edited"))
    print(f"Applied {applied} edits. Dataset now has {edited} edited rows total.")


if __name__ == "__main__":
    main()
