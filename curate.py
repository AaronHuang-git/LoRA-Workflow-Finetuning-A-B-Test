"""
curate.py — guided human curation of the distilled explanations (Night 1).

The teacher (Claude) wrote every explanation; the *lift* comes from a domain
expert hand-editing ~20% of them. This tool walks you through the dataset one
measure at a time and tracks progress toward a target.

    python curate.py                 # review toward the default target (160)
    python curate.py --target 160
    python curate.py --category ranking   # focus one category
    python curate.py --all           # also revisit already-curated rows

Per measure:
    [e] edit    open the explanation in your editor; marks curated + edited
    [a] approve keep as-is but mark reviewed; marks curated (not edited)
    [s] skip    leave untouched, move on
    [q] quit    save and exit

Edits write back to dataset/dax_explanations.jsonl immediately (crash-safe).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dataset"
EXPLAINED_FILE = DATA_DIR / "dax_explanations.jsonl"


def read_rows() -> list[dict]:
    if not EXPLAINED_FILE.exists():
        raise SystemExit("No dax_explanations.jsonl — run `distill.py explain` first.")
    with EXPLAINED_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict]) -> None:
    with EXPLAINED_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def edit_in_editor(initial: str) -> str:
    """Open `initial` in the user's editor; return the saved text."""
    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(initial)
        path = f.name
    try:
        subprocess.call([editor, path])  # blocks until the editor window closes
        return Path(path).read_text(encoding="utf-8").strip()
    finally:
        os.unlink(path)


def build_queue(rows: list[dict], category: str | None, include_curated: bool):
    """Rows to review, interleaved across categories for an even spread."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if category and r["category"] != category:
            continue
        if not include_curated and r.get("curated"):
            continue
        by_cat[r["category"]].append(r)

    queue, buckets = [], list(by_cat.values())
    while any(buckets):
        for bucket in buckets:
            if bucket:
                queue.append(bucket.pop(0))
    return queue


def wrap(text: str, width: int = 88) -> str:
    import textwrap

    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=160, help="curated-row goal")
    ap.add_argument("--category", help="restrict to one category")
    ap.add_argument("--all", action="store_true", help="include already-curated rows")
    args = ap.parse_args()

    rows = read_rows()
    queue = build_queue(rows, args.category, args.all)
    curated_now = sum(1 for r in rows if r.get("curated"))

    if not queue:
        print("Nothing to review (all matching rows already curated).")
        return

    print(f"{len(queue)} rows to review. Currently curated: {curated_now}/{args.target}.")
    print("Commands:  [e]dit  [a]pprove  [s]kip  [q]uit\n")

    edited = 0
    for i, r in enumerate(queue, 1):
        curated_now = sum(1 for x in rows if x.get("curated"))
        print("=" * 90)
        print(
            f"[{i}/{len(queue)}]  {r['id']}  |  {r['category']} / {r['difficulty']}"
            f"   (curated {curated_now}/{args.target}, edited {edited})"
        )
        print(f"\nDAX:\n  {r['dax']}\n")
        print("EXPLANATION:")
        print(wrap(r["explanation"]))
        print()

        cmd = input("[e]dit / [a]pprove / [s]kip / [q]uit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "e":
            new = edit_in_editor(r["explanation"])
            if new and new != r["explanation"]:
                r["explanation"] = new
                r["edited"] = True
                edited += 1
            r["curated"] = True
            write_rows(rows)
            print("  saved (edited).\n")
        elif cmd == "a":
            r["curated"] = True
            r["edited"] = r.get("edited", False)
            write_rows(rows)
            print("  saved (approved).\n")
        else:
            print("  skipped.\n")

    write_rows(rows)
    final = sum(1 for r in rows if r.get("curated"))
    print(f"\nDone. Curated {final}/{args.target}  (edited this session: {edited}).")


if __name__ == "__main__":
    main()
