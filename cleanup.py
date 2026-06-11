"""
cleanup.py — cosmetic normalization of explanation text across the whole dataset.

Strips stray markdown the teacher occasionally emitted (** bold, ` code ticks),
normalizes a few unicode characters to plain ASCII, and tidies whitespace.
Meaning is never changed. Run once after curation.

    python cleanup.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXPLAINED_FILE = Path(__file__).parent / "dataset" / "dax_explanations.jsonl"

REPLACEMENTS = {
    "**": "",        # markdown bold markers
    "`": "",         # inline-code ticks
    "≠": "<>",  # != not-equal sign
    "‘": "'",   # left single quote
    "’": "'",   # right single quote / apostrophe
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "–": "-",   # en dash
    "—": "-",   # em dash
}


def clean(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse runs of spaces
    return text.strip()


def main() -> None:
    rows = [json.loads(l) for l in EXPLAINED_FILE.open(encoding="utf-8") if l.strip()]
    changed = 0
    for r in rows:
        new = clean(r["explanation"])
        if new != r["explanation"]:
            r["explanation"] = new
            changed += 1
    with EXPLAINED_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Cleaned {changed}/{len(rows)} explanations.")


if __name__ == "__main__":
    main()
