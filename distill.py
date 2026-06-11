"""
distill.py — Night 1 dataset distillation for DAXplain-8B.

Pipeline (run as ordered steps):

    python distill.py smoke                 # ~1c   : prove the API key + parsing work
    python distill.py expand --target 800   # ~$1-2 : grow 30 seeds -> ~800 DAX measures
    python distill.py explain               # ~$2-3 : add a plain-English explanation to each
    python distill.py split --heldout 100   # free  : deterministic train / held-out split
    python distill.py stats                 # free  : category + difficulty counts

Claude Sonnet 4.6 is the "teacher": it both *generates* DAX measures (expand) and
*writes the explanations* (explain) that Llama 3.1 8B will later be trained to mimic.

Files (all under dataset/):
    seed_measures.jsonl      hand-written starter set (input to `expand`)
    measures.jsonl           expanded measure pool            (output of `expand`)
    dax_explanations.jsonl   {dax, explanation, ...} pairs     (output of `explain`)
    train.jsonl / heldout.jsonl                                (output of `split`)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv

    # pin to this folder (not CWD); override=True so .env wins over any empty/stale
    # ANTHROPIC_API_KEY already present in the shell environment
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:  # dotenv is optional; env vars may be set another way
    pass

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

TEACHER_MODEL = "claude-sonnet-4-6"
DATA_DIR = Path(__file__).parent / "dataset"
SEED_FILE = DATA_DIR / "seed_measures.jsonl"
MEASURES_FILE = DATA_DIR / "measures.jsonl"
EXPLAINED_FILE = DATA_DIR / "dax_explanations.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
HELDOUT_FILE = DATA_DIR / "heldout.jsonl"
CARD_TEMPLATE = DATA_DIR / "README.template.md"
CARD_FILE = DATA_DIR / "README.md"

CATEGORIES = [
    "simple_aggregation",
    "time_intelligence",
    "ratio",
    "iterator",
    "ranking",
    "filter_context",
]
DIFFICULTIES = ["easy", "medium", "hard"]

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

EXPLAIN_SYSTEM = (
    "You are a senior BI consultant who has shipped hundreds of Power BI semantic "
    "models. You explain DAX measures to capable analysts in tight, plain English."
)

EXPLAIN_USER = """Given the following DAX measure, write a concise plain-English \
explanation (3-4 sentences) covering:
1. WHAT it computes (the business meaning),
2. HOW it computes it (the mechanics — filter context, iteration, time shift, etc.),
3. ONE watch-out (a gotcha, edge case, or modelling assumption a practitioner should know).

Write only the explanation prose. No preamble, no bullet points, no restating the DAX.

DAX measure:
{dax}"""

GENERATE_SYSTEM = (
    "You are a senior BI consultant authoring a training dataset of realistic DAX "
    "measures for a star-schema sales model (tables: Sales, Product, Customer, "
    "'Date', 'Geography'). Measures must be syntactically valid DAX a real consultant "
    "would write."
)

GENERATE_USER = """Generate {n} NEW DAX measures in the category "{category}" at \
"{difficulty}" difficulty.

Category meaning:
- simple_aggregation: SUM / AVERAGE / DISTINCTCOUNT / COUNTROWS style basics.
- time_intelligence: YTD/MTD/QTD, prior-period, YoY, rolling windows, DATESINPERIOD.
- ratio: DIVIDE-based rates, shares, per-unit metrics.
- iterator: SUMX / AVERAGEX / MAXX / FILTER row-by-row evaluation.
- ranking: RANKX / TOPN / dense ranks / "is top N" flags.
- filter_context: CALCULATE with ALL / ALLEXCEPT / KEEPFILTERS / boolean filters.

Requirements:
- Each measure on ONE line in the form: Measure Name = <expression>
- Vary table/column names and business intent; make them feel like a real model.
- Do NOT duplicate any of these existing measures:
{existing}

Return ONLY a JSON array of strings, each string a full "Name = expression" measure."""

# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Teacher calls
# --------------------------------------------------------------------------- #


def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    """Single teacher call. Backs off on transient errors; fails fast otherwise."""
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text.strip()
        except (RateLimitError, APIConnectionError) as exc:
            # Transient: rate limit or network blip — worth retrying.
            wait = 2**attempt
            print(f"  ! transient error ({exc}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except APIStatusError as exc:
            # Non-retryable: bad request, auth, billing, etc. Stop immediately.
            sys.exit(f"\nAPI error {exc.status_code}: {exc.message}\nNot retrying.")
    raise RuntimeError("teacher call failed after 4 retries")


def generate_measures(
    category: str, difficulty: str, n: int, existing: list[str]
) -> list[str]:
    sample = random.sample(existing, min(len(existing), 12))
    user = GENERATE_USER.format(
        n=n,
        category=category,
        difficulty=difficulty,
        existing="\n".join(f"- {m}" for m in sample),
    )
    raw = _call(GENERATE_SYSTEM, user, max_tokens=4096)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        items = [str(x).strip() for x in json.loads(raw)]
    except json.JSONDecodeError:
        # Long, deeply-nested measures can truncate the array mid-stream. Salvage
        # every complete quoted "Name = expression" string instead of dropping all.
        items = []
        for raw_str in re.findall(r'"(?:[^"\\]|\\.)*"', raw):
            try:  # decode JSON escapes (e.g. \" inside string filters)
                decoded = json.loads(raw_str)
            except json.JSONDecodeError:
                continue
            if "=" in decoded:
                items.append(decoded.strip())
        if not items:
            print(f"  ! no measures parsed for {category}/{difficulty}", file=sys.stderr)
    return [m for m in items if m]


def explain_measure(dax: str) -> str:
    return _call(EXPLAIN_SYSTEM, EXPLAIN_USER.format(dax=dax), max_tokens=512)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def step_smoke() -> None:
    seeds = read_jsonl(SEED_FILE)[:2]
    if not seeds:
        sys.exit("No seed measures found — check dataset/seed_measures.jsonl")
    print("Smoke test — 2 measures through the explainer:\n")
    for row in seeds:
        print(f"DAX: {row['dax']}")
        print(f"Explanation: {explain_measure(row['dax'])}\n")
    print("Smoke test OK. API key, model, and parsing all work.")


def step_expand(target: int) -> None:
    seeds = read_jsonl(SEED_FILE)
    pool = read_jsonl(MEASURES_FILE) or list(seeds)
    seen = {r["dax"] for r in pool}
    per_combo = max(1, target // (len(CATEGORIES) * len(DIFFICULTIES)))

    print(f"Expanding to ~{target} measures (~{per_combo} per category/difficulty)...")
    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            existing = [r["dax"] for r in pool if r["category"] == category] or [
                r["dax"] for r in seeds
            ]
            new = generate_measures(category, difficulty, per_combo, existing)
            added = 0
            for dax in new:
                if dax in seen:
                    continue
                seen.add(dax)
                pool.append(
                    {
                        "id": f"{category[:4]}-gen-{len(pool):04d}",
                        "category": category,
                        "difficulty": difficulty,
                        "dax": dax,
                    }
                )
                added += 1
            write_jsonl(MEASURES_FILE, pool)  # checkpoint after each combo
            print(f"  {category:18s} {difficulty:6s} +{added:3d}  (pool={len(pool)})")
    print(f"Done. {len(pool)} measures in {MEASURES_FILE.name}")


def step_explain() -> None:
    measures = read_jsonl(MEASURES_FILE)
    if not measures:
        sys.exit("No measures.jsonl — run `expand` first.")
    done = {r["id"] for r in read_jsonl(EXPLAINED_FILE)}
    todo = [m for m in measures if m["id"] not in done]
    print(f"Explaining {len(todo)} measures ({len(done)} already done)...")
    for i, m in enumerate(todo, 1):
        explanation = explain_measure(m["dax"])
        append_jsonl(
            EXPLAINED_FILE,
            {
                "id": m["id"],
                "category": m["category"],
                "difficulty": m["difficulty"],
                "dax": m["dax"],
                "explanation": explanation,
                "curated": False,  # flip to True for the ~160 you hand-edit
            },
        )
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
    print(f"Done. {EXPLAINED_FILE.name} now has {len(read_jsonl(EXPLAINED_FILE))} rows.")


def step_split(heldout: int, seed: int) -> None:
    rows = read_jsonl(EXPLAINED_FILE)
    if len(rows) <= heldout:
        sys.exit(f"Only {len(rows)} rows — need more than {heldout} to split.")
    rng = random.Random(seed)
    rng.shuffle(rows)
    write_jsonl(HELDOUT_FILE, rows[:heldout])
    write_jsonl(TRAIN_FILE, rows[heldout:])
    print(f"Split: {len(rows) - heldout} train / {heldout} held-out (seed={seed}).")


def step_stats() -> None:
    for name, path in [("measures", MEASURES_FILE), ("explained", EXPLAINED_FILE)]:
        rows = read_jsonl(path)
        if not rows:
            continue
        print(f"\n{name}: {len(rows)} rows")
        print("  by category:  ", dict(Counter(r["category"] for r in rows)))
        print("  by difficulty:", dict(Counter(r["difficulty"] for r in rows)))
        curated = sum(1 for r in rows if r.get("curated"))
        if name == "explained":
            print(f"  hand-curated: {curated} ({curated / len(rows):.0%})")


def step_card() -> None:
    """Render dataset/README.md from the template, filling in real numbers."""
    if not CARD_TEMPLATE.exists():
        sys.exit("No README.template.md — cannot render the dataset card.")
    explained = read_jsonl(EXPLAINED_FILE)
    train = read_jsonl(TRAIN_FILE)
    heldout = read_jsonl(HELDOUT_FILE)
    if not train or not heldout:
        sys.exit("Run `split` before `card` (train/heldout not found).")

    total = len(explained)
    edited = sum(1 for r in explained if r.get("edited"))
    curated = sum(1 for r in explained if r.get("curated"))

    tr = Counter(r["category"] for r in train)
    ho = Counter(r["category"] for r in heldout)
    lines = ["| Category | Train | Held-out | Total |", "|---|---|---|---|"]
    for c in CATEGORIES:
        lines.append(f"| `{c}` | {tr[c]} | {ho[c]} | {tr[c] + ho[c]} |")
    lines.append(f"| **All** | **{len(train)}** | **{len(heldout)}** | **{total}** |")

    text = CARD_TEMPLATE.read_text(encoding="utf-8")
    for key, val in {
        "{{TOTAL}}": str(total),
        "{{CURATED}}": str(curated),
        "{{EDITED}}": str(edited),
        "{{CURATED_PCT}}": f"{curated / total:.0%}",
        "{{TRAIN}}": str(len(train)),
        "{{HELDOUT}}": str(len(heldout)),
        "{{CATEGORY_TABLE}}": "\n".join(lines),
    }.items():
        text = text.replace(key, val)

    CARD_FILE.write_text(text, encoding="utf-8")
    print(f"Rendered {CARD_FILE.name}: {total} rows, {curated} curated, {edited} edited.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)

    sub.add_parser("smoke", help="2 measures end-to-end (~1c)")
    p_exp = sub.add_parser("expand", help="grow seeds into ~target measures")
    p_exp.add_argument("--target", type=int, default=800)
    sub.add_parser("explain", help="add explanations (resumable)")
    p_split = sub.add_parser("split", help="train / held-out split")
    p_split.add_argument("--heldout", type=int, default=100)
    p_split.add_argument("--seed", type=int, default=42)
    sub.add_parser("stats", help="category / difficulty counts")
    sub.add_parser("card", help="render dataset/README.md from template")

    args = parser.parse_args()
    if args.step == "smoke":
        step_smoke()
    elif args.step == "expand":
        step_expand(args.target)
    elif args.step == "explain":
        step_explain()
    elif args.step == "split":
        step_split(args.heldout, args.seed)
    elif args.step == "stats":
        step_stats()
    elif args.step == "card":
        step_card()


if __name__ == "__main__":
    main()
