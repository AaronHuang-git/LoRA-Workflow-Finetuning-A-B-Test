"""
eval.py — base vs finetuned evaluation on the held-out set (Night 3).

Two phases:
  1. GENERATE (Modal A10G): for each held-out measure, produce a base-model and a
     finetuned-model explanation. Checkpointed to a Modal volume so a disconnect
     can't lose the run.
  2. JUDGE (local): Claude Sonnet 4.6 scores every explanation (blind to which model
     produced it) on accuracy / conciseness / bi_fluency, 1-5 each.

    modal run eval.py --action all  --n 100   # generate (remote) + judge (local)
    modal run eval.py --action all  --n 4     # quick smoke test
    modal run eval.py --action generate --n 100
    modal run eval.py --action judge          # judge existing eval/generations.jsonl

Outputs: eval/generations.jsonl, eval/results.json, eval/results.md
"""

import json
import re
import sys
from pathlib import Path

import modal

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_REPO = "AaronHuang160/dax-explanations"
ADAPTER_REPO = "AaronHuang160/daxplain-8b"
TEACHER_MODEL = "claude-sonnet-4-6"
USER_PROMPT = "Explain this DAX measure in plain English for a capable analyst:\n\n{dax}"

EVAL_DIR = Path(__file__).parent / "eval"
GENERATIONS_FILE = EVAL_DIR / "generations.jsonl"
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_MD = EVAL_DIR / "results.md"

# Same image definition as train.py -> reuses the cached image.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.44.2",
    "datasets==2.21.0",
    "peft==0.12.0",
    "trl==0.9.6",
    "bitsandbytes==0.43.3",
    "accelerate==0.33.0",
    "huggingface_hub==0.24.6",
    "rich==13.7.1",
)

app = modal.App("daxplain-eval", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
eval_vol = modal.Volume.from_name("daxplain-eval", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache, "/data": eval_vol},
)
def generate(n: int = 100):
    """Generate base + finetuned explanations for n held-out measures."""
    import os
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, token=token,
    )
    ft = PeftModel.from_pretrained(model, ADAPTER_REPO, token=token)

    def gen(dax):
        messages = [{"role": "user", "content": USER_PROMPT.format(dax=dax)}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(ft.device)
        out = ft.generate(
            **enc, max_new_tokens=256, do_sample=False,
            repetition_penalty=1.15, pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    ds = load_dataset(DATASET_REPO, split="heldout", token=token).select(range(n))

    # Resume from any checkpoint on the volume.
    ckpt = Path("/data/generations.jsonl")
    done = {}
    if ckpt.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in ckpt.open() if l.strip()}

    rows = []
    with ckpt.open("a") as f:
        for i, r in enumerate(ds, 1):
            if r["id"] in done:
                rows.append(done[r["id"]])
                continue
            with ft.disable_adapter():
                base = gen(r["dax"])
            fine = gen(r["dax"])
            row = {"id": r["id"], "category": r["category"], "difficulty": r["difficulty"],
                   "dax": r["dax"], "base": base, "finetuned": fine}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                eval_vol.commit()
                print(f"  generated {i}/{len(ds)}")
    eval_vol.commit()
    print(f"Generated {len(rows)} pairs.")
    return rows


# --------------------------------------------------------------------------- #
# Local judging (runs on your machine, uses ANTHROPIC_API_KEY from .env)
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a senior Power BI consultant grading plain-English explanations of DAX "
    "measures. Grade only on intrinsic quality; you are not told which model wrote it."
)
JUDGE_USER = """DAX measure:
{dax}

Explanation to grade:
{explanation}

Score the explanation 1-5 (integers) on:
- accuracy: is the DAX behaviour described correctly?
- conciseness: tight and well-structured, no padding or rambling?
- bi_fluency: does it read like a fluent senior BI consultant, with a useful watch-out?

Return ONLY a JSON object: {{"accuracy": int, "conciseness": int, "bi_fluency": int}}"""


METRICS = ("accuracy", "conciseness", "bi_fluency")


def _parse_scores(raw: str):
    """Parse the judge's reply into 3 int scores, or None if unparseable."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(raw)
        return {k: int(d[k]) for k in METRICS}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    found = {}  # salvage: pull "metric": N out of messy text
    for k in METRICS:
        m = re.search(rf'"?{k}"?\s*[:=]\s*([1-5])', raw)
        if m:
            found[k] = int(m.group(1))
    return found if len(found) == 3 else None


def _judge_one(client, dax, explanation):
    # Empty/degenerate generations get the floor score rather than a wasted call.
    if not explanation or len(explanation.strip()) < 15:
        return {k: 1 for k in METRICS}
    for _ in range(3):
        msg = client.messages.create(
            model=TEACHER_MODEL,
            max_tokens=150,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": JUDGE_USER.format(dax=dax, explanation=explanation)}],
        )
        scores = _parse_scores(msg.content[0].text)
        if scores:
            return scores
    print("  ! judge unparseable after retries; using neutral 3s", file=sys.stderr)
    return {k: 3 for k in METRICS}


def _judge_all():
    import os
    from anthropic import Anthropic
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env", override=True)
    except ImportError:
        pass

    rows = [json.loads(l) for l in GENERATIONS_FILE.open(encoding="utf-8") if l.strip()]
    client = Anthropic()
    scored = []
    for i, r in enumerate(rows, 1):
        r["base_scores"] = _judge_one(client, r["dax"], r["base"])
        r["finetuned_scores"] = _judge_one(client, r["dax"], r["finetuned"])
        scored.append(r)
        if i % 10 == 0 or i == len(rows):
            print(f"  judged {i}/{len(rows)}")

    metrics = ("accuracy", "conciseness", "bi_fluency")
    def avg(side, m):
        return round(sum(r[f"{side}_scores"][m] for r in scored) / len(scored), 3)
    summary = {side: {m: avg(side, m) for m in metrics} for side in ("base", "finetuned")}
    for side in summary:
        summary[side]["overall"] = round(sum(summary[side][m] for m in metrics) / 3, 3)

    RESULTS_JSON.write_text(
        json.dumps({"n": len(scored), "summary": summary, "rows": scored}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_md(scored, summary)
    print("\n=== RESULTS ===")
    for m in (*metrics, "overall"):
        b, f = summary["base"][m], summary["finetuned"][m]
        print(f"  {m:12s} base {b:.2f}  ->  finetuned {f:.2f}   (Δ {f - b:+.2f})")


def _write_md(rows, summary):
    metrics = ("accuracy", "conciseness", "bi_fluency", "overall")
    lines = [
        "# DAXplain-8B — Evaluation Results",
        "",
        f"Base **Llama 3.1 8B Instruct** vs **finetuned DAXplain-8B** on "
        f"{len(rows)} held-out DAX measures, scored by Claude Sonnet 4.6 "
        "(blind to source) on a 1–5 rubric.",
        "",
        "| Metric | Base | Finetuned | Δ |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        b, f = summary["base"][m], summary["finetuned"][m]
        lines.append(f"| {m} | {b:.2f} | {f:.2f} | {f - b:+.2f} |")
    lines += ["", "## Worked examples", ""]
    # show 3 examples where the finetuned overall beat base by the most
    def gain(r):
        return (sum(r["finetuned_scores"].values()) - sum(r["base_scores"].values()))
    for r in sorted(rows, key=gain, reverse=True)[:3]:
        lines += [
            f"### `{r['dax']}`",
            f"*Base scores* {r['base_scores']} — *Finetuned scores* {r['finetuned_scores']}",
            "",
            f"**Base:** {r['base']}",
            "",
            f"**Finetuned:** {r['finetuned']}",
            "",
        ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


@app.local_entrypoint()
def main(action: str = "all", n: int = 100):
    EVAL_DIR.mkdir(exist_ok=True)
    if action in ("generate", "all"):
        rows = generate.remote(n)
        with GENERATIONS_FILE.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} generations to {GENERATIONS_FILE}")
    if action in ("judge", "all"):
        _judge_all()
