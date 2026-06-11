---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
license: llama3.1
language:
- en
pipeline_tag: text-generation
tags:
- lora
- qlora
- peft
- dax
- power-bi
- business-intelligence
datasets:
- AaronHuang160/dax-explanations
---

# DAXplain-8B

A **QLoRA adapter** on **Llama 3.1 8B Instruct** that explains Power BI / Tabular
**DAX measures** in plain English, in the voice of a senior BI consultant: what the
measure computes, how it computes it, and one practical watch-out.

- **Adapter** (this repo): rank-16 LoRA, ~168 MB. Load on top of the base model.
- **Dataset:** [`AaronHuang160/dax-explanations`](https://huggingface.co/datasets/AaronHuang160/dax-explanations)
- **Code / reproducibility:** see the linked GitHub repo (`modal run train.py`).

## What it does

Given a DAX measure such as:

```
Sales YTD = TOTALYTD ( [Total Sales], 'Date'[Date] )
```

it returns a concise, structured explanation ending in a real-world watch-out
(e.g. "the year boundary defaults to December 31 — fiscal years need the optional
year-end argument").

## Intended use

- A practitioner aid for **documenting and understanding DAX** in a semantic model.
- Few-shot / batch explanation generation for BI documentation.

**Not** intended as a substitute for professional validation of DAX logic. Output is
model-generated and can be wrong — especially on very long measures or recently
released DAX functions. Always verify against the actual model behaviour.

## Training data

Finetuned on **611 training rows** from
[`AaronHuang160/dax-explanations`](https://huggingface.co/datasets/AaronHuang160/dax-explanations):
~711 synthetic DAX measures across six categories (simple aggregation, time
intelligence, ratios, iterators, ranking, filter-context), each paired with a
Claude-Sonnet-4.6-distilled explanation. 178 rows (25%) were human-reviewed and 30
were hand-edited by a BI practitioner — including factual corrections (e.g. `RANKX`'s
default SKIP ranking). 100 rows were held out for evaluation.

## How it was trained

QLoRA via `trl` `SFTTrainer` on a single Modal A10G GPU (~11 min):

| Setting | Value |
|---|---|
| Base | `meta-llama/Llama-3.1-8B-Instruct`, 4-bit (nf4) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | all attention + MLP projections |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Epochs / effective batch | 2 / 16 |
| Final train loss | 1.09 (from 2.14) |

## Evaluation

Base **Llama 3.1 8B Instruct** vs **DAXplain-8B** on **100 held-out measures**,
scored by Claude Sonnet 4.6 (blind to source) on a 1–5 rubric.

| Metric | Base Llama 3.1 8B | DAXplain-8B | Δ |
|---|---|---|---|
| Accuracy | 3.11 | 2.91 | −0.20 |
| Conciseness | 2.51 | 2.17 | −0.34 |
| BI-fluency | 2.29 | **3.21** | **+0.92** |
| **Overall** | 2.64 | **2.76** | **+0.13** |

The headline effect is a large gain in **BI-fluency / readability** (+0.92) and a
net-positive overall score: the base model tends to produce verbose, markdown-bulleted
tutorials that truncate mid-list, while DAXplain-8B produces tight prose ending in a
single actionable watch-out. This comes with a **small, honestly-reported tradeoff** in
raw accuracy (−0.20) and conciseness (−0.34) — the expected profile of a *style*
finetune, which changes *how* the model explains, not *what* it knows. Scores are
deliberately strict (the judge rarely awards 5s), so read them relatively, not absolutely.

### Worked examples

**`Min Order Value = MINX ( VALUES ( Sales[OrderNumber] ), CALCULATE ( SUMX ( Sales, Sales[Quantity] * Sales[UnitPrice] ) ) )`**
*Base 3/2/2 → Finetuned 4/3/4*
- **Base:** "Let's break down the given DAX measure step by step. **Measure Name:** … **Purpose:** … 1. **MINX**: … 2. **VALUES**: …" *(long numbered walkthrough, truncates mid-list)*
- **Finetuned:** "This measure finds the single lowest total order value across all orders visible in the current filter context… It works by using VALUES to get each distinct OrderNumber, then iterating with MINX; inside each iteration CALCULATE shifts the row context into a filter context so SUMX can sum Quantity × UnitPrice for that one order. The key watch-out is that if multiple rows share the same OrderNumber they're summed together — so you're looking at aggregated order values, not line items…"

**`Top 5 Sales Reps Profit = CALCULATE ( [Total Profit], TOPN ( 5, ALL ( Sales[SalesRepName] ), [Total Profit], DESC ) )`**
*Base 3/2/2 → Finetuned 4/3/4*
- **Base:** "Let's break down the given DAX measure step by step. **What does it do?** … 1. `CALCULATE([Total Profit])`: … 2. `TOPN(5,...)`: …" *(numbered, truncates)*
- **Finetuned:** "This measure returns the total profit contributed by only the top five sales reps… TOPN scans all reps after stripping the SalesRepName filter via ALL, ranks by Total Profit, and selects five; CALCULATE applies that set back. The key watch-out is that ties at the fifth position make TOPN return more than five rows, inflating the total — add a tie-break if exactly five matters."

See [`eval/results.md`](eval/results.md) in the GitHub repo for the full table and more examples.

## Limitations

- Trained on a **generic star-schema** vocabulary; table/column names are illustrative.
- Synthetic measures skew toward common patterns; rare or very long measures are weaker.
- A *style* finetune — it does not add DAX knowledge the base model lacks, and can
  inherit or amplify base-model factual errors.
- English only.

## License

Adapter released under the **Llama 3.1 Community License** (inherited from the base
model). Training data is CC-BY-4.0.

## Reproducibility

```bash
modal run train.py        # trains and pushes this adapter
modal run eval.py --action all --n 100   # reproduces the eval table above
```
