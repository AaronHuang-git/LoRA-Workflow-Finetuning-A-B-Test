---
license: cc-by-4.0
language:
- en
tags:
- dax
- power-bi
- business-intelligence
- synthetic
- distillation
task_categories:
- text-generation
size_categories:
- n<1K
pretty_name: DAX Measure Explanations
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: heldout
    path: heldout.jsonl
---

# DAX Measure Explanations

A dataset of **DAX measures paired with concise, plain-English explanations** written
in the voice of a senior Business Intelligence consultant. Each explanation covers
*what* the measure computes, *how* it computes it, and *one practical watch-out*.

Built to fine-tune a small open model (`AaronHuang160/daxplain-8b`, a QLoRA adapter on
Llama 3.1 8B Instruct) specialised for explaining Power BI / Tabular semantic-model DAX.

## Provenance & methodology

This is a **synthetic, distilled** dataset produced with a teacher–student workflow:

1. **Seed.** ~30 hand-written DAX measures spanning six categories (simple aggregation,
   time intelligence, ratios, iterators, ranking, filter-context manipulation) at
   easy/medium/hard difficulty.
2. **Expand.** A teacher model (**Claude Sonnet 4.6**) generated additional realistic
   measures conditioned on category + difficulty, de-duplicated against existing ones,
   yielding **{{TOTAL}}** measures.
3. **Explain.** The teacher wrote a 3–4 sentence explanation for every measure using a
   fixed senior-BI-consultant prompt (what / how / one watch-out).
4. **Curate.** A human BI domain expert reviewed **{{CURATED}} explanations
   ({{CURATED_PCT}} of the dataset)** and **hand-edited {{EDITED}}** of them, concentrating
   edits on the harder categories (time intelligence, filter context, ranking, iterators).
   The edits inject judgement the teacher lacks, tighten the watch-outs, and fix genuine
   technical errors (e.g. `RANKX`'s default SKIP — not dense — ranking, and a weighted-average
   blank-handling claim). The `curated` and `edited` boolean flags record both passes.

The curation step is deliberate: training purely on raw teacher output caps quality at
"slightly worse than the teacher." The hand-edited subset is where the genuine lift comes
from.

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier (e.g. `rank-gen-0412`). |
| `category` | string | One of: `simple_aggregation`, `time_intelligence`, `ratio`, `iterator`, `ranking`, `filter_context`. |
| `difficulty` | string | `easy` / `medium` / `hard`. |
| `dax` | string | The DAX measure as `Name = expression`. |
| `explanation` | string | Plain-English explanation (what / how / watch-out). |
| `curated` | bool | True if a human reviewed and approved the row. |
| `edited` | bool | True if a human edited the explanation text. |

## Splits

| Split | Rows | Purpose |
|---|---|---|
| `train` | {{TRAIN}} | Fine-tuning. |
| `heldout` | {{HELDOUT}} | Base-vs-finetuned evaluation (LLM-as-judge). |

## Category balance

{{CATEGORY_TABLE}}

## Intended use

- Fine-tuning / few-shot prompting for **DAX-to-English explanation** tasks.
- A practitioner aid for documenting semantic models.

**Not** intended as a substitute for professional validation of DAX logic. Explanations
are model-generated (partially human-curated) and may contain errors, especially for very
long measures or recently-released DAX functions.

## Limitations

- Synthetic measures target a generic star-schema sales model; column/table names are
  illustrative, not from any real client model.
- Teacher-introduced phrasing patterns persist in the un-curated majority.
- English only.

## License

Released under **CC-BY-4.0**. Explanations were generated with Claude Sonnet 4.6 and
curated by the author; downstream use is subject to the teacher model provider's terms.

## Citation

```
@misc{daxplain_dataset,
  title  = {DAX Measure Explanations},
  author = {AaronHuang160},
  year   = {2026},
  url    = {https://huggingface.co/datasets/AaronHuang160/dax-explanations}
}
```
