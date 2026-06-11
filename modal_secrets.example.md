# Modal secrets setup

`train.py` and `eval.py` run on Modal and need your Hugging Face token (to download
gated Llama 3.1 and push the adapter). The token is provided as a **Modal secret** —
never committed to the repo.

## 1. Authenticate Modal

```bash
modal token new      # opens a browser to confirm
```

## 2. Create the HF secret

The secret must be named `hf-token` and contain a `HF_TOKEN` variable:

```bash
modal secret create hf-token HF_TOKEN=hf_your_write_scoped_token
```

Your HF token needs:
- **write** scope (to push the adapter), and
- access to `meta-llama/Llama-3.1-8B-Instruct` (accept the license on the model page).

## 3. Local `.env` (for the eval judge + dataset distillation)

These steps run on your machine, not Modal, so they read `.env` (see `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...   # teacher (distillation) + judge (eval)
HF_TOKEN=hf_...                # dataset push
```

`.env` is git-ignored. Verify the secret registered with `modal secret list`.
