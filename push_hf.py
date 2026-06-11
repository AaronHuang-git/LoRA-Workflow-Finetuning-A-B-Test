"""
push_hf.py — publish the dataset to the Hugging Face Hub (Night 1 deliverable).

Uploads train.jsonl, heldout.jsonl, and the rendered README.md (dataset card) to
a public dataset repo. Requires HF_TOKEN (write scope) in .env and the
`huggingface_hub` package.

    python push_hf.py                 # dry run: list what WOULD be uploaded
    python push_hf.py --push          # actually create the repo and upload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

HF_USERNAME = "AaronHuang160"
REPO_ID = f"{HF_USERNAME}/dax-explanations"
DATA_DIR = Path(__file__).parent / "dataset"
FILES = ["train.jsonl", "heldout.jsonl", "README.md"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="actually upload (else dry run)")
    args = ap.parse_args()

    missing = [f for f in FILES if not (DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing files (run split + card first): {missing}")

    print(f"Target dataset repo: https://huggingface.co/datasets/{REPO_ID}")
    for f in FILES:
        kb = (DATA_DIR / f).stat().st_size / 1024
        print(f"  {f:18s} {kb:8.1f} KB")

    if not args.push:
        print("\nDry run. Re-run with --push to create the repo and upload.")
        return

    token = os.getenv("HF_TOKEN", "")
    if not token.startswith("hf_"):
        sys.exit("HF_TOKEN missing/invalid in .env (need a write-scope token).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    for f in FILES:
        api.upload_file(
            path_or_fileobj=str(DATA_DIR / f),
            path_in_repo=f,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"  uploaded {f}")
    print(f"\nDone: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
