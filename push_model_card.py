"""
push_model_card.py — upload MODEL_CARD.md as the README of the HF model repo.

Run after the eval has filled MODEL_CARD.md's placeholders.

    python push_model_card.py --push     # uploads; omit --push for a dry run
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

REPO_ID = "AaronHuang160/daxplain-8b"
CARD = Path(__file__).parent / "MODEL_CARD.md"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="actually upload")
    args = ap.parse_args()

    text = CARD.read_text(encoding="utf-8")
    if "{{" in text:
        sys.exit("MODEL_CARD.md still has unfilled {{placeholders}} — fill the eval first.")

    print(f"Model card -> https://huggingface.co/{REPO_ID}  ({len(text)} chars)")
    if not args.push:
        print("Dry run. Re-run with --push to upload.")
        return

    token = os.getenv("HF_TOKEN", "")
    if not token.startswith("hf_"):
        sys.exit("HF_TOKEN missing/invalid in .env.")

    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded model card to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
