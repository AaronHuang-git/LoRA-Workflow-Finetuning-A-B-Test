"""
train.py — QLoRA finetune of Llama 3.1 8B Instruct on the DAX-explanations dataset.

Runs on a single Modal A10G GPU, billed per second.

    modal run train.py                 # train + push the LoRA adapter to HF Hub
    modal run train.py --action sample # spot-check 5 held-out measures: base vs finetuned

Prerequisites (one-time):
    pip install modal                  # done
    modal token new                    # authenticate (opens a browser)
    modal secret create hf-token HF_TOKEN=hf_xxx   # your HF write token, gated-Llama access
    # and accept the Llama 3.1 license at huggingface.co/meta-llama/Llama-3.1-8B-Instruct
"""

import modal

APP_NAME = "daxplain-train"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_REPO = "AaronHuang160/dax-explanations"
ADAPTER_REPO = "AaronHuang160/daxplain-8b"

USER_PROMPT = "Explain this DAX measure in plain English for a capable analyst:\n\n{dax}"

# The container image: declared in code, built once, cached. These run in Modal's
# cloud, NOT on the local machine. Versions pinned for a known-compatible QLoRA stack.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.44.2",
    "datasets==2.21.0",
    "peft==0.12.0",
    "trl==0.9.6",
    "bitsandbytes==0.43.3",
    "accelerate==0.33.0",
    "huggingface_hub==0.24.6",
    "rich==13.7.1",  # trl 0.9.6 imports it at load time but doesn't declare it
)

app = modal.App(APP_NAME, image=image)

# Persistent disk so the 16 GB base model is downloaded once, not every run.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

GPU_KW = dict(
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)


def _load_base(token: str):
    """Load Llama 3.1 8B in 4-bit (the frozen QLoRA base) + its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    tok.pad_token = tok.eos_token  # Llama has no pad token; reuse EOS

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token,
    )
    return model, tok


@app.function(**GPU_KW)
def train():
    import os
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    token = os.environ["HF_TOKEN"]
    model, tok = _load_base(token)

    # --- Data: format each row as a 2-turn chat via Llama's OWN chat template ---
    ds = load_dataset(DATASET_REPO, split="train", token=token)

    def to_text(row):
        messages = [
            {"role": "user", "content": USER_PROMPT.format(dax=row["dax"])},
            {"role": "assistant", "content": row["explanation"]},
        ]
        return {"text": tok.apply_chat_template(messages, tokenize=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"Training rows: {len(ds)}")
    print("Sample formatted row:\n", ds[0]["text"][:400], "\n---")

    # --- LoRA adapter: rank 16, alpha 32 (the only weights that train) ---
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- Trainer config: sensible QLoRA defaults (effective batch = 4 x 4 = 16) ---
    cfg = SFTConfig(
        output_dir="/tmp/daxplain",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        packing=False,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        tokenizer=tok,
    )
    trainer.train()

    # --- Publish just the small adapter (+ tokenizer) to HF Hub ---
    trainer.model.push_to_hub(ADAPTER_REPO, token=token)
    tok.push_to_hub(ADAPTER_REPO, token=token)
    print(f"Pushed LoRA adapter to https://huggingface.co/{ADAPTER_REPO}")


@app.function(**GPU_KW)
def sample(n: int = 5):
    """Run n held-out measures through base vs finetuned, side by side."""
    import os
    from datasets import load_dataset
    from peft import PeftModel

    token = os.environ["HF_TOKEN"]
    model, tok = _load_base(token)
    ds = load_dataset(DATASET_REPO, split="heldout", token=token).select(range(n))

    # One PeftModel; toggle the adapter on/off to compare against the true base.
    ft = PeftModel.from_pretrained(model, ADAPTER_REPO, token=token)

    def generate(dax):
        messages = [{"role": "user", "content": USER_PROMPT.format(dax=dax)}]
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(ft.device)
        out = ft.generate(
            **enc,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    for row in ds:
        print("=" * 90)
        print("DAX:", row["dax"])
        with ft.disable_adapter():  # adapter OFF -> true base model
            print("\n[BASE]\n", generate(row["dax"]))
        print("\n[FINETUNED]\n", generate(row["dax"]))  # adapter ON
        print()


@app.local_entrypoint()
def main(action: str = "train", n: int = 5):
    if action == "train":
        train.remote()
    elif action == "sample":
        sample.remote(n)
    else:
        raise SystemExit("action must be 'train' or 'sample'")
