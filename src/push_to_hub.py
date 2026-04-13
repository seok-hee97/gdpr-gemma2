"""
Merge a LoRA/QLoRA adapter into the base Gemma-2-2B-it weights and push the
full (merged) model + tokenizer + model card to the Hugging Face Hub.

Usage:
    huggingface-cli login   # once
    python -m src.push_to_hub \
        --adapter_path ./models/gemma-2b-gdpr-dpo \
        --repo_id cycloevan/gdpr_gemma-2-2b \
        --model_card ./HF_model_card.md
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo
from . import config


def merge_and_push(args):
    # 0) Sanity checks
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter not found: {args.adapter_path}")
    if not os.path.isfile(os.path.join(args.adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"adapter_config.json missing in {args.adapter_path}"
        )

    # 1) Ensure base model is local (avoids re-downloading from Hub)
    config.ensure_base_model()
    base_path = config.BASE_MODEL_PATH

    # 2) Load base in bf16 ON CPU (one-shot merge — no GPU needed, avoids
    #    disk-offload on memory-limited devices like M1 Macs).
    print(f"[1/5] Loading base model from {base_path} (device=cpu, dtype=bf16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},          # force everything on CPU — no offload
        low_cpu_mem_usage=True,
        attn_implementation="eager",     # Gemma-2 sliding-window compat
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # 3) Apply the LoRA adapter (also on CPU)
    print(f"[2/5] Loading adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(
        base_model, args.adapter_path, device_map={"": "cpu"}
    )

    # 4) Merge LoRA weights into the base — produces a standalone model
    print("[3/5] Merging adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    # 5) Create the repo (no-op if it already exists) and push
    api = HfApi(token=config.HF_TOKEN)
    print(f"[4/5] Ensuring repo exists: {args.repo_id} (private={args.private})")
    create_repo(
        repo_id=args.repo_id,
        token=config.HF_TOKEN,
        private=args.private,
        exist_ok=True,
    )

    commit_msg = args.commit_message or (
        "Update: merged DPO adapter (3-stage SFT -> Dynamic Rejection -> DPO)"
    )

    print(f"[5/5] Pushing merged model + tokenizer to {args.repo_id}")
    model.push_to_hub(
        args.repo_id,
        token=config.HF_TOKEN,
        private=args.private,
        commit_message=commit_msg,
        safe_serialization=True,
    )
    tokenizer.push_to_hub(
        args.repo_id,
        token=config.HF_TOKEN,
        private=args.private,
        commit_message=commit_msg,
    )

    # 6) Upload the model card (README.md on the Hub)
    if args.model_card and os.path.isfile(args.model_card):
        print(f"Uploading model card: {args.model_card} -> README.md")
        api.upload_file(
            path_or_fileobj=args.model_card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message=f"docs: update model card ({commit_msg})",
        )
    else:
        print(f"[warn] model card not found at {args.model_card} — skipping")

    print(f"\nUpload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and push full model to Hugging Face Hub"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=config.DPO_MODEL_PATH,
        help="Local path to the trained adapter (default: DPO model)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="cycloevan/gdpr_gemma-2-2b",
        help="Target Hugging Face repo id (org_or_user/name)",
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default=os.path.join(config.PROJECT_ROOT, "HF_model_card.md"),
        help="Path to the model card markdown to upload as README.md",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as a private repo (default: public — preserves current visibility)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Override the commit message used on the Hub",
    )
    args = parser.parse_args()
    merge_and_push(args)
