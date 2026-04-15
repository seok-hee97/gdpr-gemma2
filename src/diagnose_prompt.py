"""
Diagnostic: compare model outputs under two prompt formats to test the
"train-inference prompt mismatch" hypothesis.

  Format A — current inference.py (with system prompt + "Question:" prefix)
  Format B — training-matched (raw user message, no system prompt)

Usage (DPO only, fastest — ~2 min on M1 / ~30 s on GPU):
    python -m src.diagnose_prompt

Compare SFT and DPO together (slower — ~4 min on M1):
    python -m src.diagnose_prompt --models sft,dpo

Include base as control (slowest — ~6 min on M1):
    python -m src.diagnose_prompt --models base,sft,dpo
"""
import argparse
import os
import textwrap
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from . import config


SYSTEM_PROMPT = (
    "You are a professional GDPR compliance assistant. "
    "Provide accurate, legal, and clear guidance based on the General "
    "Data Protection Regulation."
)

MODEL_PATHS = {
    "base": config.BASE_MODEL_PATH,
    "sft": config.SFT_MODEL_PATH,
    "dpo": config.DPO_MODEL_PATH,
}


def load_model(stage: str):
    """Load base + (optional) adapter on CPU/MPS in bf16."""
    config.ensure_base_model()
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_PATH,
        torch_dtype=config.TORCH_DTYPE,
        device_map="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if stage == "base":
        return tokenizer, base
    return tokenizer, PeftModel.from_pretrained(base, MODEL_PATHS[stage])


def make_prompt_A(tokenizer, question: str) -> str:
    """Current inference format — system prompt + 'Question:' prefix."""
    user_content = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_prompt_B(tokenizer, question: str) -> str:
    """Training-matched format — raw user message, no system prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy → deterministic, comparable
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract only the model's reply (after the last "model" turn)
    if "model\n" in full:
        return full.split("model\n", 1)[-1].strip()
    return full.strip()


def load_eval_questions(n: int):
    """Pull a few real questions from the GDPR eval dataset."""
    ds = load_dataset(
        "sims2k/GDPR_QA_instruct_dataset",
        split=f"train[:{n}]",
        cache_dir=config.DATASET_CACHE_DIR,
    )
    return [
        f"{ex['instruction']}\n\n{ex['input']}".strip()
        for ex in ds
    ]


def fmt_block(text: str, indent: str = "  ") -> str:
    """Wrap text to ~88 cols and indent for readable side-by-side output."""
    wrapped = textwrap.fill(text, width=88, replace_whitespace=False)
    return "\n".join(indent + line for line in wrapped.splitlines())


def run_diagnostic(args):
    questions = load_eval_questions(args.num_questions)
    stages = [s.strip() for s in args.models.split(",") if s.strip()]
    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    emit("=" * 90)
    emit(f"PROMPT-FORMAT DIAGNOSTIC  ({args.num_questions} questions × 2 formats × {len(stages)} models)")
    emit("=" * 90)

    for stage in stages:
        emit("")
        emit(f"\n>>> Loading model: {stage.upper()}  ({MODEL_PATHS[stage]})")
        tokenizer, model = load_model(stage)
        model.eval()

        for q_idx, question in enumerate(questions, 1):
            emit("")
            emit("─" * 90)
            emit(f"[{stage.upper()}] Q{q_idx}: {question[:200]}{'...' if len(question) > 200 else ''}")
            emit("─" * 90)

            prompt_A = make_prompt_A(tokenizer, question)
            prompt_B = make_prompt_B(tokenizer, question)

            ans_A = generate(model, tokenizer, prompt_A, args.max_new_tokens)
            ans_B = generate(model, tokenizer, prompt_B, args.max_new_tokens)

            emit("")
            emit("[Format A — current inference (system prompt + 'Question:')]")
            emit(fmt_block(ans_A))
            emit("")
            emit("[Format B — training-matched (raw user message)]")
            emit(fmt_block(ans_B))

        # Free memory between models
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    emit("")
    emit("=" * 90)
    emit("DIAGNOSTIC COMPLETE — manually compare A vs B for each (model, question).")
    emit("If B answers are clearly better/different on SFT/DPO → mismatch is real → run option 1.")
    emit("If A and B look nearly identical → prompt format is NOT the regression cause.")
    emit("=" * 90)

    # Save to file for later review
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "diagnose_prompt.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\nSaved transcript to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose train-inference prompt mismatch (Format A vs B)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="dpo",
        help="Comma-separated stages to test: base,sft,dpo (default: dpo)",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=3,
        help="Number of questions to test (default: 3)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Generation length cap (default: 300)",
    )
    args = parser.parse_args()
    run_diagnostic(args)
