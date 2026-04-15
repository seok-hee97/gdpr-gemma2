"""
Generate targeted (controlled) rejection answers for DPO training.

Supports BOTH Upstage Solar and OpenAI (default: gpt-4o-mini — same pricing as Solar Pro).

Error types (rotated evenly):
  1. Wrong article number (e.g., Article 35 cited as Article 25)
  2. Misapplied legal principle (e.g., consent vs legitimate interest confusion)
  3. Fictional/non-existent rule (e.g., Article 6(1)(h) — doesn't exist)
  4. Scope confusion (applying GDPR where it doesn't apply)
  5. Incomplete but confident (missing key obligations, sounds authoritative)

Usage:
    # Default: OpenAI gpt-4o-mini (since Upstage credits expired)
    python -m src.generate_targeted_rejections --input data/gdpr_sft_combined.jsonl

    # Upstage Solar (if credits available)
    python -m src.generate_targeted_rejections --provider upstage --input data/gdpr_sft_combined.jsonl

    # Higher-quality OpenAI model (16x cost)
    python -m src.generate_targeted_rejections --model gpt-4o --input data/gdpr_sft_combined.jsonl
"""
import argparse
import json
import os
import random
import time
from openai import OpenAI
from tqdm import tqdm
from . import config


ERROR_TYPES = [
    {
        "name": "wrong_article",
        "instruction": (
            "Cite WRONG article numbers that sound plausible. For example, "
            "if the correct answer references Article 35 (DPIA), cite Article 25 "
            "(Privacy by Design) instead. Replace 1-3 article citations with nearby "
            "but incorrect numbers. Keep everything else accurate."
        ),
    },
    {
        "name": "misapplied_principle",
        "instruction": (
            "Confuse a key legal principle with a related but different one. "
            "For example, answer a consent question using 'legitimate interest' "
            "reasoning, or confuse 'data minimization' with 'storage limitation'. "
            "The confusion should be subtle and sound plausible."
        ),
    },
    {
        "name": "fictional_rule",
        "instruction": (
            "Invent 1-2 fictional GDPR provisions that don't exist. For example, "
            "cite 'Article 6(1)(h)' (doesn't exist — only a-f), or reference a "
            "'GDPR Chapter XII' (doesn't exist). Mix these with real provisions "
            "so the answer looks mostly correct."
        ),
    },
    {
        "name": "scope_confusion",
        "instruction": (
            "Apply GDPR rules to a situation where they don't fully apply, or "
            "omit a crucial geographic/material scope limitation. For example, "
            "claim GDPR applies to purely personal activities, or forget to mention "
            "that certain provisions only apply within the EEA."
        ),
    },
    {
        "name": "incomplete_confident",
        "instruction": (
            "Omit 1-2 critical obligations or requirements while sounding confident "
            "and complete. For example, describe breach notification but forget the "
            "72-hour deadline (Article 33), or explain DPIA without mentioning when "
            "it's mandatory. The omission should matter for compliance."
        ),
    },
]

SYSTEM_PROMPT = """You are a GDPR training data specialist. Your task is to generate a PLAUSIBLE BUT INCORRECT answer to a GDPR question.

Critical constraints:
1. Your incorrect answer MUST be approximately the SAME LENGTH (within ±15%) as the correct answer provided.
2. Your incorrect answer must SOUND professional and convincing — a non-expert should believe it.
3. The errors must be SUBTLE, not obvious — wrong article numbers, misapplied principles, or missing obligations.
4. Structure and formatting should MATCH the correct answer (bullet points, headers, etc.).
5. Output ONLY the incorrect answer text. No meta-commentary, no "here is the incorrect version" preamble."""


def build_prompt(instruction: str, correct_output: str, error_type: dict) -> str:
    target_len = len(correct_output)
    return f"""Here is a GDPR question and its CORRECT answer:

QUESTION:
{instruction}

CORRECT ANSWER ({target_len} characters):
{correct_output}

---

Now generate an INCORRECT answer with this specific error type:
**{error_type['name'].upper()}**: {error_type['instruction']}

Your incorrect answer must be {target_len} characters (±15%).
Write ONLY the incorrect answer, nothing else."""


def call_with_retry(client, model, messages, max_retries=3, **kwargs):
    """Call API with exponential backoff on rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)
                print(f"\n  Rate limited — retry {attempt+1}/{max_retries} after {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate_rejections(args):
    if not args.api_key:
        print(f"Error: API key not found for provider '{getattr(args, 'provider', 'unknown')}'. "
              f"Set OPENAI_API_KEY or UPSTAGE_API_KEY in .env, or pass --api_key.")
        return

    # OpenAI SDK uses its own default URL when base_url is None
    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Load input data (either original dataset or synthetic SFT data)
    if not os.path.isfile(args.input):
        print(f"Error: Input not found: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Resume support
    existing_count = 0
    if os.path.isfile(args.output):
        with open(args.output, "r") as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count > 0:
            print(f"Resuming: {existing_count} rejections already exist in {args.output}")

    records = records[existing_count:]
    total = existing_count + len(records)
    print(f"Loaded {len(records)} remaining pairs from {args.input} (total target: {total})")
    print(f"Generating targeted rejections using {args.model}...")

    generated = 0
    errors = 0

    write_mode = "a" if existing_count > 0 else "w"
    with open(args.output, write_mode, encoding="utf-8") as out_f:
        for i, record in enumerate(tqdm(records, initial=existing_count, total=total)):
            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            correct_output = record.get("output", "")

            if not correct_output.strip():
                continue

            # Rotate error types evenly (offset by existing_count for consistency)
            error_type = ERROR_TYPES[(existing_count + i) % len(ERROR_TYPES)]

            prompt = build_prompt(
                f"{instruction}\n\n{input_text}".strip(),
                correct_output,
                error_type,
            )

            try:
                response = call_with_retry(
                    client, args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                rejected = response.choices[0].message.content.strip()

                # Basic quality check — length within 50%-200% of correct
                ratio = len(rejected) / max(len(correct_output), 1)
                if 0.5 < ratio < 2.0:
                    out_f.write(json.dumps({
                        "instruction": instruction,
                        "input": input_text,
                        "output": correct_output,  # chosen
                        "rejected": rejected,       # targeted incorrect
                    }, ensure_ascii=False) + "\n")
                    out_f.flush()
                    generated += 1
                else:
                    errors += 1

            except Exception as e:
                errors += 1
                print(f"\n  Error at {existing_count + i}: {e}")

            time.sleep(0.7)

    # Quality report
    total_generated = existing_count + generated
    if total_generated > 0:
        # Re-read full output for stats
        with open(args.output, "r") as f:
            all_results = [json.loads(line) for line in f if line.strip()]
        c_lens = [len(r["output"]) for r in all_results]
        r_lens = [len(r["rejected"]) for r in all_results]
        import statistics
        avg_ratio = statistics.mean(r_lens) / statistics.mean(c_lens)

        print(f"\n{'=' * 60}")
        print(f"TARGETED REJECTION GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  New:       {generated} pairs")
        print(f"  Previous:  {existing_count} pairs")
        print(f"  Total:     {total_generated} pairs")
        print(f"  Errors:    {errors}")
        print(f"  Avg length ratio (rejected/chosen): {avg_ratio:.2f}")
        print(f"    (target: ~1.0, previous self-generated: 0.56)")
        print(f"  Output:    {args.output}")


PROVIDER_DEFAULTS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": config.OPENAI_API_KEY,
        "base_url": None,  # OpenAI default (SDK uses https://api.openai.com/v1)
    },
    "upstage": {
        "model": config.UPSTAGE_MODEL,
        "api_key": config.UPSTAGE_API_KEY,
        "base_url": config.UPSTAGE_BASE_URL,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate targeted DPO rejections with controlled error types"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(config.DATA_DIR, "gdpr_sft_combined.jsonl"),
        help="Input JSONL with instruction/input/output fields",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(config.DATA_DIR, "gdpr_targeted_dpo.jsonl"),
        help="Output JSONL with chosen + targeted rejected pairs",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "upstage"],
        help="API provider (default: openai — since Upstage credits expired)",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-set by --provider if omitted)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (auto-loaded from .env if omitted)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Override base URL (leave unset for provider default)")
    args = parser.parse_args()

    # Apply provider defaults for any unset fields
    defaults = PROVIDER_DEFAULTS[args.provider]
    if args.model is None:
        args.model = defaults["model"]
    if args.api_key is None:
        args.api_key = defaults["api_key"]
    if args.base_url is None:
        args.base_url = defaults["base_url"]

    print(f"Provider: {args.provider} | Model: {args.model} | "
          f"Base URL: {args.base_url or '(default OpenAI)'}")
    generate_rejections(args)
