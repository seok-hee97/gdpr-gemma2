"""
Push GDPR fine-tuning datasets to the Hugging Face Hub.

Two datasets produced during Phase 5:
  1. SFT instruct dataset (2,277 pairs) — original 316 + Upstage Solar synthetic 1,961
  2. Targeted DPO preference dataset (2,277 pairs) — GPT-4o-mini generated rejections
      with 5 controlled error types, length-matched (±15%) to chosen.

Usage:
    huggingface-cli login   # once (or set HF_TOKEN in .env)
    python -m src.push_datasets --which sft
    python -m src.push_datasets --which dpo
    python -m src.push_datasets --which all
"""
import argparse
import json
import os
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from . import config


SFT_REPO_ID = "cycloevan/gdpr-sft-2277-combined"
DPO_REPO_ID = "cycloevan/gdpr-dpo-2277-targeted"

SFT_PATH = os.path.join(config.DATA_DIR, "gdpr_sft_combined.jsonl")
DPO_PATH = os.path.join(config.DATA_DIR, "gdpr_targeted_dpo.jsonl")


SFT_CARD = """---
license: apache-2.0
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- gdpr
- compliance
- legal
- privacy
- instruction-tuning
- sft
size_categories:
- 1K<n<10K
pretty_name: GDPR SFT Combined (2,277 pairs)
---

# GDPR SFT Combined Dataset (2,277 instructions)

Supervised Fine-Tuning dataset for GDPR (General Data Protection Regulation)
compliance Q&A in English. Produced during the `gdpr-gemma2` Phase 5
experiment.

## Composition

| Source | Count | Method |
|---|---|---|
| `sims2k/GDPR_QA_instruct_dataset` | 316 | Original expert-authored pairs |
| Upstage Solar-Pro synthetic | 1,961 | LLM-generated, deduplicated (from 2,812 raw → 1,961 unique) |
| **Total** | **2,277** | |

The synthetic half covers **101 GDPR topics** across 9 categories:

- Chapter I–IX (core articles, 46 topics)
- SCENARIO (business scenarios, 10 topics)
- SECTOR (healthcare / fintech / HR, 12 topics)
- ADVANCED (sub-processor / BCR / one-stop-shop, 8 topics)
- SPECIAL_DATA (biometric / genetic / children, 6 topics)
- WORKFLOW (DPIA / ROPA / breach, 7 topics)
- ENFORCEMENT (fines / case law, 5 topics)
- CROSS_REG (ePrivacy / AI Act / NIS2, 4 topics)
- INTL (EU-US DPF / UK GDPR / LGPD, 3 topics)

Three question types rotated: `article_focused`, `scenario_based`,
`comparative`.

## Schema

```json
{
  "instruction": "How does the GDPR distinguish ...",
  "input":       "Differentiate between ...",
  "output":      "The GDPR distinguishes between ..."
}
```

## Quick load

```python
from datasets import load_dataset
ds = load_dataset("cycloevan/gdpr-sft-2277-combined", split="train")
print(ds[0])
```

## Known limitations

- **Synthetic duplicate rate**: ~30% of raw Solar-Pro generations overlapped
  in instruction prefix; deduplicated at the instruction level (longest
  output kept).
- **English only**.
- **Not a retrieval substitute**: article citations inside outputs are
  model-generated and should be verified against the official GDPR text.
- **Empirically does not lift a 2B SFT above Base**: in the source project,
  training `gemma-2-2b-it` on this data produced SFT v2 ≈ Base on all four
  GPT-4o qualitative criteria (see Phase 5 findings).

## Intended use

- SFT / instruction-tuning for GDPR compliance assistants.
- As a starting point for larger base models (9B+) where data quantity
  may translate to measurable quality lift.
- For studying synthetic-data duplication behaviour under rotated topic
  prompts at temperature 0.7.

## Citation

```bibtex
@misc{gdpr_sft_combined_2026,
  title  = {GDPR SFT Combined: 2,277 GDPR Instruct Pairs (Original + Upstage Solar Synthetic)},
  author = {seok-hee97},
  year   = {2026},
  howpublished = {Hugging Face Datasets},
  url    = {https://huggingface.co/datasets/cycloevan/gdpr-sft-2277-combined}
}
```

## License

Apache-2.0. Downstream users are responsible for ensuring compliance with
the license terms of the seed dataset (`sims2k/GDPR_QA_instruct_dataset`)
and of Upstage API outputs used in the synthetic half.
"""


DPO_CARD = """---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- gdpr
- compliance
- legal
- privacy
- dpo
- preference-learning
- rlhf
size_categories:
- 1K<n<10K
pretty_name: GDPR DPO Targeted Rejections (2,277 pairs)
---

# GDPR DPO Targeted Rejections (2,277 preference pairs)

Direct Preference Optimization (DPO) dataset for GDPR compliance Q&A in
English. Unlike self-play rejections (same model's degraded outputs), this
dataset uses an **external LLM (GPT-4o-mini) to generate rejections with
five controlled error types**, length-matched to the chosen answer.

## How rejections were generated

Rejections deliberately introduce one of five controlled error types, evenly
distributed:

| Error Type | Description | DPO Signal Learned |
|---|---|---|
| `wrong_article` | Cite plausible but incorrect article numbers | correct citation > wrong citation |
| `misapplied_principle` | Confuse consent with legitimate interest, etc. | correct principle > misapplied |
| `fictional_rule` | Invent non-existent articles (e.g., Art 6(1)(h)) | real provisions > fabricated |
| `scope_confusion` | Apply GDPR where it doesn't apply | correct scope > over/under-application |
| `incomplete_confident` | Omit critical obligations while sounding sure | complete > overconfident partial |

### Quality contrast vs self-play rejections

| Metric | Self-play (typical) | **This dataset** | Target |
|---|---|---|---|
| Length ratio (rejected/chosen) | 0.56 | **1.06** | ~1.0 |
| Within ±15% target range | — | **86.7%** | — |
| Article citation gap | 2.34/sample | **0.18/sample** | ~0 |
| JSON parsing success | — | 100% | — |

This length-matching and citation-density-matching is intentional: it
prevents DPO from learning the trivial shortcut "longer + more citations =
better" that plagues self-play data.

## Schema

```json
{
  "instruction": "Detail the obligations of ...",
  "input":       "Can you detail the specific obligations ...",
  "output":      "Under the Clinical Trials Regulation (CTR), ... Article 47 ... Article 56 ...",
  "rejected":    "Under the Clinical Trials Regulation (CTR), ... Article 45 ... Article 54 ..."
}
```

`output` is the "chosen" answer; `rejected` is the targeted wrong answer.

## Quick load (for DPO training)

```python
from datasets import load_dataset
from trl import DPOTrainer

ds = load_dataset("cycloevan/gdpr-dpo-2277-targeted", split="train")

def to_dpo_format(ex, tokenizer):
    messages = [{"role": "user", "content": f"{ex['instruction']}\\n\\n{ex['input']}"}]
    return {
        "prompt":   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        "chosen":   ex["output"],
        "rejected": ex["rejected"],
    }
```

## Honest empirical findings

In the source project (`gdpr-gemma2`, Gemma-2-2B-it, QLoRA r=16, β=0.1,
sigmoid DPO loss, 3 epochs), training on this dataset produced a **DPO v3-external**
model that:

- **Beats self-play DPO** on all four GPT-4o qualitative criteria (Legal
  Correctness +0.52, Compliance +0.74, Clarity +0.56). This confirms the
  direction expected from the Zephyr/UltraFeedback literature (external
  preference > self-play).
- **Does NOT beat Base `gemma-2-2b-it`** on any qualitative criterion.
  N=2,277 is insufficient to overcome the Base ceiling for a 2B model on
  legal Q&A — consistent with DPO's documented signal-to-noise scaling
  behaviour (Zephyr uses 60k, Tulu-2 uses 32k).

So: **the dataset provides a cleaner preference signal than self-play but
cannot by itself lift a near-ceiling 2B base**. It is expected to yield
measurable improvement on larger bases (9B+) or when combined with
retrieval augmentation.

## Known limitations

- **English only**.
- **Error types are synthetic**: not sampled from real model failure
  patterns; simulated via prompt engineering of GPT-4o-mini.
- **Generated by a 2024-class proprietary model**: fidelity may degrade
  as GDPR evolves.
- **Not a benchmark**: intended for training, not for evaluation. Use
  `sims2k/GDPR_QA_instruct_dataset` or held-out custom test sets for
  evaluation.

## Citation

```bibtex
@misc{gdpr_dpo_targeted_2026,
  title  = {GDPR DPO Targeted Rejections: 2,277 length-matched preference pairs with 5 controlled error types},
  author = {seok-hee97},
  year   = {2026},
  howpublished = {Hugging Face Datasets},
  url    = {https://huggingface.co/datasets/cycloevan/gdpr-dpo-2277-targeted}
}
```

## License

Apache-2.0. Downstream users are responsible for compliance with OpenAI's
terms of service for content generated via `gpt-4o-mini`.
"""


def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def push_one(jsonl_path: str, repo_id: str, card: str, private: bool):
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Dataset JSONL not found: {jsonl_path}")
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Check .env or run `huggingface-cli login`.")

    print(f"[1/4] Loading {jsonl_path}")
    rows = _load_jsonl(jsonl_path)
    print(f"       {len(rows)} rows loaded")

    print(f"[2/4] Creating dataset object")
    ds = Dataset.from_list(rows)

    print(f"[3/4] Ensuring repo: {repo_id} (private={private})")
    create_repo(
        repo_id=repo_id,
        token=config.HF_TOKEN,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    print(f"[4/4] Pushing dataset to {repo_id}")
    ds.push_to_hub(repo_id, token=config.HF_TOKEN, private=private)

    # Upload the README (dataset card)
    api = HfApi(token=config.HF_TOKEN)
    tmp_card = jsonl_path + ".README.md.tmp"
    with open(tmp_card, "w", encoding="utf-8") as f:
        f.write(card)
    api.upload_file(
        path_or_fileobj=tmp_card,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: dataset card",
    )
    os.remove(tmp_card)

    print(f"\n✅ {repo_id} pushed: https://huggingface.co/datasets/{repo_id}\n")


def main(args):
    which = args.which.lower()
    if which in ("sft", "all"):
        push_one(SFT_PATH, SFT_REPO_ID, SFT_CARD, args.private)
    if which in ("dpo", "all"):
        push_one(DPO_PATH, DPO_REPO_ID, DPO_CARD, args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push GDPR datasets to HF Hub")
    parser.add_argument(
        "--which",
        type=str,
        choices=["sft", "dpo", "all"],
        default="all",
        help="Which dataset(s) to push",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as private repo (default: public)",
    )
    args = parser.parse_args()
    main(args)
