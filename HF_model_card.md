---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model: google/gemma-2-2b-it
tags:
- gemma
- gemma-2
- gdpr
- compliance
- legal
- dpo
- qlora
- sft
datasets:
- sims2k/GDPR_QA_instruct_dataset
model-index:
- name: gdpr_gemma-2-2b
  results:
  - task:
      type: text-generation
      name: GDPR Q&A
    dataset:
      type: sims2k/GDPR_QA_instruct_dataset
      name: GDPR_QA_instruct_dataset
      split: train[:100]
    metrics:
    - type: rouge
      name: ROUGE-L
      value: 0.2252
    - type: bleu
      name: BLEU
      value: 0.1034
    - type: bertscore
      name: BertScore F1
      value: 0.8527
---

# GDPR-Gemma-2-2B — GDPR Compliance Assistant

A specialized fine-tune of **`google/gemma-2-2b-it`** for English GDPR
(General Data Protection Regulation) Q&A. The model is aligned with expert
GDPR answers via a **3-stage pipeline** — Supervised Fine-Tuning, Dynamic
Rejection sampling, and Direct Preference Optimization (DPO) — using QLoRA
for resource-friendly training.

> **Disclaimer**: This model provides informational guidance only and **does
> not constitute legal advice**. Always consult a qualified legal
> professional for binding GDPR compliance decisions.

- 🔗 GitHub: <https://github.com/seok-hee97/gdpr-gemma2>
- 🧑‍💻 Author: **seok-hee97** (HF: `cycloevan`)
- 🏷️ Base: `google/gemma-2-2b-it`
- 🌐 Language: English

---

## Training Pipeline (3-Stage)

```
                ┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
 Base Gemma-2 ─►│ Stage 1: SFT │ ──► │ Stage 2: Dynamic   │ ──► │ Stage 3: DPO │
                │  (knowledge) │     │ Rejection Sampling │     │ (alignment)  │
                └──────────────┘     └────────────────────┘     └──────────────┘
```

| Stage | Goal | Method |
|---|---|---|
| 1. SFT | Inject GDPR domain knowledge | QLoRA SFT on expert Q&A |
| 2. Dynamic Rejection | Build *realistic* preference pairs | Sample SFT outputs (T=0.9) as `rejected`; expert answer = `chosen` |
| 3. DPO | Align preferences toward expert answers | DPO on top of SFT adapter (β=0.1) |

This pipeline is more faithful than naive DPO because Stage 2 produces
rejection candidates that match the model's *actual* failure modes, rather
than synthetic or generic wrong answers.

---

## Training Configuration

| Component | Value |
|---|---|
| Base model | `google/gemma-2-2b-it` |
| Quantization | 4-bit NF4 (QLoRA), bf16 compute |
| LoRA `r` / `alpha` / `dropout` | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| SFT epochs / LR | 3 / 2e-5 |
| DPO epochs / LR / β | 3 / 5e-6 / 0.1 |
| Batch size / Grad accum | 1 / 4 |
| Max prompt / total length | 1024 / 2048 |
| Optimizer | `paged_adamw_8bit` |
| Hardware | NVIDIA DGX Spark (CUDA, bf16) |

---

## Evaluation

Quantitative on 100 samples from `sims2k/GDPR_QA_instruct_dataset`;
qualitative via GPT-4o LLM-as-a-Judge on 10 samples (1–5 scale).

### Quantitative (ROUGE / BLEU / BertScore)

| Metric        | Base   | SFT        | **DPO (this model)** |
|---------------|--------|------------|----------------------|
| ROUGE-L       | 0.2072 | **0.2331** | 0.2252               |
| BLEU          | 0.0838 | **0.1146** | 0.1034               |
| BertScore F1  | 0.8432 | **0.8541** | 0.8527               |

### Qualitative (GPT-4o Judge, 1–5)

| Criterion             | Base | SFT  | **DPO (this model)** |
|-----------------------|------|------|----------------------|
| Legal Correctness     | 3.10 | 3.00 | **3.40**             |
| Article Accuracy      | 2.20 | 2.30 | **2.60**             |
| Compliance Alignment  | 3.70 | 3.40 | **3.80**             |
| Clarity               | **4.10** | **4.10** | 3.80         |

DPO improves legal correctness, GDPR-article citation accuracy, and
compliance alignment over both Base and SFT. It trades a small amount of
surface-level lexical overlap (ROUGE/BLEU) and clarity in exchange for
substantively more accurate legal content — a typical alignment trade-off.

---

## Quickstart

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "cycloevan/gdpr_gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # recommended for Gemma-2
)

SYSTEM = (
    "You are a professional GDPR compliance assistant. "
    "Provide accurate, legal, and clear guidance based on the General Data "
    "Protection Regulation."
)

def ask_gdpr(question: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": f"{SYSTEM}\n\nQuestion: {question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.2,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("model")[-1].strip() if "model" in text else text

print(ask_gdpr("What are the main principles of GDPR?"))
```

---

## Intended Use

- **In-scope**: Educational explanations of GDPR articles and principles,
  drafting first-pass compliance summaries, internal training material,
  GDPR-aware chatbot prototypes.
- **Out-of-scope**: Binding legal opinions, jurisdiction-specific advice
  outside the EU/EEA, regulated decisions affecting individuals' rights,
  enforcement/litigation strategy.

## Limitations & Risks

- **Snapshot of the regulation**: Trained on a static GDPR Q&A dataset;
  does not reflect post-training case law (CJEU rulings, EDPB guidelines)
  or national supervisory authority decisions.
- **English only**: No multilingual coverage; legal language outside English
  may degrade significantly.
- **Article-citation accuracy**: Average ~2.6/5 — the model occasionally
  cites incorrect or non-existent article numbers. Always verify citations
  against the official GDPR text.
- **Alignment trade-off**: DPO improves substantive legal accuracy at a
  small cost to surface fluency vs the SFT-only variant.
- **Hallucination**: As with any LLM, it can fabricate plausible-looking
  legal references. Treat outputs as drafts, not authoritative sources.

## Ethical Considerations

GDPR compliance affects individuals' fundamental rights to privacy and data
protection. Errors in legal interpretation may cause organisations to
mishandle personal data or mislead data subjects. Use only as a
decision-support tool, never as the sole basis for compliance actions.

## Citation

```bibtex
@misc{gdpr_gemma_2_2b_2024,
  title  = {GDPR-Gemma-2-2B: A 3-Stage Aligned GDPR Compliance Assistant},
  author = {seok-hee97},
  year   = {2024},
  howpublished = {Hugging Face Model Hub},
  url    = {https://huggingface.co/cycloevan/gdpr_gemma-2-2b}
}
```
