---
library_name: transformers
tags:
- gemma
- gdpr
- compliance
- dpo
- qlora
base_model: google/gemma-2-2b-it
model_name: GDPR-Gemma-2-2B
---

# GDPR-Gemma-2-2B: AI-Powered GDPR Compliance Assistant

## Model Description
GDPR-Gemma-2-2B is a specialized AI model designed to provide guidance on GDPR (General Data Protection Regulation) compliance. By fine-tuning Google's Gemma-2B-it model using Direct Preference Optimization (DPO) and a GDPR-specific dataset, we've created a tool that aligns with data protection principles and provides accurate regulatory insights.

- **Developed by:** seok-hee97
- **Model type:** Causal Language Model
- **Language(s):** English (GDPR Context)
- **Finetuned from model:** google/gemma-2-2b-it
- **Training Method:** QLoRA + DPO (Direct Preference Optimization)

## Key Features
- Specialized in GDPR compliance and data protection inquiries.
- Aligned with GDPR principles via DPO to reduce hallucination in regulatory context.
- Efficient 4-bit quantization for resource-friendly execution.

## How to Get Started
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "cycloevan/gpdr_gemma_2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def ask_gdpr(question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(ask_gdpr("What are the main principles of GDPR?"))
```

## Training Details
- **Dataset:** [sims2k/GDPR_QA_instruct_dataset](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset)
- **Epochs:** 10
- **Learning Rate:** 5e-6
- **Batch Size:** 1 (with Gradient Accumulation 3)
- **LoRA Config:** r=16, alpha=32, target_modules="all-linear"
- **DPO Config:** beta=0.1

## Bias, Risks, and Limitations
This model is intended for guidance purposes only and **does not constitute legal advice**. Users should consult with legal professionals for official GDPR compliance audits.
