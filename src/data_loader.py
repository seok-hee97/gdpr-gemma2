import os
from datasets import load_dataset
from transformers import AutoTokenizer
from . import config


def get_gdpr_dataset(tokenizer: AutoTokenizer, stage="sft", split='train[:]',
                     local_path: str | None = None):
    """Load and format the GDPR dataset based on training stage.

    Args:
        tokenizer: HF tokenizer with chat template.
        stage: "sft" or "dpo" — chooses formatting function.
        split: HF split spec (used only when loading the HF dataset).
        local_path: If provided, load JSONL from this local path instead of HF.
                    JSONL must have fields: instruction, input, output
                    (and 'rejected' for DPO stage).
    """
    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local dataset not found: {local_path}")
        print(f"Loading local JSONL: {local_path}")
        dataset = load_dataset("json", data_files=local_path, split="train")
    else:
        print(f"Loading HF dataset: sims2k/GDPR_QA_instruct_dataset[{split}]")
        dataset = load_dataset(
            "sims2k/GDPR_QA_instruct_dataset",
            split=split,
            cache_dir=config.DATASET_CACHE_DIR,
        )

    def format_sft(example):
        """Format for Supervised Fine-Tuning (Prompt + Response)."""
        instruction = example['instruction']
        input_text = example.get('input', '') or ''
        response = example['output']

        user_content = f"{instruction}\n\n{input_text}".strip()
        messages = [
            {"role": "user", "content": user_content},
            {"role": "model", "content": response},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    def format_dpo(example):
        """Format for Direct Preference Optimization (Prompt + Chosen + Rejected)."""
        instruction = example['instruction']
        input_text = example.get('input', '') or ''

        user_content = f"{instruction}\n\n{input_text}".strip()
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        chosen = example['output'] + tokenizer.eos_token
        rejected = example.get(
            'rejected',
            "I'm not familiar with the specific GDPR regulations for this case."
        ) + tokenizer.eos_token

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    if stage == "sft":
        dataset = dataset.map(format_sft, remove_columns=dataset.column_names)
    else:
        dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

    return dataset
