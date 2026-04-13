from datasets import load_dataset
from transformers import AutoTokenizer
from . import config

def get_gdpr_dataset(tokenizer: AutoTokenizer, stage="sft", split='train[:]'):
    """Load and format the GDPR dataset based on training stage."""
    dataset = load_dataset(
        "sims2k/GDPR_QA_instruct_dataset",
        split=split,
        cache_dir=config.DATASET_CACHE_DIR
    )
    
    def format_sft(example):
        """Format for Supervised Fine-Tuning (Prompt + Response)."""
        instruction = example['instruction']
        input_text = example['input']
        response = example['output']
        
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "model", "content": response}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    def format_dpo(example):
        """Format for Direct Preference Optimization (Prompt + Chosen + Rejected)."""
        instruction = example['instruction']
        input_text = example['input']
        
        messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        chosen = example['output'] + tokenizer.eos_token
        # Note: In Stage 3, this will be replaced by actual SFT model rejections.
        rejected = example.get('rejected', "I'm not familiar with the specific GDPR regulations for this case.") + tokenizer.eos_token
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    if stage == "sft":
        dataset = dataset.map(format_sft, remove_columns=dataset.column_names)
    else:
        dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
        
    return dataset
