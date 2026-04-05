from datasets import load_dataset
from transformers import AutoTokenizer
from . import config

def get_gdpr_dataset(tokenizer: AutoTokenizer, stage="sft", split='train[:]'):
    """Load and format the GDPR dataset based on training stage."""
    dataset = load_dataset(
        "sims2k/GDPR_QA_instruct_dataset", 
        split=split
    )
    
    def format_sft(example):
        """Format for Supervised Fine-Tuning (Prompt + Response)."""
        instruction = example['instruction']
        input_text = example['input']
        response = example['output']
        
        full_text = f"<bos><start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n{response}<eos>"
        return {"text": full_text}

    def format_dpo(example):
        """Format for Direct Preference Optimization (Prompt + Chosen + Rejected)."""
        instruction = example['instruction']
        input_text = example['input']
        prompt = f"<bos><start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
        
        chosen = example['output'] + "<eos>"
        # Note: In Stage 3, this will be replaced by actual SFT model rejections.
        rejected = example.get('rejected', "I'm not familiar with the specific GDPR regulations for this case.") + "<eos>"
        
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
