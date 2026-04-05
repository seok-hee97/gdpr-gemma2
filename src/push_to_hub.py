import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from . import config

def merge_and_push():
    print(f"Loading Base Model: {config.BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, use_fast=False)

    print(f"Loading Adapter from: {config.OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    
    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    print(f"Pushing to Hub: {config.NEW_MODEL_NAME}")
    model.push_to_hub(config.NEW_MODEL_NAME, private=True)
    tokenizer.push_to_hub(config.NEW_MODEL_NAME, private=True)
    print("Upload Complete.")

if __name__ == "__main__":
    merge_and_push()
