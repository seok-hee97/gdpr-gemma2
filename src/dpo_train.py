import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig
from trl import DPOTrainer
from datasets import load_dataset
from . import config

def train_dpo():
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, cache_dir=config.CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dynamic Dataset (generated in Stage 2)
    print(f"Loading dynamic dataset from {config.DYNAMIC_DATASET_PATH}")
    dataset = load_dataset("json", data_files=config.DYNAMIC_DATASET_PATH, split="train", cache_dir=config.CACHE_DIR)

    def format_dpo(example):
        prompt = f"<bos><start_of_turn>user\n{example['instruction']}\n\n{example['input']}<end_of_turn>\n<start_of_turn>model\n"
        return {
            "prompt": prompt,
            "chosen": example['output'] + "<eos>",
            "rejected": example['rejected'] + "<eos>"
        }
    
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

    # 3. Model Configuration (Load SFT-trained adapter)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Note: DPO starts from the SFT model
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.CACHE_DIR
    )
    # Important: Load SFT weights before DPO
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, config.SFT_MODEL_PATH, is_trainable=True)

    # 4. LoRA Configuration (Reuse or Define New)
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.DPO_MODEL_PATH,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.DPO_LEARNING_RATE,
        num_train_epochs=config.DPO_EPOCHS,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    # 6. DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # PEFT handles reference model internally to save VRAM
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=config.DPO_BETA,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_length=config.MAX_LENGTH,
    )

    print(f"Starting Stage 3: DPO Alignment with {len(dataset)} samples...")
    trainer.train()

    print(f"Saving DPO model to {config.DPO_MODEL_PATH}...")
    trainer.model.save_pretrained(config.DPO_MODEL_PATH)
    tokenizer.save_pretrained(config.DPO_MODEL_PATH)
    print("DPO Training Complete.")

if __name__ == "__main__":
    train_dpo()
