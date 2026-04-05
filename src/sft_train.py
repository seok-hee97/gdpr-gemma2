import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig
from trl import SFTTrainer
from . import config
from .data_loader import get_gdpr_dataset

def train_sft():
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load SFT Dataset
    dataset = get_gdpr_dataset(tokenizer, stage="sft")

    # 3. Model Configuration (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.use_cache = False

    # 4. LoRA Configuration
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
        output_dir=config.SFT_MODEL_PATH,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.SFT_LEARNING_RATE,
        num_train_epochs=config.SFT_EPOCHS,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        report_to="none"
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.MAX_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"Starting Stage 1: SFT Training with {len(dataset)} samples...")
    trainer.train()

    print(f"Saving SFT model to {config.SFT_MODEL_PATH}...")
    trainer.model.save_pretrained(config.SFT_MODEL_PATH)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)
    print("SFT Training Complete.")

if __name__ == "__main__":
    train_sft()
