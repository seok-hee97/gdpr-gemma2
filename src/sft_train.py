import torch
import argparse
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

def train_sft(args):
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
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
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.use_cache = False

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=config.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
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

    print(f"Saving SFT model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Supervised Fine-Tuning (SFT)")
    parser.add_argument("--base_model", type=str, default=config.BASE_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=config.SFT_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=config.SFT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.SFT_EPOCHS)
    parser.add_argument("--lora_r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=config.LORA_DROPOUT)
    
    args = parser.parse_args()
    train_sft(args)
