import torch
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, PeftModel
from trl import DPOTrainer
from datasets import load_dataset
from . import config

def train_dpo(args):
    set_seed(42)

    # 1. Load Tokenizer (SFT 단계에서 사용된 것과 동일하게)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dynamic Dataset
    print(f"Loading dynamic dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_dpo(example):
        # ChatML 포맷 유지
        prompt = f"<bos><start_of_turn>user\n{example['instruction']}\n\n{example['input']}<end_of_turn>\n<start_of_turn>model\n"
        return {
            "prompt": prompt,
            "chosen": example['output'] + "<eos>",
            "rejected": example['rejected'] + "<eos>"
        }
    
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

    # 3. Model Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Base Model 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # [중요] Stage 1에서 학습한 SFT 어댑터를 먼저 로드
    print(f"Loading SFT adapter from {args.sft_model_path} as a starting point for DPO...")
    model = PeftModel.from_pretrained(base_model, args.sft_model_path, is_trainable=True)

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
        remove_unused_columns=False,
        report_to="none"
    )

    # 6. DPO Trainer (PEFT 모델 전달 시 ref_model은 자동 처리됨)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=args.beta,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_length=config.MAX_LENGTH,
    )

    print(f"Starting Stage 3: DPO Alignment with {len(dataset)} samples...")
    trainer.train()

    print(f"Saving final DPO model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Direct Preference Optimization (DPO)")
    parser.add_argument("--base_model", type=str, default=config.BASE_MODEL_NAME)
    parser.add_argument("--sft_model_path", type=str, default=config.SFT_MODEL_PATH)
    parser.add_argument("--dataset_path", type=str, default=config.DYNAMIC_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=config.DPO_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=config.DPO_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.DPO_EPOCHS)
    parser.add_argument("--beta", type=float, default=config.DPO_BETA)
    parser.add_argument("--lora_r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=config.LORA_DROPOUT)
    
    args = parser.parse_args()
    train_dpo(args)
