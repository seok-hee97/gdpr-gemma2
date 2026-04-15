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
    config.ensure_base_model()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load SFT Dataset with Validation Split (90/10)
    full_dataset = get_gdpr_dataset(
        tokenizer, stage="sft", local_path=args.local_path
    )
    dataset_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 3. Model Configuration
    load_kwargs = {
        "torch_dtype": config.TORCH_DTYPE,
        "device_map": "auto",
    }
    if config.USE_QUANTIZATION:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.TORCH_DTYPE
        )
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", **load_kwargs
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
        evaluation_strategy="epoch", # 매 에폭마다 평가
        logging_steps=10,
        bf16=(config.DEVICE == "cuda"),
        fp16=False,
        report_to="none",
        load_best_model_at_end=True # 가장 성능 좋은 모델 저장
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.MAX_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"Starting Stage 1: SFT Training with {len(train_dataset)} train and {len(eval_dataset)} eval samples...")
    trainer.train()

    print(f"Saving SFT model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Supervised Fine-Tuning (SFT)")
    parser.add_argument("--base_model", type=str, default=config.BASE_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=config.SFT_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=config.SFT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.SFT_EPOCHS)
    parser.add_argument("--lora_r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=config.LORA_DROPOUT)
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Path to local JSONL dataset (overrides HF dataset). "
             "Must have instruction/input/output fields.",
    )

    args = parser.parse_args()
    train_sft(args)
