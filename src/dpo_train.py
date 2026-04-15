import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import PeftModel
from trl import DPOTrainer
from datasets import load_dataset
from . import config

def train_dpo(args):
    set_seed(42)
    config.ensure_base_model()

    # 1. Load Tokenizer (SFT 단계에서 사용된 것과 동일하게)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dynamic Dataset
    print(f"Loading dynamic dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_dpo(example):
        messages = [{"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "chosen": example['output'] + tokenizer.eos_token,
            "rejected": example['rejected'] + tokenizer.eos_token
        }
    
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
    
    # Validation Split
    dataset_split = dataset.train_test_split(test_size=0.1)
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

    # Base Model 로드 (Gemma2: sliding window + softmax capping 호환을 위해 eager 사용)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", **load_kwargs
    )

    # [중요] Stage 1에서 학습한 SFT 어댑터를 먼저 로드
    print(f"Loading SFT adapter from {args.sft_model_path} as a starting point for DPO...")
    model = PeftModel.from_pretrained(base_model, args.sft_model_path, is_trainable=True)

    # [필수] Gradient checkpointing 호환 설정
    model.config.use_cache = False              # use_cache와 gradient checkpointing은 충돌
    model.enable_input_require_grads()           # quantized + PEFT 모델에서 gradient checkpointing 작동에 필수

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        evaluation_strategy="no",   # epoch 경계 eval 제거 → 피크 메모리 방지
        logging_steps=10,
        bf16=(config.DEVICE == "cuda"),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=False,  # best model 메모리 유지 제거 → 마지막 epoch 모델 저장
        optim="paged_adamw_8bit",  # QLoRA 표준: optimizer state 메모리 75% 절감 + OOM 시 CPU offload
        # Gradient Checkpointing: activation 메모리 60-70% 절약 (DPO의 OOM 방지 핵심)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT 모델 호환 필수
    )

    # 5. DPO Trainer (SFT PEFT 모델이 이미 로드되어 있으므로 peft_config 불필요)
    #    loss_type="ipo" — more robust to noisy preference pairs than the
    #    default sigmoid loss (see config.DPO_LOSS_TYPE for rationale).
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=args.beta,
        loss_type=args.loss_type,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_length=config.MAX_LENGTH,
    )

    print(f"Starting Stage 3: DPO Alignment with {len(train_dataset)} train and {len(eval_dataset)} eval samples...")
    trainer.train()

    print(f"Saving final DPO model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Direct Preference Optimization (DPO)")
    parser.add_argument("--base_model", type=str, default=config.BASE_MODEL_PATH)
    parser.add_argument("--sft_model_path", type=str, default=config.SFT_MODEL_PATH)
    parser.add_argument("--dataset_path", type=str, default=config.DYNAMIC_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=config.DPO_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=config.DPO_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.DPO_EPOCHS)
    parser.add_argument("--beta", type=float, default=config.DPO_BETA)
    parser.add_argument(
        "--loss_type",
        type=str,
        default=config.DPO_LOSS_TYPE,
        choices=["sigmoid", "ipo", "hinge", "kto_pair"],
        help="DPO loss variant (default 'ipo' — robust to noisy pairs)",
    )
    args = parser.parse_args()
    train_dpo(args)
