import argparse
import torch

# PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
# transformers<4.46 checkpoint resume (rng_state.pth contains numpy globals).
# Force weights_only=False for all torch.load calls in this process.
_orig_torch_load = torch.load
def _compat_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _compat_torch_load

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dynamic Dataset
    print(f"Loading dynamic dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_dpo(example):
        messages = [{"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "chosen": example['output'],
            "rejected": example['rejected'],
        }
    
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
    
    # Validation Split (seed 고정 — resume 시 동일 분할 보장)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 3. Model Configuration
    # Force single-GPU placement (same rationale as sft_train.py)
    load_kwargs = {
        "torch_dtype": config.TORCH_DTYPE,
        "device_map": {"": 0} if config.DEVICE == "cuda" else "auto",
    }
    if config.USE_QUANTIZATION:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.TORCH_DTYPE,
            llm_int8_enable_fp32_cpu_offload=False,
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
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_pin_memory=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=5,
        warmup_ratio=0.1,  # DPO 권장: 학습 초기 안정화
        bf16=(config.DEVICE == "cuda"),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        # Load best model (by eval_loss) to avoid overfit last-epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",  # QLoRA 표준: optimizer state 메모리 75% 절감
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT 호환
        max_grad_norm=1.0,  # Gradient clipping (DPO 안정성)
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
    import os
    resume = os.path.isdir(args.output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(args.output_dir)
    )
    trainer.train(resume_from_checkpoint=resume)

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
        help="DPO loss variant (default from config.DPO_LOSS_TYPE — currently 'sigmoid').",
    )
    args = parser.parse_args()
    train_dpo(args)
