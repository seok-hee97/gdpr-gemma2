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
from . import config
from .data_loader import get_gdpr_dataset

def train():
    # 재현성을 위한 시드 고정
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_NAME,
        cache_dir=config.CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. Load Dataset
    dataset = get_gdpr_dataset(tokenizer)

    # 3. BitsAndBytes Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 4. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.CACHE_DIR # 로컬 폴더에 저장
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.TARGET_MODULES
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_checkpointing=True,
        remove_unused_columns=False,
        learning_rate=config.LEARNING_RATE,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type=config.LR_SCHEDULER,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        save_strategy="epoch",
        output_dir=config.OUTPUT_DIR,
        optim=config.OPTIMIZER,
        warmup_steps=config.WARMUP_STEPS,
        bf16=True,
        report_to="none",
    )

    # 7. DPO Trainer
    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=config.DPO_BETA,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_length=config.MAX_LENGTH,
    )

    # 8. Start Training
    print(f"Starting DPO Fine-tuning with {len(dataset)} samples...")
    trainer.train()

    # 9. Save the Fine-tuned Adapter
    print(f"Saving model to {config.OUTPUT_DIR}...")
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("Training Complete.")

if __name__ == "__main__":
    train()
