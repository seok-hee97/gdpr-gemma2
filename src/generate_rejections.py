import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
from . import config

def generate_rejections():
    # 1. Load SFT Model (Base + Adapter)
    print("Loading SFT model for rejection generation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, config.SFT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Batch inference를 위해 왼쪽 패딩 권장
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto"
    )

    # 2. Load Original Dataset
    dataset = load_dataset("sims2k/GDPR_QA_instruct_dataset", split='train[:]')
    
    # 3. Prepare Prompts
    print("Preparing prompts...")
    prompts = []
    for example in dataset:
        instruction = example['instruction']
        input_text = example['input']
        prompt_content = f"{instruction}\n\n{input_text}"
        messages = [{"role": "user", "content": prompt_content}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    # 4. Generate Responses (Batch Inference)
    print("Generating rejections from SFT model (Batch mode)...")
    batch_size = 4  # VRAM 용량에 따라 4~8 정도로 조절 가능
    results = []
    
    # pipe에 리스트를 직접 전달하고 batch_size를 설정하면 내부적으로 최적화됩니다.
    outputs = pipe(
        prompts, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7, 
        batch_size=batch_size,
        return_full_text=False # 결과에서 프롬프트 제외하고 답변만 받기
    )

    dynamic_data = []
    for i, output in enumerate(outputs):
        rejected_response = output[0]['generated_text'].strip()
        
        dynamic_data.append({
            "instruction": dataset[i]['instruction'],
            "input": dataset[i]['input'],
            "output": dataset[i]['output'], # 전문가 정답 (Chosen)
            "rejected": rejected_response  # SFT 모델의 답변 (Rejected)
        })

    # 5. Save to JSONL
    with open(config.DYNAMIC_DATASET_PATH, 'w', encoding='utf-8') as f:
        for entry in dynamic_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dynamic DPO dataset saved to {config.DYNAMIC_DATASET_PATH}")

if __name__ == "__main__":
    generate_rejections()
