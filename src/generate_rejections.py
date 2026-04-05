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
    tokenizer.padding_side = "left" # 필수: Batch inference를 위해 왼쪽 패딩
    
    # 2. Pipeline setup
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto"
    )

    # 3. Load Original Dataset
    dataset = load_dataset("sims2k/GDPR_QA_instruct_dataset", split='train[:]')
    
    # Generator for memory efficiency
    def data_generator():
        for example in dataset:
            instruction = example['instruction']
            input_text = example['input']
            prompt_content = f"{instruction}\n\n{input_text}"
            messages = [{"role": "user", "content": prompt_content}]
            yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Generate Responses (Batch Inference with Progress Bar)
    print("Generating rejections from SFT model (Batch mode)...")
    batch_size = 4  # VRAM 용량에 따라 4~16 정도로 조절 가능
    
    dynamic_data = []
    
    # pipe에 generator를 넘기면 배치를 유지하며 결과를 하나씩 내뱉습니다.
    # return_full_text=False를 사용해 프롬프트를 제외한 답변만 추출합니다.
    for i, output in enumerate(tqdm(pipe(data_generator(), batch_size=batch_size, max_new_tokens=256, do_sample=True, temperature=0.7, return_full_text=False), total=len(dataset))):
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
