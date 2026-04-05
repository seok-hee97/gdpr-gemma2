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
        device_map="auto",
        cache_dir=config.CACHE_DIR
    )
    model = PeftModel.from_pretrained(base_model, config.SFT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_PATH)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # 2. Load Original Dataset
    dataset = load_dataset("sims2k/GDPR_QA_instruct_dataset", split='train[:]', cache_dir=config.CACHE_DIR)
    
    dynamic_data = []

    # 3. Generate Responses (Rejected candidates)
    print("Generating rejections from SFT model...")
    for example in tqdm(dataset):
        instruction = example['instruction']
        input_text = example['input']
        prompt_content = f"{instruction}\n\n{input_text}"
        
        messages = [{"role": "user", "content": prompt_content}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # SFT 모델의 답변 생성
        output = pipe(formatted_prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
        rejected_response = output.split("<start_of_turn>model\n")[-1].replace("<eos>", "").strip()
        
        dynamic_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": example['output'], # 전문가 정답 (Chosen)
            "rejected": rejected_response  # SFT 모델의 답변 (Rejected)
        })

    # 4. Save to JSONL
    with open(config.DYNAMIC_DATASET_PATH, 'w', encoding='utf-8') as f:
        for entry in dynamic_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dynamic DPO dataset saved to {config.DYNAMIC_DATASET_PATH}")

if __name__ == "__main__":
    generate_rejections()
