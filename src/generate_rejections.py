import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
from . import config

def generate_rejections(args):
    config.ensure_base_model()

    # 1. Load SFT Model (Base + Adapter)
    # Same device_map fix as sft_train.py for DGX GH200 (unified memory)
    device_map = {"": 0} if config.DEVICE == "cuda" else "auto"
    print(f"Loading SFT model from {args.sft_model_path} for rejection generation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=config.TORCH_DTYPE,
        device_map=device_map,
        attn_implementation="eager"  # Gemma2 권장
    )
    model = PeftModel.from_pretrained(base_model, args.sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 필수: Batch inference를 위해 왼쪽 패딩

    # 2. Pipeline setup — don't re-specify device_map (model already placed)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # 3. Load Original Dataset
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=config.DATASET_CACHE_DIR)
    
    # Generator for memory efficiency
    def data_generator():
        for example in dataset:
            instruction = example['instruction']
            input_text = example['input']
            prompt_content = f"{instruction}\n\n{input_text}"
            messages = [{"role": "user", "content": prompt_content}]
            yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Generate Responses (Batch Inference with Progress Bar)
    print(f"Generating rejections from SFT model (Batch mode, size={args.batch_size})...")
    
    dynamic_data = []
    
    # pipe에 generator를 넘기면 배치를 유지하며 결과를 하나씩 내뱉습니다.
    # return_full_text=False를 사용해 프롬프트를 제외한 답변만 추출합니다.
    for i, output in enumerate(tqdm(pipe(data_generator(), batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.9, return_full_text=False), total=len(dataset))):
        rejected_response = output[0]['generated_text'].strip()
        
        dynamic_data.append({
            "instruction": dataset[i]['instruction'],
            "input": dataset[i]['input'],
            "output": dataset[i]['output'], # 전문가 정답 (Chosen)
            "rejected": rejected_response  # SFT 모델의 답변 (Rejected)
        })

    # 5. Save to JSONL
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for entry in dynamic_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dynamic DPO dataset saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Generate Dynamic Rejections for DPO")
    parser.add_argument("--base_model", type=str, default=config.BASE_MODEL_PATH)
    parser.add_argument("--sft_model_path", type=str, default=config.SFT_MODEL_PATH)
    parser.add_argument("--dataset_name", type=str, default="sims2k/GDPR_QA_instruct_dataset")
    parser.add_argument("--split", type=str, default="train[:]")
    parser.add_argument("--output_path", type=str, default=config.DYNAMIC_DATASET_PATH)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens per rejection. 256 creates length bias (rejected ~45%% of chosen). "
             "512 is closer to chosen average (~2,226 chars ≈ 550 tokens).",
    )
    
    args = parser.parse_args()
    generate_rejections(args)
