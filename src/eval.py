import torch
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import pandas as pd
from .inference import GDPRInference
from . import config

def run_evaluation(model_path=None, split="train[:100]"): # 샘플 수를 100개로 확대
    """
    정량적 성능 평가 수행 (ROUGE, BLEU)
    """
    print(f"--- Starting Evaluation for {model_path or config.NEW_MODEL_NAME} ---")
    
    # 1. 데이터셋 로드
    try:
        # 학습에 쓰이지 않은 전문 평가 데이터셋 사용 권장
        dataset_name = "sims2k/GDPR_QA_instruct_dataset"
        test_ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Dataset load failed: {e}")
        return

    # 2. 모델 로드
    infer = GDPRInference(model_path=model_path)
    
    # 3. 지표 로더
    rouge = load("rouge")
    bleu = load("bleu")
    
    predictions = []
    references = []
    results_log = []

    # 4. 추론 루프
    for i, example in enumerate(tqdm(test_ds)):
        prompt = f"{example['instruction']}\n\n{example['input']}"
        reference = example['output']
        
        # 모델 답변 생성 (충분한 길이 확보)
        prediction = infer.generate(prompt, max_new_tokens=512)
        
        predictions.append(prediction)
        references.append(reference)
        
        results_log.append({
            "id": i,
            "prompt": prompt[:100],
            "reference": reference[:100],
            "prediction": prediction[:100]
        })

    # 5. 점수 계산
    print("Calculating scores...")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    print("\n--- Evaluation Results ---")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    print(f"BLEU: {bleu_results['bleu']:.4f}")
    
    # 상세 로그 저장
    df = pd.DataFrame(results_log)
    output_path = os.path.join(config.EVAL_RESULTS_DIR, "evaluation_results_comprehensive.csv")
    df.to_csv(output_path, index=False)
    print(f"Detailed logs saved to {output_path}")
    
    return rouge_results, bleu_results

if __name__ == "__main__":
    run_evaluation()
