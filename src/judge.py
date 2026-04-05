import os
import json
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from . import config

def evaluate_with_llm(args):
    """
    GPT-4를 판사로 사용하여 모델 답변의 법적 정확성과 품질을 평가합니다.
    """
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # 평가 결과 입력 경로
    input_path = os.path.join(config.EVAL_RESULTS_DIR, args.input_csv)
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run src.eval first.")
        return

    judge_results = []

    print(f"Starting LLM-as-a-judge using {args.judge_model}...")
    
    # 샘플 개수 제한
    test_samples = df.head(args.num_samples)

    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
        prompt = f"""
        Evaluate the AI model's response for compliance with the General Data Protection Regulation (GDPR) based on the provided Reference.

        [Instruction]
        Compare the [Prediction] with the [Reference] based on the [Question].
        Assess the accuracy, alignment with GDPR principles, and the overall quality of the prediction.

        [Question]
        {row['prompt']}

        [Reference (Expert Answer)]
        {row['reference']}

        [Prediction (AI Model)]
        {row['prediction']}

        Please provide your evaluation on a scale of 1 to 5 for the following criteria:
        1. Legal Correctness: Does the prediction align with specific GDPR articles and legal facts?
        2. Compliance Alignment: Does it correctly reflect core GDPR principles (e.g., Lawfulness, Transparency, Data Minimization)?
        3. Clarity & Professionalism: Is the response clear, structured, and maintaining a professional tone?

        You MUST respond ONLY in the following JSON format:
        {{
            "scores": {{
                "correctness": <int: 1-5>,
                "compliance": <int: 1-5>,
                "clarity": <int: 1-5>
            }},
            "reasoning": "Detailed explanation of your scores in English."
        }}
        """

        try:
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": "You are a senior legal expert specializing in GDPR and an experienced AI evaluation specialist. Your goal is to provide rigorous, accurate, and objective scores for AI-generated legal advice."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            judge_results.append({
                "id": row.get('id', 0),
                "scores": result['scores'],
                "reasoning": result['reasoning']
            })
        except Exception as e:
            print(f"API Error at sample {row.get('id')}: {e}")

    # 결과 저장 경로
    output_path = os.path.join(config.EVAL_RESULTS_DIR, args.output_csv)
    result_df = pd.DataFrame(judge_results)
    result_df.to_csv(output_path, index=False)
    
    # 평균 점수 계산 및 출력
    if judge_results:
        avg_correctness = result_df['scores'].apply(lambda x: x['correctness']).mean()
        avg_compliance = result_df['scores'].apply(lambda x: x['compliance']).mean()
        avg_clarity = result_df['scores'].apply(lambda x: x['clarity']).mean()
        
        print("\n--- LLM Judge Results (Averages) ---")
        print(f"Legal Correctness: {avg_correctness:.2f}/5.0")
        print(f"Compliance Alignment: {avg_compliance:.2f}/5.0")
        print(f"Clarity: {avg_clarity:.2f}/5.0")
        print(f"Detailed logs saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for GDPR Model Evaluation")
    parser.add_argument("--input_csv", type=str, default="evaluation_results_comprehensive.csv")
    parser.add_argument("--output_csv", type=str, default="llm_judge_results.csv")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--judge_model", type=str, default=config.JUDGE_MODEL)
    
    args = parser.parse_args()
    evaluate_with_llm(args)
