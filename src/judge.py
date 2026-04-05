import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from . import config

def evaluate_with_llm(csv_name="evaluation_results_comprehensive.csv"):
    """
    GPT-4를 판사로 사용하여 모델 답변의 법적 정확성과 품질을 평가합니다.
    """
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # 평가 결과 입력 경로
    input_path = os.path.join(config.EVAL_RESULTS_DIR, csv_name)
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run src.eval first.")
        return

    judge_results = []

    print(f"Starting LLM-as-a-judge using {config.JUDGE_MODEL}...")
    
    # 샘플 10개만 우선 테스트 (비용 고려)
    test_samples = df.head(10)

    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
        prompt = f"""
        당신은 GDPR(유럽 일반 데이터 보호 규칙) 법률 전문가이자 AI 모델 평가관입니다.
        아래의 질문에 대해 AI 모델이 내놓은 답변을 전문가의 정답(Reference)과 비교하여 평가해 주세요.

        [질문]
        {row['prompt']}

        [전문가 정답 (Reference)]
        {row['reference']}

        [AI 모델 답변 (Prediction)]
        {row['prediction']}

        다음 3가지 항목에 대해 1점(매우 나쁨)에서 5점(매우 좋음) 사이의 점수를 부여하고, 그 이유를 짧게 설명해 주세요.
        1. 법적 정확성 (Legal Correctness): 모델 답변이 GDPR 조문 및 법적 사실과 일치하는가?
        2. 준수성 (Compliance Alignment): GDPR의 핵심 원칙을 잘 반영하고 있는가?
        3. 가독성 및 완성도 (Clarity): 답변이 명확하고 전문가다운 어조를 유지하는가?

        반드시 아래 JSON 형식으로만 응답해 주세요:
        {{
            "scores": {{
                "correctness": 0,
                "compliance": 0,
                "clarity": 0
            }},
            "reasoning": "점수 부여 이유 요약"
        }}
        """

        try:
            response = client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "system", "content": "You are a GDPR legal expert."},
                          {"role": "user", "content": prompt}],
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
    output_path = os.path.join(config.EVAL_RESULTS_DIR, "llm_judge_results.csv")
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
    evaluate_with_llm()
