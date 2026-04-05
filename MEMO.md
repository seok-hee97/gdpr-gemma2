
### 개선사항
- Ipynb -> .py  ml 프로젝트 구조로 변환.
- 추가로 모델 테스트할수 있는 웹사이트 배포 코드 (Streamlit)
- gdpr 테스트 진행 및 평가지표 테스트 및 추가.
- requirements.txt, .env.example
- +@



# linkedin-post


Excited to share the release of GDPR-Gemma-2-2B: An AI-Powered GDPR Compliance Assistant!

🚀 Project Highlights:
• Specialized AI model for GDPR compliance guidance
• Fine-tuned Google's Gemma 2B using Direct Preference Optimization (DPO)
• Implemented QLoRA for efficient, resource-friendly training
• Designed to provide accurate, relevant responses to GDPR-related inquiries

🔍 Key Features:
• In-depth knowledge of GDPR and data protection regulations
• Precise alignment with GDPR principles through DPO
• 4-bit quantization for improved efficiency

This project aims to empower organizations navigating the complexities of data protection regulations. By leveraging advanced AI techniques, we've created a tool that can assist with GDPR compliance queries and offer valuable insights into regulatory requirements.

🔗 Check out the model: [cycloevan/gdpr_gemma-2-2b](https://huggingface.co/cycloevan/gdpr_gemma-2-2b)
📚 GitHub Repository: [gitHub code](https://github.com/seok-hee97/gdpr-gemma2)

🔗 Check out the model: https://huggingface.co/cycloevan/gdpr_gemma-2-2b
📚 GitHub Repository: https://github.com/seok-hee97/gdpr-gemma2

#GoogleMLBootcamp #GemmaSprint #AI #GDPR #DataProtection #MachineLearning #Compliance


## 포트폴리오

### **Google ML Bootcamp**
> 1인 (Google ML Bootcamp 5th) | 2024.07 - 2024.10
- (Coursera) Deep Learning Specialization 교육 과정 수료 및 스터디 진행.
- (Kaggle Competition) Binary Prediction of Poisonous Mushrooms 상위 5% 달성.    
  MCC Score : 0.98502 (Rank :76/2,422, top 3.1% | Feature engineering과 XGBoost, LightGBM 등 활용)         
- (Gemma Sprint Project) GDPR compliance Q&A assistant 개발.    
  Gemma-2-2B model을 Direct Preference Optimization (DPO) 방식으로 fine-tuning.   
- Skills : PyTorch, TensorFlow, Transformers, XGBoost, LightGBM, ML
- Link : [gdpr-gemma model](https://huggingface.co/cycloevan/gdpr_gemma-2-2b)



### 현재 실행환경.

이렇게 구성되는데

dataset : 
https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset

columns
- instruction
- input
- output
- text
- disscussion_text

지금 수정된 DPO 학습 방법이 바람직한건지 다시한번 검토해줘

@notebooks/GDPR_Gemma_2b.ipynb


## 개선사항.
데이터셋이나 모델 로드 및 다운로드할때
프로젝트 폴더에 로드 및 다운로드 진행.

학습 및 실행환경.
개인 노트북 : m1 macbook air
추가사용가능 : dgx spark 서버


추가로 지금 데이터셋에 dataset_loader와
train.py 방법이 DPO 방식으로 학습하는게 맞는지 검토해줘






추가 변경사항 고려:
data/ , models/ 폴더 생성
데이터셋 로드나 생성은 로컬 프로젝트에 생성.\




아니굳이 .cache 파일을 사용해야해                                                                                                                                 
models/ 경로에 베이스 모델 저장하고 학습된 모델 만들어서 저장하고

data/ 경로에 사용하는 데이터 저장하고 생성된 데이터도 저장하는 걸로 하면 되지