# GDPR-Gemma-2-2B
## GDPR Compliance Assistant: AI-Powered Data Protection Guidance

## Project Description
This project develops an advanced AI model specialized in providing guidance on GDPR (General Data Protection Regulation) compliance.     
By fine-tuning Google's Gemma 2B model using Direct Preference Optimization (DPO) and a GDPR-specific dataset,     
we've created a powerful tool to assist organizations with data protection queries and regulatory compliance.     


## Key Features

- Specialized in GDPR compliance and data protection regulations
- Utilizes DPO for precise alignment with GDPR principles
- Implements QLoRA for efficient and resource-friendly training
- Designed to provide accurate and relevant responses to GDPR-related inquiries

## Model Details

- Base Model: Google Gemma 2B
- Fine-tuning Method: Direct Preference Optimization (DPO)
- Training Dataset: [sims2k/GDPR_QA_instruct_dataset](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset)
- Quantization: 4-bit quantization using QLoRA


----------------------------------------------------------------------------------------------------------------------------

## Project Structure

```text
/gdpr-gemma2
├── src/                    # Source Modules
│   ├── config.py           # Hyperparameters & Local Paths
│   ├── data_loader.py      # Multi-stage data processing
│   ├── sft_train.py        # [Stage 1] Knowledge injection
│   ├── generate_rejections.py # [Stage 2] Dynamic data prep
│   ├── dpo_train.py        # [Stage 3] Preference alignment
│   ├── inference.py        # Hybrid inference engine
│   └── eval.py             # ROUGE/BLEU Evaluation
├── data/                   # Dataset storage (.cache included)
├── models/                 # Model artifacts (SFT/DPO)
├── app.py                  # Streamlit Web Interface
├── Dockerfile              # Containerized Deployment
└── requirements.txt        # Python dependencies
```

## Getting Started (for DGX Spark / Server)

### 1. Environment Setup
```bash
conda create -n gdpr-env python=3.10 -y
conda activate gdpr-env
pip install -r requirements.txt
```

### 2. 2-Stage Training Pipeline
To achieve industry-standard performance, follow these steps:

1. **Stage 1 (SFT):** Teach the model GDPR facts.
   ```bash
   python -m src.sft_train
   ```
2. **Stage 2 (Data Prep):** Generate real-world rejections from the SFT model.
   ```bash
   python -m src.generate_rejections
   ```
3. **Stage 3 (DPO):** Align the model to prefer expert answers over SFT errors.
   ```bash
   python -m src.dpo_train
   ```

### 3. Evaluation & Inference
- **Benchmark:** `python -m src.eval`
- **Web Assistant:** `streamlit run app.py`

## Evaluation Results
Current evaluation performed on M1 Mac (20 samples):
- **ROUGE-L:** 0.2094
- **BLEU:** 0.1129

---

## Reference

#### project-reference
- QLoRa Fine-tuning
  - [Fine-tuning Gemma with QLoRa](https://medium.com/google-developer-experts/fine-tuning-gemma-with-qlora-407e56c36026)
  - [Fine-Tune Gemma Using QLoRA](https://medium.com/@samvardhan777/fine-tune-gemma-using-qlora-%EF%B8%8F-6b2f2e76dc55)
- Code Reference
  - [(github)UKPLab/sentance-transformers](https://github.com/UKPLab/sentence-transformers)
  - [(cookbook)Aligning_DPO_Gemma_2b_it.ipynb](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Aligning_DPO_Gemma_2b_it.ipynb)
  - 
- Related LLM project
  - [(hugging face)sims2k/Saul-Instruct-v1-gdpr-finetuned-v5.2](https://huggingface.co/sims2k/Saul-Instruct-v1-gdpr-finetuned-v5.2)

- Concept
  - [Data Loss Prevention, an EU/GDPR perspective](https://grcoutlook.com/data-loss-prevention-an-eu-gdpr-perspective/)
  - [How to Cato Uses Large Language Models to Improve Data Loss Prvention](https://www.catonetworks.com/blog/how-cato-uses-large-language-models-to-improve-data-loss-prevention/)


#### Dataset
- [sims2k/GDPR_QA_instruct_dataset](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset)
- [sims2k/GDPR_QA_instruct_eval_dataset](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_eval_dataset)
- [(github)gdpr-dataset](https://github.com/tamjidrahat/gdpr-dataset)
- [Is Your Policy Compliant?: A Deep Learning-based Empirical Study of Privacy Policies' Compliance with GDPR](https://dl.acm.org/doi/10.1145/3559613.3563195)


#### Gemma-docs
- [Gemma](https://ai.google.dev/gemma/) 
- [Announcement](https://blog.google/technology/developers/google-gemma-2/)
- [Docs](https://ai.google.dev/gemma/docs)
- [Blog posts](https://developers.googleblog.com/en/search/?query=gemma&product_categories=Gemma) 
- [Cookbook](https://github.com/google-gemini/gemma-cookbook)


#### Gemma Model
- [(구글코라이 블로그)구글의 최첨단 오픈 모델 '젬마(Gemma)'를 공개합니다](https://blog.google/intl/ko-kr/products/explore-get-answers/-gemma-open-models-kr/)
- [Gemma 2 model card](https://ai.google.dev/gemma/docs/model_card_2#model_information)
- [Encoder Only 와 Decoder Only 언어모델에 대한 고찰](https://medium.com/@hugmanskj/encoder-only-%EC%99%80-decoder-only-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B3%A0%EC%B0%B0-9852213dbb72)


#### reference
- [(youtube)Fine-tuning LLMs | w/ Example Code](https://www.youtube.com/watch?v=eC6Hd1hFvos)
- [Low-Rank Adapter (LoRA) Explained](https://medium.com/@shelikohan/low-rank-adapter-lora-explained-0d3677395639)
- [(medium)Getting Started with Google's Gemma LLM using HuggingFace Libaries](https://medium.com/@coldstart_coder/getting-started-with-googles-gemma-llm-using-huggingface-libraries-a0d826c552ae)
- [(DEVOCEAN) Gemma 한국어 요약 모델 파인튜닝 빠르게 해보기](https://devocean.sk.com/blog/techBoardDetail.do?ID=165703&boardType=techBlog&ref=blog.update.sh)
- [(DEVOCEAN)오픈소스 LLM에 새로운 표준을 제시할 구글 Gemma](https://devocean.sk.com/blog/techBoardDetail.do?ID=165709)
- [(blog)Gemma: Open Models Based on GeminiResearch and Technology 논문 리뷰](https://wiz-tech.tistory.com/entry/Gemma-Open-Models-Based-on-GeminiResearch-and-Technology-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)
- [Sherlock Holmes Q&A with Gemma fine tuning](https://www.kaggle.com/code/lucamassaron/sherlock-holmes-q-a-with-gemma-fine-tuning/notebook)

#### Community
- [Build with Google AI forum](https://discuss.ai.google.dev/)
- [Discord 채널](https://discord.com/channels/1009525727504384150/1209857547390025768)