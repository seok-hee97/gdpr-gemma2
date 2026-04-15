# GDPR-Gemma-2-2B
## GDPR Compliance Assistant: AI-Powered Data Protection Guidance

🚀 **Project Highlights:**
- Specialized AI model for GDPR compliance guidance
- Fine-tuned Google's Gemma 2B using a 3-Stage Training Pipeline (SFT -> Dynamic Rejection -> DPO)
- Implemented QLoRA for efficient, resource-friendly training
- Designed to provide accurate, relevant responses to GDPR-related inquiries
- Qualitative evaluation using LLM-as-a-judge (GPT-4o)

## Project Description
This project develops an advanced AI model specialized in providing guidance on GDPR (General Data Protection Regulation) compliance.     
By fine-tuning Google's Gemma 2B model using Direct Preference Optimization (DPO) and a GDPR-specific dataset,     
we've created a powerful tool to assist organizations with data protection queries and regulatory compliance.

🔗 **Hugging Face Model:** [cycloevan/gdpr_gemma-2-2b](https://huggingface.co/cycloevan/gdpr_gemma-2-2b)
📚 **GitHub Repository:** [seok-hee97/gdpr-gemma2](https://github.com/seok-hee97/gdpr-gemma2)

## Key Features

- **GDPR Expertise:** Specialized in GDPR compliance and data protection regulations.
- **DPO Alignment:** Utilizes Direct Preference Optimization (DPO) with Dynamic Rejection for precise alignment with GDPR principles.
- **Resource Efficient:** Implements 4-bit quantization using QLoRA for efficient training on standard hardware.
- **Comprehensive Evaluation:** Combines ROUGE/BLEU scores with qualitative assessment via GPT-4o.

## Technical Specifications

- **Base Model:** `google/gemma-2-2b-it`
- **Fine-tuning Method:** 3-Stage Pipeline (SFT -> Dynamic Rejection -> DPO)
- **Training Dataset:** [sims2k/GDPR_QA_instruct_dataset](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset)
- **Quantization:** 4-bit (QLoRA)
- **Judge Model:** `gpt-4o` (OpenAI API) for qualitative evaluation.

----------------------------------------------------------------------------------------------------------------------------

## Project Structure

```text
/gdpr-gemma2
├── src/                    # Source Modules
│   ├── config.py           # Hyperparameters & Local Paths
│   ├── data_loader.py      # Multi-stage data processing
│   ├── sft_train.py        # [Stage 1] Knowledge injection
│   ├── generate_rejections.py # [Stage 2] Dynamic data prep (Dynamic Rejection)
│   ├── dpo_train.py        # [Stage 3] Preference alignment
│   ├── inference.py        # Hybrid inference engine
│   ├── eval.py             # ROUGE/BLEU/BertScore Evaluation
│   ├── judge.py            # LLM-as-a-judge (GPT-4o) Assessment
│   ├── filter_rejections.py # DPO pair quality filter (length/citation bias)
│   ├── diagnose_prompt.py  # Prompt-format A/B diagnostic
│   └── push_to_hub.py      # Merge LoRA adapter & push to HF Hub
├── data/                   # Dataset storage (.cache included)
├── models/                 # Model artifacts (Base/SFT/DPO)
├── eval/                   # Evaluation results and LLM-judge reports
├── app.py                  # Streamlit Web Interface
├── Dockerfile              # Containerized Deployment
└── requirements.txt        # Python dependencies
```

## Getting Started

### 1. Environment Setup
```bash
conda create -n gdpr-env python=3.11 -y
conda activate gdpr-env
pip install -r requirements.txt
```

### 2. 3-Stage Training Pipeline
To achieve industry-standard performance, follow these steps:

1. **Stage 1 (SFT):** Teach the model GDPR facts.
   ```bash
   python -m src.sft_train
   ```
2. **Stage 2 (Data Prep):** Generate real-world rejections from the SFT model (Dynamic Rejection).
   ```bash
   python -m src.generate_rejections
   ```
3. **Stage 3 (DPO):** Align the model to prefer expert answers over SFT errors.
   ```bash
   python -m src.dpo_train
   ```

### 3. Evaluation & Inference
- **Benchmark:** `python -m src.eval`
- **Qualitative Judge:** `python -m src.judge`
- **Web Assistant:** `streamlit run app.py`

## Limitations & Future Improvements

> Honest assessment of what this model does *not* do well, and the concrete next steps that would address each gap. All limitations are grounded in the evaluation results above.

### Current Limitations
- **Base model at ceiling** — `gemma-2-2b-it` already handles GDPR questions at a level that 316 Q&A samples cannot meaningfully surpass. All fine-tuned variants (SFT, DPO v1, DPO v2) match or underperform the base model on qualitative metrics at n=50.
- **Dynamic Rejection signal quality** — Data inspection revealed that auto-generated rejections have systematic length asymmetry (rejected ~56% of chosen length) and citation density gap (1.41 vs 3.75 articles/sample), with only 9% Jaccard overlap on cited articles. In ~2/3 of inspected pairs, the "rejected" answer was not clearly worse than "chosen." DPO learns spurious "longer + more citations" signals rather than genuine accuracy.
- **Article citation accuracy ~2.5-2.6/5** — The model occasionally hallucinates GDPR article numbers or misapplies references. This is a retrieval problem — fine-tuning cannot reliably encode 100+ GDPR articles into a 2B model from 316 samples.
- **English only** — Trained on `sims2k/GDPR_QA_instruct_dataset` (English-only). Although `gemma-2-2b-it` is multilingual, this fine-tune is not aligned for non-English GDPR queries.
- **Static knowledge snapshot** — Reflects the regulation text only; does not incorporate post-training EDPB guidelines, CJEU rulings, or national supervisory authority decisions.

### What We Tried (Experiments Log)
| Experiment | Hypothesis | Result |
|---|---|---|
| n=10 → n=50 re-evaluation | Initial n=10 results showed DPO outperforming Base; suspected noise | Confirmed: n=10 was a false positive. Base ≥ all variants at n=50 |
| Prompt mismatch diagnosis | inference.py adds system prompt absent from training → harms fine-tuned models | Rejected: A/B test showed no meaningful difference between formats |
| Tier 1 fix (filter + IPO + β=0.3) | Remove length/citation bias from DPO pairs, use robust loss | DPO v2 regressed further — data reduction (316→103) outweighed noise removal |

### Future Directions
- **RAG (Retrieval-Augmented Generation)** — Index GDPR article text in a vector DB and retrieve at inference time. This directly addresses article hallucination, which fine-tuning alone cannot solve. Most promising next step.
- **Targeted negative generation** — Use GPT-4o to create controlled rejections with intentionally wrong article citations, producing a cleaner preference signal than self-generated negatives.
- **Dataset expansion** — Augment beyond 316 samples using GPT-4o synthesis or additional legal Q&A datasets to provide more headroom for fine-tuning.

## Evaluation Results

Benchmarked on DGX Spark — quantitative metrics on 100 samples, qualitative LLM-as-a-Judge (GPT-4o) on **50 samples** (re-evaluated from initial n=10 to ensure statistical reliability).

### Quantitative (ROUGE / BLEU / BertScore)
| Metric       | Base   | SFT        | DPO v1 | DPO v2 (Tier 1) |
|--------------|--------|------------|--------|------------------|
| ROUGE-L      | 0.2072 | **0.2331** | 0.2252 | 0.2165           |
| BLEU         | 0.0838 | **0.1146** | 0.1034 | 0.1045           |
| BertScore F1 | 0.8432 | **0.8541** | 0.8527 | 0.8486           |

### Qualitative (LLM-as-a-Judge, GPT-4o, n=50)
| Criterion            | Base     | SFT      | DPO v1 | DPO v2 (Tier 1) |
|----------------------|----------|----------|--------|------------------|
| Legal Correctness    | **3.18** | **3.18** | 3.06   | 2.86             |
| Article Accuracy     | **2.64** | 2.52     | 2.50   | 2.50             |
| Compliance Alignment | **3.62** | **3.62** | 3.40   | 3.18             |
| Clarity              | 4.10     | **4.12** | 3.74   | 3.32             |

### Interpretation

The base `gemma-2-2b-it` model achieves the highest (or tied-highest) scores across all qualitative criteria. Neither SFT nor DPO produces statistically significant improvements at n=50.

- **SFT** improves surface-level text overlap (ROUGE/BLEU) because it directly maximizes reference-text likelihood, but adds no measurable gain on legal quality metrics.
- **DPO v1** (standard sigmoid, β=0.1) slightly degrades all judge scores vs Base.
- **DPO v2** (Tier 1 fix: IPO loss, β=0.3, filtered data 316→103 pairs) regresses further — the 66% data reduction outweighed the noise-reduction benefit.

This pattern is consistent with the known failure conditions of self-generated preferences (SPIN; Chen et al., 2024) on small datasets with a strong instruction-tuned base model. See *Limitations* below for the full analysis.

---

## References

### Model & Official Docs
- [Gemma — Official site](https://ai.google.dev/gemma/)
- [Gemma 2 model card](https://ai.google.dev/gemma/docs/model_card_2#model_information)
- [Gemma 2 announcement](https://blog.google/technology/developers/google-gemma-2/)
- [Gemma docs](https://ai.google.dev/gemma/docs)
- [Gemma Cookbook (GitHub)](https://github.com/google-gemini/gemma-cookbook)
- [Aligning DPO Gemma 2B-it (Cookbook notebook)](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Aligning_DPO_Gemma_2b_it.ipynb)

### Datasets
- [sims2k/GDPR_QA_instruct_dataset (HF)](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_dataset) — primary training set
- [sims2k/GDPR_QA_instruct_eval_dataset (HF)](https://huggingface.co/datasets/sims2k/GDPR_QA_instruct_eval_dataset) — evaluation split
- [tamjidrahat/gdpr-dataset (GitHub)](https://github.com/tamjidrahat/gdpr-dataset)
- [Is Your Policy Compliant? — ACM paper (2022)](https://dl.acm.org/doi/10.1145/3559613.3563195)

### Fine-tuning Techniques (QLoRA · LoRA · DPO)
- [Fine-tuning Gemma with QLoRA — Google Developer Experts](https://medium.com/google-developer-experts/fine-tuning-gemma-with-qlora-407e56c36026)
- [Fine-Tune Gemma Using QLoRA — Samvardhan](https://medium.com/@samvardhan777/fine-tune-gemma-using-qlora-%EF%B8%8F-6b2f2e76dc55)
- [Low-Rank Adapter (LoRA) Explained](https://medium.com/@shelikohan/low-rank-adapter-lora-explained-0d3677395639)
- [Fine-tuning LLMs w/ Example Code (YouTube)](https://www.youtube.com/watch?v=eC6Hd1hFvos)
- [Getting Started with Gemma using HuggingFace Libraries](https://medium.com/@coldstart_coder/getting-started-with-googles-gemma-llm-using-huggingface-libraries-a0d826c552ae)
- [Sherlock Holmes Q&A with Gemma fine-tuning (Kaggle)](https://www.kaggle.com/code/lucamassaron/sherlock-holmes-q-a-with-gemma-fine-tuning/notebook)

### Related Projects
- [sims2k/Saul-Instruct-v1-gdpr-finetuned-v5.2 (HF)](https://huggingface.co/sims2k/Saul-Instruct-v1-gdpr-finetuned-v5.2)
- [UKPLab/sentence-transformers (GitHub)](https://github.com/UKPLab/sentence-transformers)

### Domain Background (GDPR & DLP)
- [Data Loss Prevention — an EU/GDPR perspective](https://grcoutlook.com/data-loss-prevention-an-eu-gdpr-perspective/)
- [How Cato uses LLMs to improve Data Loss Prevention](https://www.catonetworks.com/blog/how-cato-uses-large-language-models-to-improve-data-loss-prevention/)

### Korean-language Resources
- [구글의 최첨단 오픈 모델 '젬마(Gemma)' 공개 — Google Korea Blog](https://blog.google/intl/ko-kr/products/explore-get-answers/-gemma-open-models-kr/)
- [Encoder Only 와 Decoder Only 언어모델에 대한 고찰](https://medium.com/@hugmanskj/encoder-only-%EC%99%80-decoder-only-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B3%A0%EC%B0%B0-9852213dbb72)
- [Gemma 한국어 요약 모델 파인튜닝 — DEVOCEAN](https://devocean.sk.com/blog/techBoardDetail.do?ID=165703&boardType=techBlog&ref=blog.update.sh)
- [오픈소스 LLM에 새로운 표준 — DEVOCEAN](https://devocean.sk.com/blog/techBoardDetail.do?ID=165709)
- [Gemma 논문 리뷰 — wiz-tech](https://wiz-tech.tistory.com/entry/Gemma-Open-Models-Based-on-GeminiResearch-and-Technology-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)

### Community
- [Build with Google AI forum](https://discuss.ai.google.dev/)
- [Gemma developer blog posts](https://developers.googleblog.com/en/search/?query=gemma&product_categories=Gemma)