import os
import torch
from dotenv import load_dotenv

load_dotenv()

# --- Hardware Auto-Detection ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.bfloat16
    USE_QUANTIZATION = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16
    USE_QUANTIZATION = False   # bitsandbytes는 CUDA 전용
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    USE_QUANTIZATION = False

# --- Model & Tokenizer ---
BASE_MODEL_NAME = "google/gemma-2-2b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Evaluation (LLM-as-a-judge) ---
JUDGE_MODEL = "gpt-4o" 
BERT_SCORE_MODEL = "roberta-large" # Recommended for English

# --- Paths (Local Storage) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval")

# Local paths
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "gemma-2-2b-it")
DATASET_CACHE_DIR = os.path.join(DATA_DIR, "huggingface")

def ensure_base_model():
    """로컬에 베이스 모델이 없으면 Hub에서 다운로드."""
    if os.path.exists(os.path.join(BASE_MODEL_PATH, "config.json")):
        return
    from huggingface_hub import snapshot_download
    print(f"Downloading {BASE_MODEL_NAME} -> {BASE_MODEL_PATH}")
    snapshot_download(BASE_MODEL_NAME, local_dir=BASE_MODEL_PATH, token=HF_TOKEN)

# Stage-specific output paths
SFT_MODEL_PATH = os.path.join(MODELS_DIR, "gemma-2b-gdpr-sft")
DPO_MODEL_PATH = os.path.join(MODELS_DIR, "gemma-2b-gdpr-dpo")
NEW_MODEL_NAME = DPO_MODEL_PATH # Default model for evaluation
DYNAMIC_DATASET_PATH = os.path.join(DATA_DIR, "gdpr_dynamic_dpo.jsonl")

# --- QLoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- SFT Training Parameters ---
SFT_LEARNING_RATE = 2e-5
SFT_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# --- DPO Configuration ---
DPO_BETA = 0.1
DPO_LEARNING_RATE = 5e-6
DPO_EPOCHS = 3
MAX_PROMPT_LENGTH = 1024
MAX_LENGTH = 2048

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
