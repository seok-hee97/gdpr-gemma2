import os
from dotenv import load_dotenv

load_dotenv()

# --- Model & Tokenizer ---
BASE_MODEL_NAME = "google/gemma-2-2b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Evaluation (LLM-as-a-judge) ---
JUDGE_MODEL = "gpt-4o" 

# --- Paths (Local Storage) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval")

# Redirect Hugging Face to download base models into the 'models/' directory
# This avoids permission issues and keeps all model artifacts in one place.
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

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
