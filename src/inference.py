from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from . import config
import os

class GDPRInference:
    def __init__(self, model_path="dpo"):
        """
        model_path: "base", "sft", "dpo" 중 선택하거나 특정 경로 입력
        """
        if model_path == "base":
            self.model_name = config.BASE_MODEL_PATH
        elif model_path == "sft":
            self.model_name = config.SFT_MODEL_PATH
        elif model_path == "dpo":
            self.model_name = config.DPO_MODEL_PATH
        else:
            self.model_name = model_path

        print(f"Initializing Inference Engine for path: {self.model_name}")
        config.ensure_base_model()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model (하드웨어 자동 감지 + Gemma2 최적화)
        load_kwargs = {
            "torch_dtype": config.TORCH_DTYPE,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # Gemma2 sliding window 호환
        }
        if config.USE_QUANTIZATION:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.TORCH_DTYPE,
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_PATH, **load_kwargs
        )

        # Check if it's an adapter path (SFT/DPO outputs are adapters by default in our scripts)
        # 경로 비교는 realpath로 정규화 (상대경로 vs 절대경로 불일치 방지)
        adapter_path = os.path.join(self.model_name, "adapter_config.json")
        is_base_model = os.path.realpath(self.model_name) == os.path.realpath(config.BASE_MODEL_PATH)

        if is_base_model:
            print(f"Using Base Model (already loaded): {self.model_name}")
            self.model = base_model
        elif os.path.exists(adapter_path):
            print(f"Loading Adapter from: {self.model_name}")
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_name}\n해당 단계의 학습을 먼저 완료해 주세요.")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
        )

    def generate(self, prompt, max_new_tokens=512, temperature=0.1, top_p=0.2):
        # System prompt for GDPR expertise
        system_prompt = "You are a professional GDPR compliance assistant. Provide accurate, legal, and clear guidance based on the General Data Protection Regulation."
        
        # Gemma-2 often works best when system prompt is integrated into the user message if not natively supported
        full_user_content = f"{system_prompt}\n\nQuestion: {prompt}"
        
        messages = [{"role": "user", "content": full_user_content}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            truncation=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract response after the last turn
        generated_text = outputs[0]["generated_text"]
        if "<start_of_turn>model\n" in generated_text:
            return generated_text.split("<start_of_turn>model\n")[-1].strip()
        return generated_text.strip()

if __name__ == "__main__":
    # 간단한 테스트 실행
    print("Loading DPO model for test...")
    # Fix: model_stage -> model_path
    infer = GDPRInference(model_path="dpo")
    question = "What is the legal basis for processing data in clinical trials?"
    print(f"\nQuestion: {question}")
    print(f"Answer: {infer.generate(question)}")
