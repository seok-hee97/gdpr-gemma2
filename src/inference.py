import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from . import config
import os

class GDPRInference:
    def __init__(self, model_stage="dpo"):
        """
        model_stage: "base", "sft", "dpo" 중 선택하거나 특정 경로 입력
        """
        if model_stage == "base":
            self.model_name = config.BASE_MODEL_NAME
        elif model_stage == "sft":
            self.model_name = config.SFT_MODEL_PATH
        elif model_stage == "dpo":
            self.model_name = config.DPO_MODEL_PATH
        else:
            self.model_name = model_stage

        print(f"Initializing Inference Engine for stage: {model_stage}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_NAME
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Check if it's an adapter path (SFT/DPO outputs are adapters by default in our scripts)
        adapter_path = os.path.join(self.model_name, "adapter_config.json")
        if os.path.exists(adapter_path):
            print(f"Loading Adapter from: {self.model_name}")
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
        else:
            print(f"Loading Full Model from: {self.model_name}")
            self.model = base_model if self.model_name == config.BASE_MODEL_NAME else AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
        )

    def generate(self, prompt, max_new_tokens=512, temperature=0.1, top_p=0.2):
        messages = [{"role": "user", "content": prompt}]
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
        
        return outputs[0]["generated_text"].split("<start_of_turn>model\n")[-1]

if __name__ == "__main__":
    # 간단한 테스트 실행
    print("Loading DPO model for test...")
    infer = GDPRInference(model_stage="dpo")
    question = "What is the legal basis for processing data in clinical trials?"
    print(f"\nQuestion: {question}")
    print(f"Answer: {infer.generate(question)}")
