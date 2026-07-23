"""
Load model architecture from different sources
"""

import torch

from transformers import AutoModelForCausalLM

# TODO: expand this list to support more model architectures
MODEL_LIBRARY_MAP = {
    'meta-llama/Llama-2-7b-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-1B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-70B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Meta-Llama-3-8B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Meta-Llama-3-8B': ('transformers', 'AutoModelForCausalLM'),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': ('transformers', 'AutoModelForCausalLM'),
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-3B': ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-7B': ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-7B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'mistralai/Mixtral-8x7B-v0.1': ('transformers', 'AutoModelForCausalLM'),
}


class ModelMap:
    def __init__(self, model_name:str):
        self.model_name = model_name
        
    def fetch(self):
        if self.model_name not in MODEL_LIBRARY_MAP:
            raise ValueError(f"Model: {self.model_name} is unknown! Available models: {MODEL_LIBRARY_MAP.keys()}")

        lib_name, sub_name = MODEL_LIBRARY_MAP[self.model_name]

        if lib_name == "transformers":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown model library {lib_name}")

        return model