import os
import sys
import time
sys.path.append("./")

import torch
import argparse
import yaml
import logging

from tqdm import tqdm
from transformers import AutoTokenizer
from src.patch import PatchLlama, PatchQwen
from src.auto_map import ModelMap
from src.evaluator import WikiText, LM_eval
from src.ptq import PTQ

parser = argparse.ArgumentParser(description='HBQ')
parser.add_argument('--config_dir', type=str, default='config/llama3-8b_wiki.yaml', help="Path to the configuration file (.yaml)")
parser.add_argument('--quant_config', type=str, default='config/baseline.yaml', help="Path to the quantization configuration file (.yaml)")
args = parser.parse_args()

class CompressLLM():
    def __init__(self, config_dir, quant_config_dir):
        with open(config_dir, 'r') as f:
            # load inference config
            self.config = yaml.full_load(f)
        with open(quant_config_dir, 'r') as f:
            # load quantization config
            self.quant_config = yaml.full_load(f)

        # detect device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.logger = self.initialize_logger()

        model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()
        
        # Patch model
        if "llama".lower() in self.config["model"]["model_type"].lower():
            patcher = PatchLlama(model)
        elif "qwen".lower() in self.config["model"]["model_type"].lower():
            patcher = PatchQwen(model)
        else:
            raise ValueError(f"Unsupported model type: {self.config['model']['model_type']}")
        
        start_time = time.time()
        print(f"Patching model")
        self.model = patcher.patch()
        print(f"Model patched, time: {(time.time() - start_time):.2f}s")
        self.task = PTQ(self.quant_config, self.model, self.tokenizer, self.logger)

    def initialize_logger(self):
        if self.config["save"]["logger"] is None:
            return None
        logname = self.config["save"]["logger"]
        run_dir = self.config["save"]["run_dir"]
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        logpath = os.path.join(run_dir, logname)

        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logpath, mode="w")
        console_handler = logging.StreamHandler()
        
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def create_model(self):
        # mapper
        model_type = self.config["model"]["model_type"]
        self.logger.info(f"Creating model {model_type}...")

        model_func = ModelMap(model_type)
        model = model_func.fetch()

        # map to device
        # model.to(self.device)
        return model
    
    def prepare_tokenizer(self):
        model_type = self.config["model"]["model_type"]
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        return tokenizer
    
    def ptq(self):
        faked_quantized_model = self.task.run()
        return faked_quantized_model

    def run(self):
        faked_quantized_model = self.ptq()

        task_names = self.config["eval"].get("tasks", None)
        if task_names is None:
            raise ValueError(f"Task is not specified in the config file")

        results = []
        for task in task_names:
            if task == "wikitext":
                evaluator = WikiText(faked_quantized_model, self.tokenizer)
                result = evaluator.run()
                results.append(result)
                self.logger.info(f"Wikitext2 PPL: {result:.4f}")
            else:
                evaluator = LM_eval(faked_quantized_model, self.tokenizer, task, self.config.get("eval", {}).get("batch_size", 1))
                result = evaluator.run()
                results.append(result)
                self.logger.info(f"{task} result: {results}")
                  
        return results

def starter():
    executor = CompressLLM(args.config_dir, args.quant_config)
    executor.run()

if __name__ == "__main__":
    # start task
    starter()