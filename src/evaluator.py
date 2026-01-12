"""
Evaluator 
"""
import torch
import lm_eval
from tqdm import tqdm

from datasets import load_dataset
from transformers import set_seed
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM
set_seed(5000)

class WikiText():
    def __init__(self, model, tokenizer, chunk_size=2048):

        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        # prepare dataset
        self.text = self.get_text()

    def __name__(self):
        return "WikiText"
    
    def get_text(self):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])

    @torch.no_grad
    def run(self):
        self.model.eval()
        input_ids = self.tokenizer(self.text, return_tensors="pt").input_ids.to(next(self.model.parameters()).device)
        nsamples = input_ids.size(1) // self.chunk_size
        losses = []
        for i in tqdm(range(nsamples), desc="Perplexity on Wikitext-2", unit="chunk"):
            x = input_ids[:, i*self.chunk_size:(i+1)*self.chunk_size]
            losses.append(self.model(x, labels=x).loss.float().cpu())
        return torch.exp(torch.stack(losses).mean()).item()
   
class LM_eval:
    lm_eval_task_list = [
        "gsm8k_cot_llama",
        "humaneval",
        "coqa",
        "winogrande",
        "mmlu_flan_n_shot_generative_abstract_algebra",
        "mmlu",
        "longbench_qasper",
        "squadv2",
        "piqa",
        "c4"
    ]
    def __init__(self, model, tokenizer,task_name:str, batch_size:int=1):
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.batch_size = batch_size
        if task_name not in self.lm_eval_task_list:
            raise ValueError(f"Invalid task: {task_name}")
    def run(self):
        return run_lm_eval_zero_shot(self.model, self.tokenizer, [self.task_name], limit=None, batch_size=self.batch_size)

def run_lm_eval_zero_shot(model, tokenizer, task_list=["gsm8k_cot_llama"], limit=100, batch_size=128):
    # indexes all tasks from the lm_eval/tasks subdirectory.
    # Alternatively, you can set TaskManager(include_path="path/to/my/custom/task/configs")
    # to include a set of tasks in a separate directory.
    lm_object = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # lm_object = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    # logger.info(f"Evaluation, Task(s): {task_list}")
    with torch.no_grad():
        if "humaneval" in task_list:
            import os
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model=lm_object,
                #model_args= "add_bos_token=True" if model_type == "jamba" else "",
                tasks=task_list,
                # task_manager=task_manager,
                log_samples=False,
                limit=limit,
                confirm_run_unsafe_code=True
            ) 
        elif "mmlu" in task_list:
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model=lm_object,
                tasks=task_list,
                log_samples=False,
                limit=limit,
                num_fewshot=5
            ) 
        elif "gsm8k_cot_llama" in task_list:
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model=lm_object,
                #model_args= "add_bos_token=True" if model_type == "jamba" else "",
                tasks=task_list,
                # task_manager=task_manager,
                log_samples=False,
                # num_fewshot=0,
                limit=limit,
                fewshot_as_multiturn=True, apply_chat_template=True,
                gen_kwargs="max_length=2048"
            ) 
        else:
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model=lm_object,
                #model_args= "add_bos_token=True" if model_type == "jamba" else "",
                tasks=task_list,
                # task_manager=task_manager,
                log_samples=False,
                limit=limit,
            ) 
    res = make_table(results)
    print(res)
    
    return results['results']
