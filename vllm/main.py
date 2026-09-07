from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Match HBQ-llm: avoid FP16 reduced-precision reductions in large fake-quantized matmuls.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Registers the hbq_nvfp4 vLLM quantization plugin.
import src.hbq_vllm_quant  # noqa: F401,E402
from src.hbq_vllm_quant import set_active_quant_config  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def resolve_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    rooted = ROOT / candidate
    if rooted.exists():
        return rooted
    return candidate


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(resolve_path(path), "r") as f:
        return yaml.safe_load(f)


def initialize_logger(config: dict[str, Any]) -> logging.Logger | None:
    save_cfg = config.get("save", {})
    logname = save_cfg.get("logger")
    if logname is None:
        return None

    run_dir = Path(save_cfg.get("run_dir", "save"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logpath = run_dir / logname

    logger = logging.getLogger(f"hbq-vllm:{logpath}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(logpath, mode="w")
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_wikitext2() -> str:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def prepare_tokenizer(model_type: str):
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return tokenizer


def uses_hbq_quantization(quant_config: dict[str, Any]) -> bool:
    quantization = quant_config.get("quantization", {})
    return quantization.get("xqtype", "none") != "none" or quantization.get("wqtype", "none") != "none"


def hbq_quantization_name(config: dict[str, Any]) -> str | None:
    return "hbq_nvfp4" if config.get("_hbq_use_quantization", True) else None


def build_llm(config: dict[str, Any]) -> LLM:
    model_type = config["model"]["model_type"]
    vllm_cfg = config.get("vllm", {})
    return LLM(
        model=model_type,
        tokenizer=model_type,
        dtype=vllm_cfg.get("dtype", "float16"),
        quantization=hbq_quantization_name(config),
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.90),
        enforce_eager=vllm_cfg.get("enforce_eager", True),
        max_model_len=vllm_cfg.get("max_model_len", 2048),
        trust_remote_code=vllm_cfg.get("trust_remote_code", True),
    )


    try:
        import ray  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lm-eval==0.4.8 requires Ray for its vLLM adapter. "
            "Install the environment.yml dependencies or run: pip install ray==2.55.1"
        ) from exc

def build_lm_eval_vllm(config: dict[str, Any]):
    import lm_eval.models.vllm_causallms as lm_eval_vllm
    from lm_eval.models.vllm_causallms import VLLM as LMEvalVLLM

    if not hasattr(lm_eval_vllm, "get_tokenizer"):
        def _get_tokenizer(tokenizer_name: str, tokenizer_mode: str = "auto", trust_remote_code: bool = False, revision: str | None = None):
            use_fast = tokenizer_mode != "slow"
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
                use_fast=use_fast,
            )

        lm_eval_vllm.get_tokenizer = _get_tokenizer

    if not getattr(lm_eval_vllm.VLLM, "_hbq_vllm_022_generate_patch", False):
        def _model_generate_compat(self, requests=None, generate: bool = False, max_tokens: int | None = None, stop=None, **kwargs):
            from more_itertools import distribute
            from vllm import LLM as VLLMEngine
            from vllm import SamplingParams as VLLMSamplingParams

            if requests is not None:
                fallback_token_id = (
                    getattr(self.tokenizer, "bos_token_id", None)
                    or getattr(self.tokenizer, "eos_token_id", None)
                    or self.eot_token_id
                    or 0
                )
                requests = [list(req) if len(req) > 0 else [fallback_token_id] for req in requests]

            if generate:
                kwargs = self.modify_gen_kwargs(kwargs)
                sampling_params = VLLMSamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
            else:
                sampling_params = VLLMSamplingParams(
                    temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
                )

            if self.data_parallel_size > 1:
                @lm_eval_vllm.ray.remote
                def run_inference_one_model(model_args, sampling_params, request_batch, lora_request):
                    llm = VLLMEngine(**model_args)
                    return llm.generate(
                        prompts=request_batch,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                    )

                request_batches = [list(x) for x in distribute(self.data_parallel_size, requests)]
                object_refs = [
                    run_inference_one_model.remote(self.model_args, sampling_params, request_batch, self.lora_request)
                    for request_batch in request_batches
                ]
                results = lm_eval_vllm.ray.get(object_refs)
                lm_eval_vllm.ray.shutdown()
                return lm_eval_vllm.undistribute(results)

            return self.model.generate(
                prompts=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )

        lm_eval_vllm.VLLM._model_generate = _model_generate_compat
        lm_eval_vllm.VLLM._hbq_vllm_022_generate_patch = True

    model_type = config["model"]["model_type"]
    vllm_cfg = config.get("vllm", {})
    eval_cfg = config.get("eval", {})

    kwargs: dict[str, Any] = {
        "pretrained": model_type,
        "tokenizer": model_type,
        "dtype": vllm_cfg.get("dtype", "float16"),
        "quantization": hbq_quantization_name(config),
        "tensor_parallel_size": vllm_cfg.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": vllm_cfg.get("gpu_memory_utilization", 0.90),
        "max_model_len": vllm_cfg.get("max_model_len", 2048),
        "trust_remote_code": vllm_cfg.get("trust_remote_code", True),
        "batch_size": eval_cfg.get("batch_size", 1),
        "add_bos_token": eval_cfg.get("add_bos_token", False),
        "max_gen_toks": eval_cfg.get("max_gen_toks", 2048),
    }
    if "max_batch_size" in eval_cfg:
        kwargs["max_batch_size"] = eval_cfg["max_batch_size"]
    if "seed" in vllm_cfg:
        kwargs["seed"] = vllm_cfg["seed"]
    if "enforce_eager" in vllm_cfg:
        kwargs["enforce_eager"] = vllm_cfg["enforce_eager"]

    return LMEvalVLLM(**kwargs)


def prompt_logprob_nll(prompt_logprobs, token_ids: list[int]) -> tuple[float, int]:
    total_nll = 0.0
    count = 0

    # vLLM returns one entry per prompt token. The first token has no previous
    # context, so it is intentionally excluded like HF label shifting.
    for idx in range(1, len(token_ids)):
        entry = prompt_logprobs[idx]
        token_id = token_ids[idx]
        if entry is None:
            continue

        if token_id in entry:
            logprob = entry[token_id].logprob
        else:
            # With prompt_logprobs=0, vLLM should include only the sampled prompt
            # token. This fallback handles versions that return string keys.
            token_key = str(token_id)
            if token_key not in entry:
                raise KeyError(f"Missing prompt logprob for token id {token_id} at position {idx}")
            logprob = entry[token_key].logprob

        total_nll -= float(logprob)
        count += 1

    return total_nll, count


def run_wikitext_ppl(llm: LLM, tokenizer, chunk_size: int) -> float:
    text = get_wikitext2()
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    nsamples = len(input_ids) // chunk_size
    if nsamples == 0:
        raise ValueError(f"WikiText-2 tokenized length is smaller than chunk_size={chunk_size}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
    )

    total_nll = 0.0
    total_tokens = 0
    for i in tqdm(range(nsamples), desc="Perplexity on Wikitext-2", unit="chunk"):
        chunk = input_ids[i * chunk_size : (i + 1) * chunk_size]
        outputs = llm.generate(
            prompts=[chunk],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        nll, count = prompt_logprob_nll(outputs[0].prompt_logprobs, chunk)
        total_nll += nll
        total_tokens += count

    return math.exp(total_nll / total_tokens)


def run_lm_eval_task(
    task: str,
    config: dict[str, Any],
    limit: int | float | None,
    logger: logging.Logger | None,
) -> dict[str, Any]:
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    eval_cfg = config.get("eval", {})
    local_task_dir = ROOT / "lm_eval_tasks"
    task_manager = TaskManager(include_path=str(local_task_dir)) if local_task_dir.exists() else None
    lm_object = build_lm_eval_vllm(config)

    task_limit = limit if limit is not None else eval_cfg.get("limit")
    if logger is not None:
        logger.info("Running lm-eval task %s with limit=%s", task, task_limit)

    gen_kwargs = eval_cfg.get("gen_kwargs")

    confirm_run_unsafe_code = eval_cfg.get("confirm_run_unsafe_code", task.startswith("humaneval"))
    if confirm_run_unsafe_code:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    results = lm_eval.simple_evaluate(
        model=lm_object,
        tasks=[task],
        log_samples=eval_cfg.get("log_samples", False),
        limit=task_limit,
        num_fewshot=eval_cfg.get("num_fewshot"),
        fewshot_as_multiturn=eval_cfg.get("fewshot_as_multiturn", (task == "gsm8k_cot_llama") or (task == "hendrycks_math500")),
        apply_chat_template=eval_cfg.get("apply_chat_template", (task == "gsm8k_cot_llama") or (task == "hendrycks_math500")),
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
    )

    task_result = results["results"].get(task, {})
    table = make_table(results)
    result_json = json.dumps(task_result, sort_keys=True)
    if logger is not None:
        logger.info("\n%s", table)
        logger.info("%s result: %s", task, result_json)
    else:
        print(table)
        print(f"{task} result: {result_json}")

    return task_result


def validate_quant_config(quant_config: dict[str, Any]) -> None:
    quantization = quant_config.get("quantization", {})
    unsupported = [
        "quant_attn_wgt",
        "quant_output_layer",
    ]
    enabled = [name for name in unsupported if quantization.get(name, False)]
    if enabled:
        raise NotImplementedError(
            "HBQ-vllm currently implements direct linear W/A PTQ plus optional K/V cache fake quantization. "
            f"Disable unsupported flags: {enabled}"
        )


def main() -> list[float]:
    parser = argparse.ArgumentParser(description="HBQ-vllm direct PTQ")
    parser.add_argument("--config_dir", type=str, default="config/llama3-8b_wiki.yaml")
    parser.add_argument("--quant_config", type=str, default="config/nvfp4.yaml")
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit for smoke tests")
    parser.add_argument("--max_gen_toks", type=int, default=None, help="Optional lm-eval generation cap override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional lm-eval batch size override")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None, help="Optional vLLM GPU memory utilization override")
    parser.add_argument("--max_model_len", type=int, default=None, help="Optional vLLM max model length override")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Optional vLLM tensor parallel size override")
    args = parser.parse_args()

    config = load_yaml(args.config_dir)
    if args.max_gen_toks is not None:
        config.setdefault("eval", {})["max_gen_toks"] = args.max_gen_toks
    if args.batch_size is not None:
        config.setdefault("eval", {})["batch_size"] = args.batch_size
    if args.gpu_memory_utilization is not None:
        config.setdefault("vllm", {})["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        config.setdefault("vllm", {})["max_model_len"] = args.max_model_len
    if args.tensor_parallel_size is not None:
        config.setdefault("vllm", {})["tensor_parallel_size"] = args.tensor_parallel_size
    quant_config = load_yaml(args.quant_config)
    validate_quant_config(quant_config)
    use_hbq_quantization = uses_hbq_quantization(quant_config)
    config["_hbq_use_quantization"] = use_hbq_quantization
    if use_hbq_quantization:
        set_active_quant_config(quant_config)

    logger = initialize_logger(config)

    task_names = config.get("eval", {}).get("tasks")
    if task_names is None:
        raise ValueError("Task is not specified in the config file")

    results = []
    llm = None
    tokenizer = None
    for task in task_names:
        if task == "wikitext":
            if llm is None:
                if logger is not None:
                    logger.info("Creating vLLM model %s...", config["model"]["model_type"])
                tokenizer = prepare_tokenizer(config["model"]["model_type"])
                llm = build_llm(config)
            chunk_size = config.get("eval", {}).get("chunk_size", 2048)
            result = run_wikitext_ppl(llm, tokenizer, chunk_size)
            results.append(result)
            message = f"Wikitext2 PPL: {result:.4f}"
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        else:
            results.append(run_lm_eval_task(task, config, args.limit, logger))

    return results


if __name__ == "__main__":
    main()
