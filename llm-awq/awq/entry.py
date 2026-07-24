import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument(
    "--dump_fake", type=str, default=None, help="save fake-quantized model"
)
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
args = parser.parse_args()

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "MixtralDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        # With multiple GPUs, stream the fp16 weights straight onto them instead
        # of materializing the whole model in CPU RAM first. A 70B fp16 model is
        # ~140GB and OOM-kills the host *during loading* (before AWQ even runs).
        # We reserve some headroom per GPU for AWQ's per-layer activations, and
        # drop accelerate's offload hooks afterwards so run_awq / quantization
        # can manage device placement manually.
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            # Shard the model across GPUs at load time. For the AWQ search
            # (run_awq), keep device 0 as light as possible (balanced_low_0) so
            # it has room to serve as the AWQ compute device; otherwise use all
            # GPUs evenly — a 70B fp16 model (~141GB) doesn't fit if one GPU is
            # held back. Reserve a little headroom on each GPU for
            # fragmentation / activations.
            reserve = 4 * (1024**3)
            kwargs["device_map"] = "balanced_low_0" if args.run_awq else "balanced"
            kwargs["max_memory"] = {
                i: max(
                    torch.cuda.get_device_properties(i).total_memory - reserve, 0
                )
                for i in range(n_gpu)
            }
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, **kwargs
        )
        device_map_used = getattr(model, "hf_device_map", None)
        if device_map_used is not None:
            # If accelerate couldn't fit the fp16 model on the visible GPUs
            # it silently spills the remainder to CPU and then to disk
            # ("meta" device). The downstream AWQ-apply / real-quantize
            # passes then touch those meta tensors, thrash host RAM, and get
            # SIGKILL'd by the OOM killer with NO Python traceback. Detect
            # that here and fail loudly with an actionable message instead.
            offloaded = {
                name: dev
                for name, dev in device_map_used.items()
                if dev in ("cpu", "disk") or (isinstance(dev, str) and dev == "meta")
            }
            if offloaded:
                visible = [
                    torch.cuda.get_device_properties(i).name
                    for i in range(n_gpu)
                ]
                raise RuntimeError(
                    "The fp16 model did not fit entirely on the visible "
                    f"GPUs, so {len(offloaded)} module(s) were offloaded to "
                    f"{sorted(set(offloaded.values()))}. Quantizing offloaded "
                    "(meta/cpu/disk) weights does not work and will be "
                    "OOM-killed silently.\n"
                    f"Visible GPUs ({n_gpu}): {visible}\n"
                    "Fixes: (1) make sure all GPUs are actually free "
                    "(`nvidia-smi`) and not held by a previous killed run; "
                    "(2) check CUDA_VISIBLE_DEVICES exposes all of them; "
                    "(3) lower the per-GPU `reserve` above if you have only a "
                    "little overflow."
                )
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "MixtralDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def run_lm_eval(model, tokenizer, task_names, limit=None, batch_size=1, num_fewshot=0):
    lm_object = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        add_bos_token=False,
        batch_size=batch_size,
    )

    eval_kwargs = {
        "model": lm_object,
        "tasks": task_names,
        "log_samples": False,
        "limit": limit,
    }

    if "humaneval" in task_names:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        eval_kwargs["confirm_run_unsafe_code"] = True
    elif "mmlu" in task_names:
        eval_kwargs["num_fewshot"] = 5
    elif "gsm8k_cot_llama" in task_names:
        eval_kwargs["fewshot_as_multiturn"] = True
        eval_kwargs["apply_chat_template"] = True
        eval_kwargs["gen_kwargs"] = "max_length=2048"
    elif num_fewshot > 0:
        eval_kwargs["num_fewshot"] = num_fewshot

    with torch.no_grad():
        results = lm_eval.simple_evaluate(**eval_kwargs)

    print(make_table(results))
    return results


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        print(f"Results {args.output_path} already generated. Overwrite.")

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()
    model, enc = build_model_and_enc(args.model_path, args.dtype)

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")
            results = run_lm_eval(
                model=model,
                tokenizer=enc,
                task_names=task_names,
                limit=None,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
            )

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            if isinstance(results, dict) and "config" in results:
                results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                # lm_eval's result dict carries non-JSON-native values (e.g.
                # torch.dtype in the model config). Without a fallback encoder
                # the dump raises *after* the evaluation has finished and the
                # whole run's results are lost.
                json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
