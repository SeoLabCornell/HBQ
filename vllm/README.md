# HBQ vLLM evaluation

This is a minimal self-contained directory needed to evaluate `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` on GSM8K and MATH-500 with vLLM using fake-quantized MXFP, NVFP, or HBQ weights, activations, and KV cache.

## Environment

Create the dedicated environment from this directory:

```bash
conda env create -f environment.yml
conda activate hbq-vllm
```

Hugging Face access to the model and datasets is required on first use.

## Run

Run commands from `HBQ-llm/vllm`.

Choose one model task config:

- `config/llama-8b-distill_gsm8k.yaml`
- `config/llama-8b-distill_math500.yaml`

Choose one KV-quantized format config:

- `config/mxfp_w4a8kv4.yaml` — MXFP W4A8, 4-bit KV
- `config/mxfp_w4a8kv8.yaml` — MXFP W4A8, 8-bit KV
- `config/nvfp_w4a8kv4.yaml` — NVFP W4A8, 4-bit KV
- `config/hbq_a_new_kv.yaml` — HBQ-A, 4-bit KV
- `config/hbq_e_new_kv.yaml` — HBQ-E, 4-bit KV

For example:

```bash
python main.py \
  --config_dir config/llama-8b-distill_gsm8k.yaml \
  --quant_config config/mxfp_w4a8kv8.yaml

python main.py \
  --config_dir config/llama-8b-distill_math500.yaml \
  --quant_config config/nvfp_w4a8kv4.yaml

python main.py \
  --config_dir config/llama-8b-distill_gsm8k.yaml \
  --quant_config config/hbq_a_new_kv.yaml
```

To run MXFP W4A8KV8, NVFP W4A8KV4, HBQ-A-KV, and HBQ-E-KV on both GSM8K and MATH-500:

```bash
./run_all_benchmarks.sh
```

The eight runs execute sequentially. Their complete console logs are retained under `run_logs/<format>/<benchmark>.log`. A failure in one run does not prevent the remaining configurations from running; the script prints a summary and returns a nonzero status after all runs if any failed.

All arguments passed to the script are forwarded to `main.py`. For example, a short eight-run smoke test is:

```bash
./run_all_benchmarks.sh --limit 2 --max_gen_toks 128
```

`PYTHON_BIN` and `LOG_ROOT` can override the Python executable and output
directory:

```bash
PYTHON_BIN=/path/to/python LOG_ROOT=/path/to/logs ./run_all_benchmarks.sh
```

For a quick smoke test, add `--limit 2 --max_gen_toks 128`. MATH-500 normally uses a 32K model context and up to 8192 generated tokens, so it needs substantially more GPU memory and time than that smoke test.

## Included files

`main.py` drives vLLM and lm-eval. `src/hbq_vllm_quant.py` registers the vLLM quantization plugin and patches Llama attention for post-RoPE Q/K/V fak quantization. `src/quantizer.py` implements MXFP, NVFP, and HBQ arithmetic. `lm_eval_tasks/` contains only the custom MATH-500 task; GSM8K is supplied by lm-eval.

---

## Paper Result (Table 12) Reproduction Guideline

4 configurations across two benchmarks, in total 8 evaluations. Each evaluation takes 1~2 days to finish.

```
conda activate hbq-vllm
./run_all_benchmarks.sh # result recorded under run_logs/
```