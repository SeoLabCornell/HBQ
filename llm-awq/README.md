# AWQ Baseline

AWQ (W4, group size 128) baseline used for the comparisons in the HBQ paper,
adapted from [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq).

Changes from upstream:

- Evaluation runs on `lm-evaluation-harness` 0.4 (`wikitext`, `winogrande`,
  `piqa`, `gsm8k_cot_llama`, `humaneval`, `mmlu`).
- Weight quantization is simulated (`--q_backend fake`), so no CUDA kernel
  build is required. `--q_backend real` still exists but needs the
  `awq_inference_engine` kernels from the upstream repo.
- Multi-GPU model sharding for the AWQ search on models that do not fit on a
  single device.

## Install

```bash
conda create -n awq python=3.10 -y
conda activate awq
pip install -e .
```

## Usage

To reproduce the whole Table 9 AWQ row in one go (all five base models plus a
final summary table; set `PARALLEL=1` to run one model per GPU):

```bash
bash scripts/run_table9_awq.sh
```

Each per-model script runs the AWQ scale/clip search, then evaluates the
quantized model on the tasks reported in the paper:

```bash
bash scripts/llama2_7b_example.sh          # wikitext / winogrande / piqa
bash scripts/llama3_8b_example.sh
bash scripts/llama3.2_3b_example.sh
bash scripts/qwen2.5_3b_example.sh
bash scripts/qwen2.5_7b_example.sh
bash scripts/llama3.1_8b_instruct_example.sh   # gsm8k / humaneval / mmlu
bash scripts/llama3.2_3b_instruct_example.sh
```

The search result is cached in `awq_cache/` and reused on re-runs; metrics are
written to `results/<model>-w4-g128-awq/<task>.json`. For the fp16 baseline,
run the same evaluation commands without `--w_bit`, `--q_group_size`,
`--load_awq`, and `--q_backend`.

## Reproduced results

Wikitext-2 perplexity and 0-shot accuracy (mean of Winogrande and PIQA),
AWQ W4/g128:

| Model | Wikitext-2 PPL | 0-shot avg |
|---|---|---|
| Llama-2-7B | 5.60 | 73.1 |
| Llama-3-8B | 6.53 | 76.5 |
| Llama-3.2-3B | 8.22 | 72.6 |
| Qwen2.5-3B | 8.46 | 72.6 |
| Qwen2.5-7B | 7.09 | 75.6 |

## Reference

```bibtex
@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
```
