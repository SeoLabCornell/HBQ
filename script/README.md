# Result Reproduction Guideline

This file contains reproduction script and guideline for tables and figures in original HBQ papers.

These scripts should be run under script/, with `conda activate hbq` unless otherwise specified. The conda env setup guideline is under HBQ-llm/README.md.

GPU requirement: For evaluation on Llama3.1-70B/Mixtral8x7B models, we use 4 A6000-48GB GPUs, for all other evaluation 1 A6000-48GB should be sufficient.

---

Figure 3:
- Reproduce PPL results: `python3 scale_scheme_block_size.py`, results recorded under `save/scale_scheme_block_size/results.tsv`
- Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Plotting: `python3 plot_fig3.py`

Figure 4(b):
- Reproduce PPL results: `python3 W4A8_format_block_size.py`, results recorded under `save/W4A8_format_block_size/results.tsv` 
- Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Plotting: `python3 plot_fig4b.py`

Figure 5:
- Reproduce PPL results: `python3 W4A5_format_block_size.py`, results recorded under `save/scale_scheme_block_size/W4A5_format_block_size/results.tsv` 
- Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Plotting: `python3 plot_fig5.py`

Figure 6:
- Reproduce PPL results: `python3 block_size_activation_bitwidth.py`, results recorded under `save/block_size_activation_bitwidth/results.tsv` 
- Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Reproduce energy results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Plotting: `python3 plot_fig6.py`

Figure 10:
- Reproduce PPL results: `python3 micro_block_size.py` (for L2 PPL), `python3 block_size_activation_bitwidth.py` (for L1-only PPL), results recorded under `save/micro_block_size/` and `save/block_size_activation_bitwidth/`.
- Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- Plotting: `python3 plot_fig10.py`

Figure 11:
(We hand-draw this figure, but the numbers can be found from other figures, which are reproduced above)
- BQ PPL and area: from Figure 6
- HBQ PPL and area: from Figure 10

Figure 16:
- Reproduce energy numbers + plotting: `python3 plot_fig16.py` (writes `fig16.pdf`)
- All energy numbers are computed by `perf_model.py` from the model architectures and hardware parameters (PrimeTime pJ/MAC, DRAM/SRAM energy per access); run `python3 perf_model.py` to dump every intermediate number
- Perplexity line uses the measured KV-cache-quantized PPL (same as Table 10)

Figure 18:
- Reproduce speedup numbers + plotting: `python3 plot_fig18.py` (writes `fig18.pdf`)
- Speedups are computed by `perf_model.py` from prefill MAC counts (1K/4K/16K) and per-scheme area (um^2/MAC, from synthesis)

Table 2: 
- Reproduce PPL results: `python3 nvfp4_scaling_ppl.py`, results recorded under `save/nvfp4_scaling_ppl/`.

Table 3:
- Reproduce accuracy results: `python3 activation_bit_benchmark.py`, results recorded under `save/activation_bit_benchmark/results.tsv` (We use 4 A6000-48GB to run Llama3.1-70b/Mixtral-8x7b)

Table 4:
- Reproduce PPL results: `python3 KV_block_size.py`, results recorded under `save/KV_block_size/`

Table 6:
- Reproduce PPL results: `python3 L2_scale_ablation.py`, results recorded under `save/L2_scale_ablation/`

Table 9:
- Reproduce MXFP4/NVFP4/HBQ PPL numbers: `python3 main_eval.py`, results recorded under `save/main_eval/results.tsv`
- Reproduce AWQ numbers: `cd ../llm-awq && bash scripts/run_table9_awq.sh` (runs the AWQ search + wikitext/winogrande/piqa evals for all five base models and prints the summary table; `PARALLEL=1` runs one model per GPU; see llm-awq/README.md for setup)
- Amove[1]/MANT[2]/MicroScropiQ[3] results are from original papers

Table 10:
- Reproduce MXFP/NVFP/HBQ PPL numbers: `python3 kv_eval.py`, results recorded under `save/kv_eval/`. Please use only one GPU to run this evaluation, otherwise Torch Dynamo Compilation might cause cache problem across ranks.
- Amove[1]/MXFP+[4]/MXFP++[4] results are from original papers

Table 11:
- Reproduce MXFP/NVFP/HBQ accuracy numbers: `export CUDA_VISIBLE_DEVICES=0 && python3 reasoning.py`, results recorded under `save/reasoning/` (Each evaluation takes up to 2 days on a single A6000 GPU with 48GB)

Table 12:
- For long generation task accuracy evaluation, we use vllm framework to evaluate. Please refer to vllm/README.md and follow the instruction to install environment and run the evaluations. (Each evaluation takes 1~2 days on a single A6000 with 48GB)

### Reference
1. Xie, Xilong, et al. "Amove: Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type." Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture. 2025.
2. Hu, Weiming, et al. "M-ANT: Efficient low-bit group quantization for LLMs via mathematically adaptive numerical type." 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2025.
3. Ramachandran, Akshat, Souvik Kundu, and Tushar Krishna. "Microscopiq: Accelerating foundational models through outlier-aware microscaling quantization." Proceedings of the 52nd Annual International Symposium on Computer Architecture. 2025.
4. Lee, Jungi, et al. "MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving." Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture. 2025.