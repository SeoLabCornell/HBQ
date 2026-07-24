# Result Reproduction Guideline

This file contains reproduction script and guideline for tables and figures in original HBQ papers.

*These scripts should be run under script/.*

---

Figure 3:
- [v] Reproduce PPL results: `python3 scale_scheme_block_size.py`
- [v] Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- [v] Plotting: `python3 plot_fig3.py`

Figure 4(b):
- [v] Reproduce PPL results: `python3 W4A8_format_block_size.py`
- [v] Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- [v] Plotting: `python3 plotfig4b.py`

Figure 5:
- [v] Reproduce PPL results: `python3 W4A5_format_block_size.py`
- [v] Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- [v] Plotting: `python3 fig5.py`

Figure 6:
- [] Reproduce PPL results: `python3 block_size_activation_bitwidth.py`
- [v] Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- [] Plotting

Figure 10:
- [] Reproduce PPL results: `python3 micro_block_size.py` (for L2 PPL), `python3 block_size_activation_bitwidth.py` (for L1-only PPL)
- [v] Reproduce area results: Please follow hardware/README.md: *Paper Result Reproduction Guideline*
- [] Plotting

Figure 11:
- HBQ PPL numbers are the same as Figure 10
- BQ PPL numbers are the same as Figure 6

Figure 16:
- [v] Reproduce energy numbers + plotting: `python3 plot_fig16.py` (writes `fig16.pdf`)
- All energy numbers are computed by `perf_model.py` from the model architectures and hardware parameters (PrimeTime pJ/MAC, DRAM/SRAM energy per access); run `python3 perf_model.py` to dump every intermediate number
- Perplexity line uses the measured KV-cache-quantized PPL (same as Table 10)

Figure 18:
- [v] Reproduce speedup numbers + plotting: `python3 plot_fig18.py` (writes `fig18.pdf`)
- Speedups are computed by `perf_model.py` from prefill MAC counts (1K/4K/16K) and per-scheme area (um^2/MAC, from synthesis)

Table 2: 
- [v] Reproduce PPL results: `python3 nvfp4_scaling_ppl.py`

Table 3:
- [] Reproduce accuracy results: `python3 activation_bit_benchmark.py`

Table 4:
- [v] Reproduce PPL results: `python3 KV_block_size.py`

Table 6:
- [v] Reproduce PPL results: `python3 L2_scale_ablation.py`

Table 7:
- [v] Reproduce results: `python3 psum_quant.py`

Table 9:
- [] Reproduce MXFP4/NVFP4/HBQ PPL numbers: `python3 main_eval.py`
- [v] Reproduce AWQ numbers: `cd ../llm-awq && bash scripts/run_table9_awq.sh` (runs the AWQ search + wikitext/winogrande/piqa evals for all five base models and prints the summary table; `PARALLEL=1` runs one model per GPU; see llm-awq/README.md for setup)
- Amove[1]/MANT[2]/MicroScropiQ[3] results are from original papers

Table 10:
- [v] Reproduce MXFP/NVFP/HBQ PPL numbers: `python3 kv_eval.py`
- Amove[1]/MXFP+[4]/MXFP++[4] results are from original papers

Table 11:
- [] Reproduce MXFP/NVFP/HBQ accuracy numbers: `python3 reasoning.py`

Table 12:
- TODO: hardware
- TODO: accuracy

### Nuances didn't mention in the paper:
- When using FP8-scale under high activation precision (>= 6-bit), FP8-scale face similar scaling factor quantization error as the floor/rounding issue in PoT-scale. So we use use_ceil to calculate FP8 scaling when activation is >= 6-bit. See quantizer.py MXFPQuantizer.get_shared_scale() for how the scaling factor is calculated.
- For PoT-round, we found out that only qk projection benefits from rounding, floor is better for other layers. So we only use rounding in QK.

### Reference
1. Xie, Xilong, et al. "Amove: Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type." Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture. 2025.
2. Hu, Weiming, et al. "M-ANT: Efficient low-bit group quantization for LLMs via mathematically adaptive numerical type." 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2025.
3. Ramachandran, Akshat, Souvik Kundu, and Tushar Krishna. "Microscopiq: Accelerating foundational models through outlier-aware microscaling quantization." Proceedings of the 52nd Annual International Symposium on Computer Architecture. 2025.
4. Lee, Jungi, et al. "MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving." Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture. 2025.