Figure 2: 
- [v] getting ppl numbers: nvfp4_scaling_ppl.py

Figure 3:
- [v] getting ppl numbers: scale_scheme_block_size.py
- [] getting area numbers:
- [] plotting

Figure 4(b):
- [v] getting ppl numbers: W4A8_format_block_size.py
- [] getting area numbers:
- [] plotting

Figure 5:
- [v] getting numbers: W4A5_format_block_size.py (need rerun)
- [] getting area numbers:
- [] plotting

Table 3:
- [] getting numbers: activation_bit_benchmark.py

Table 4:
- [v] getting ppl numbers: KV_block_size.py

Figure 10:
- [v] getting ppl numbers: micro_block_size.py
- [] getting area numbers:
- [] plotting

Figure 11:
- HBQ PPL numbers are the same as Figure 10
- BQ PPL numbers are the same as Figure 6

Table 6: TODO: ptq.py does not have mixtral
- getting numbers: L2_scale_ablation.py

Table 7:
- psum_quant.py

Table 9:
- MXFP4/NVFP4/HBQ PPL numbers: main_eval.py

Table 10:
- MXFP/NVFP/HBQ PPL numbers: kv_eval.py TODO: debug
- Amove/MXFP+/MXFP++ PPL numbers: from original papers

Table 11:
- MXFP/NVFP/HBQ accuracy numbers: reasoning.py

Table 12: Todo, vllm

Todo
- fig 4a
- fig 8
- tab 7

Nuances didn't mention in the paper:
- When using FP8-scale under high activation precision (>= 6-bit), FP8-scale face similar scaling factor quantization error as the floor/rounding issue in PoT-scale. So we use use_ceil to calculate FP8 scaling when activation is >= 6-bit. See quantizer.py MXFPQuantizer.get_shared_scale() for how the scaling factor is calculated.
- For PoT-round, we found out that only qk projection benefits from rounding, floor is better for other layers. So we only use rounding in QK.