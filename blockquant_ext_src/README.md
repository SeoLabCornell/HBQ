# blockquant_ext (HBQ local extension)

CUDA GEMM extension used by HBQ accumulation modes in `src/ops.py`.

## Build

With `hbq` activated, build from repo root.

From repo root:

```bash
conda activate hbq
cd blockquant_ext_src
pip install ninja setuptools wheel
pip install -e . --no-build-isolation
```

## Python API

```python
import torch
import blockquant_ext as ext

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(K, N, dtype=torch.float16, device="cuda")

C_fp32 = ext.gemm_fp32_accum(A, B)
C_fp16 = ext.gemm_fp16_accum(A, B)
C_grouped = ext.gemm_fp16_grouped_accum(A, B, group_k=64)
C_mxfp8 = ext.gemm_blockquant_mxfp8(A, B, group_k=64, group_n=32)
C_mxint8 = ext.gemm_blockquant_mxint8(A, B, group_k=64, group_n=32)
```

## Constraints

- CUDA only.
- Inputs must be `torch.float16` 2D tensors.
- For block-quant kernels: `K % group_k == 0` and `N % group_n == 0`.
- Accumulation styles:
  - `gemm_fp32_accum`: FP32 accumulation.
  - Grouped/BlockQuant kernels: FP32 within each `GROUP_K`, FP16 accumulation across groups.
