import torch
import torch.nn as nn
import numpy as np

FP32_EXPONENT_BIAS = 127
FP16_EXPONENT_BIAS = 15
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
FP16_MIN_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 1)
FP16_MAX_NORMAL = 2 ** (FP16_EXPONENT_BIAS - 1)

try:
    import blockquant_ext as ext
except:
    print("BlockQuant is not installed!")
    
class FP16AccumMatMul(nn.Module):
    """
    FP16 accumulation matrix multiplication
    Args:
        group_size: int, the group size for grouped accumulation
        quant_block_size: int, the block size for mxint8 psum quantization (set to None for no quantization)
    """
    def __init__(self, group_size:int=None, quant_block_size:int=None):
        super().__init__()
        self.group_size = group_size
        self.quant_block_size = quant_block_size

    def forward(self, x, w, bias=None) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.view(-1, x.shape[-1])

        # Ensure dtype/device: our kernel expects half on CUDA
        if x2d.dtype != torch.float16:
            x2d = x2d.half()
        x2d = x2d.to(w.device, non_blocking=True)

        # weight^T contiguous half on the same device
        w_t = w.t().contiguous()
        if w_t.dtype != torch.float16:
            w_t = w_t.half()

        if self.group_size is not None: # no group, just use FP16 accumulation
            out2d = ext.gemm_fp16_accum(x2d, w_t)
        elif self.quant_block_size is None:
            if self.group_size not in [4, 8, 16, 32, 64, 128]: 
                raise ValueError(f"Invalid group size: {self.group_size}")
            out2d = ext.gemm_fp16_grouped_accum(x2d, w_t, self.group_size)
        else: 
            if self.group_size not in [4, 8, 16, 32, 64, 128]: 
                raise ValueError(f"Invalid group size: {self.group_size}")
            if self.quant_block_size not in [4, 8, 16, 32]: 
                raise ValueError(f"Invalid quant block size: {self.quant_block_size}")
            out2d = ext.gemm_blockquant_mxint8_cuda(x2d, w_t, self.group_size, self.quant_block_size)

        # Bias add (bias is already on correct device/dtype via tied parameter)
        if bias is not None:
            out2d = out2d + bias
        
        return out2d.view(*orig_shape[:-1], -1)
