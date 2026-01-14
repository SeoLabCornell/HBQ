import torch
import torch.nn as nn
import numpy as np

FP32_EXPONENT_BIAS = 127
FP16_EXPONENT_BIAS = 15
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
FP16_MIN_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 1)
FP16_MAX_NORMAL = 2 ** (FP16_EXPONENT_BIAS - 1)

try: 
    import t2c_gemm
    INTMM = True
except:
    print("Torch-gemm is not installed!")
    INTMM = False
try:
    import blockquant_ext as ext
except:
    print("BlockQuant is not installed!")

class BatchIntMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit
    
    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)
        y = y.to(torch.int8)

        z = t2c_gemm.bmm_int8(x, y, 1.0)
        return z

class BatchHeadIntMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit
    
    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)
        y = y.to(torch.int8)

        z = t2c_gemm.bcmm_int8(x, y, 1.0)
        return z

class IntActWeight(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, 1, 1, dtype=torch.float32))
        self.nbit = nbit

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)

        z = t2c_gemm.bmw_int8(x, y, self.scale)
        return z.to(torch.float16)

class FP16AccumMatMul(nn.Module):
    """
    FP16 accumulation matrix multiplication
    Args:
    """
    def __init__(self, block_size:int=None):
        super().__init__()
        self.block_size = block_size

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

        if self.block_size is not None: # no group, just use FP16 accumulation
            out2d = ext.gemm_fp16_accum(x2d, w_t)
        elif self.block_size in [4, 8, 16, 32, 64, 128]: # grouped accumulation
            out2d = ext.gemm_fp16_grouped_accum(x2d, w_t, self.block_size)
        else:
            raise ValueError(f"Invalid block size: {self.block_size}")

        # Bias add (bias is already on correct device/dtype via tied parameter)
        if bias is not None:
            out2d = out2d + bias
        
        return out2d.view(*orig_shape[:-1], -1)

class FloatMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()

        z = torch.matmul(x, y)
        return z

class Matmul_fp_pre(nn.Module):
    def __init__(self, g_num:int=16, a_bit:int=8, w_bit:int=8):
        super().__init__()
        self.g_num = g_num
        self.a_bit = a_bit
        self.w_bit = w_bit

    def pre_align(self, x:torch.Tensor, r_bit):
        # scale x
        x_exp = torch.floor(
            torch.log2( x.abs() + FP16_MIN_NORMAL * (x==0).type(x.dtype))
        )
        qx = x / 2**(x_exp)
        # qx should be in [1,2] ? 
        qx = qx * 2 ** (r_bit-2)
        qx = torch.clamp(qx,  max = 2**(r_bit-1)-1, min = -(2**(r_bit-1)))
        qx = torch.sign(qx) * torch.floor(torch.abs(qx) + float(0.5))
        qx = qx / 2 ** (r_bit-2)
        qx = qx * 2**(x_exp)
        return qx
    
    def trunc_fp(self, x:torch.Tensor, r_bit:int = 8):
        # assume x is in fp16
        x_int = x.view(torch.int16)
        # Mask to keep only the top 6 bits of the 10-bit mantissa
        mant_rbit = r_bit-2 # subtract 1b sign, 1b hidden 1
        mantissa_mask = 2**10 - 2**(10-mant_rbit)
        mask = mantissa_mask | 0xFC00
        # mantissa_mask = 0b1111110000  # Keep 6 MSB, zero out 4 LSB
        return  torch.bitwise_and(x_int, mask).view(torch.float16)  # Preserve sign and exponent
    
    def forward(self, x:torch.Tensor, y:torch.Tensor, b:torch.Tensor)-> torch.Tensor:
        # expand x 
        
        x_dup = x.unsqueeze(2).expand(1,x.shape[1],y.shape[-2],x.shape[-1])
        x_dup = x_dup.reshape(*x_dup.shape[:-1],x_dup.shape[-1]//self.g_num, self.g_num)
        
        # yq = yq.unsqueeze(1)
        # import pdb; pdb.set_trace()

        # truncate w
        yq = y.reshape(*y.shape[:-1], y.shape[-1]//self.g_num, self.g_num)
        yq = self.trunc_fp(yq, self.w_bit)
        # yq = yq.reshape(*yq.shape[:-2], y.shape[-1])

        # slicing input channel
        prod_list = []
        x_dup = torch.split(x_dup, 8, dim=1)
        for i, slice in enumerate(x_dup):
            # import pdb; pdb.set_trace()
            prod = slice * yq
            prod_max = torch.amax(prod.abs(), dim=-1, keepdim=True)
            prod_exp = torch.log2(prod.abs() + FP16_MIN_NORMAL * (prod==0).type(prod.dtype))
            prod_exp = torch.log2(prod_max + FP16_MIN_NORMAL * (prod_max ==0).type(prod_max.dtype)) - prod_exp
            prod_exp = torch.floor(prod_exp)
            shift_bit = self.a_bit - prod_exp
            shift_bit = torch.clamp(shift_bit, min=0)

            # bin count shift_bit
            # tmp = shift_bit.int().flatten()
            # tensor_np = tmp.cpu().numpy()
            # values, counts = np.unique(tensor_np, return_counts=True)

            # pre align act
            slice = self.pre_align(slice, shift_bit)
            # slice = slice.reshape(*slice.shape[:-2],x.shape[-1])
            
            # multiply x&y
            prod = torch.sum(slice * yq, dim=(-2,-1))
            # if prod.isnan().sum() > 0 or prod.isinf().sum()>0:
            #     inf_indices = torch.nonzero(prod.isinf(), as_tuple = False)
            #     nan_indices = torch.nonzero(prod.isnan(), as_tuple = False)
            #     import pdb; pdb.set_trace()
            prod_list.append(prod)

        prod = torch.cat(prod_list, dim=1)
        prod.reshape(x.shape[0], x.shape[1], y.shape[0])

        # for reference
        # import pdb; pdb.set_trace()
        

        if b == None:
            # prod_origin = torch.matmul(x,y.T) 
            return prod 
        else:
            # prod_origin = torch.matmul(x,y.T) + b
            return prod + b