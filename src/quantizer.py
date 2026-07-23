"""
Extending Micro Scaling format by MicroSoft
Paper: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

This implementation also supports following features
- FP scaling factor (e.g. E4M3 for NVFP, E5M2 in AMXFP)
- Per-tensor scale (for NVFP)
- HBQ quantization
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Callable
from functools import partial

FP32_EXPONENT_BIAS = 127
FP16_EXPONENT_BIAS = 15
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
FP16_MIN_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 1)
FP32_MAX_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 2**10)
FP16_MAX_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 2**5)

@dataclass(frozen=True)
class FPQuantConfig:
    ebit: int
    mbit: int
    allow_subnormal: bool=True
    detailed_output: bool=False
    use_ceil: bool=False
    emax: int=None
    emin: int=None

    @property
    def key(self):
        return (self.ebit, self.mbit, self.allow_subnormal, self.detailed_output, self.use_ceil, self.emax, self.emin)

class FPQuantCompilerCache:
    def __init__(self):
        self._cache: Dict[tuple, Callable[[torch.Tensor], torch.Tensor]] = {}
    def get(self, cfg: FPQuantConfig) -> Callable[[torch.Tensor], torch.Tensor]:
        key = cfg.key
        fn = self._cache.get(key)
        if fn is not None:
            return fn

        # Bake constants as default args so compiler sees literals:
        impl = make_fp_quant(cfg.ebit, cfg.mbit, cfg.allow_subnormal, cfg.detailed_output, cfg.use_ceil, cfg.emax, cfg.emin)

        compiled = torch.compile(impl, fullgraph=True, dynamic=True)
        # (Optional) warmup to pay compile cost now:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ = compiled(torch.empty(4096, 256, 16, device=device, dtype=torch.float16))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            # Fallback gracefully if compile isn’t supported in your env
            print(f"[LPCompilerCache] compile failed, using eager: {e}")
            compiled = impl

        self._cache[key] = compiled
        return compiled

GLOBAL_FP_QUANT_CACHE = FPQuantCompilerCache()

def _reshape_to_blocks(A, axes, block_size):
    """
    Adopt the reshape function from microscaling
    https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py#L95
    """
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension to apply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    """
    Adopt the reshape function from microscaling
    https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py#L95
    """
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

def get_max_norm(ebit: int, mbit: int, unsigned: bool=False):
    """
    Get the maximum normal number of specific bit width
    if ebit = 0: integer case with INT(mbit)
    else: fp case with E(ebit)M(mbit)
    unsigned: only used for integer case,whether the integer is unsigned
    """
    if ebit == 0: # integer case
        return 2**(mbit) - 1 if unsigned else 2**(mbit-1)-1
    elif (ebit, mbit) == (4, 3):
        return 448 # in e4m3, max is 448 instead of 480 (1111 111 is reserved for special value)
    else:
        if ebit == 5:
            emax = 15
        else:
            emax = 2**(ebit-1)
        return 2**emax * float(2**(mbit+1) - 1) / 2**(mbit)

def get_min_norm(ebit: int, mbit:int):
    """
    Get the minimum normal number of specific bit width
    if ebit = 0: integer case with INT(mbit)
    else: fp case with E(ebit)M(mbit)
    """
    if ebit == 0: # integer case
        return 0
    elif ebit == 1:
        return 1
    else:
        emin = 2-2**(ebit-1)
        return 2**emin

def get_min_subnorm(ebit: int, mbit: int):
    """
    Get the minimum subnormal number of specific bit width
    """
    emin = 2-2**(ebit-1)
    return 2**emin * 2**(-mbit)
   
def make_fp_quant(ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False, emax: int=None, emin: int=None):
    if emax is None:
        if ebit == 5: # in FP16/E5M2 case, emax is 15
            emax = 15
        else:
            emax = 2**(ebit-1)
    if emin is None:
        emin = -(2**(ebit -1)) + 2

    subnormal_promote_rtn_range = 2**-(mbit+1)*2**emin
    mant_max = 2**mbit - 1
    exp_bias = 2**(ebit-1) - 1
    mant_shift = 2**mbit
    min_norm = get_min_norm(ebit, mbit)
    subnormal_threshold = min_norm-subnormal_promote_rtn_range
    def fp_quant(x:torch.Tensor):
        """
        Quantize input tensor to specific fp format
        """

        # zero mask
        x_is_zero = x == 0
        # promote subnormal to normal if it'll end up rounding up to min normal number
        promote_to_normal_mask = (x.abs() < min_norm) & (x.abs() >= subnormal_threshold)
        # subnormal mask
        x_is_subnormal = (x.abs() < subnormal_threshold) & (x != 0) if allow_subnormal else torch.zeros_like(x, dtype=torch.bool)
        # if don't allow subnormal, handle underflow cases where we set them to 0
        underflow_mask = (x.abs() < subnormal_threshold) & (x != 0) if not allow_subnormal else torch.zeros_like(x, dtype=torch.bool)

        # set exponent
        full_exp = torch.floor(torch.log2(torch.abs(x) + (x == 0).type(x.dtype)))
        exp = full_exp.clip(min = emin, max = emax)

        # calculate mantissa
        mant = torch.where(x_is_subnormal,
                            x / torch.exp2(exp) * mant_shift, # subnormal case, no implicit 1
                            (x / torch.exp2(exp) - x.sign()) * mant_shift) # subtract implicit 1 for normal case

        # here, the reason why we add 0.5004 instead of 0.5 is because torch.compile will fuse operation
        # but I want to keep the original behavior, for instance, 0.4997+0.5 will be rounded to 1.0 if not using kernel because of fp16
        # but with kernel, it will be rounded to 0 since kernel fused op together and will not round to 1.0
        # in order to keep the original behavior (un-compiled version), we add 0.5004 instead of 0.5
        mant_round = torch.abs(mant) + 0.5004
        mant_round = mant_round.to(x.dtype)
        mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.floor(mant_round) # use rounding as default (don't use +0.5 with floor for float16)
        # mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.round(torch.abs(mant)+1e-5) # use rounding as default, add 1e-5 to avoid 0.5 round to 0
        overflow_mask = (mant > mant_max) & (exp != emax)
        mant = mant.clip(max = mant_max)
        zero_mask_for_mant = x_is_zero | underflow_mask | overflow_mask | promote_to_normal_mask
        mant = torch.where(zero_mask_for_mant, 0, mant)
        exp = torch.where(overflow_mask, exp + 1, exp)

        # reconstruct quantized value
        signed_mant = x.sign() * mant
        x_q = torch.where(x_is_subnormal,
                        signed_mant / mant_shift * torch.exp2(exp),
                        (signed_mant / mant_shift + x.sign()) * torch.exp2(exp))
        zero_mask_for_x_q = x_is_zero | underflow_mask
        x_q = torch.where(zero_mask_for_x_q, 0, x_q)
        if (ebit, mbit) == (4, 3): # in e4m3, max is 448 instead of 480 (1111 111 is reserved for special value)
            x_q = x_q.clip(max = 448, min = -448)

        if detailed_output:
            exp = exp+exp_bias
            zero_mask_for_exp = x_is_zero | underflow_mask | x_is_subnormal
            exp = torch.where(zero_mask_for_exp, 0, exp)
            return x_q, exp, mant
        else:
            return x_q

    return fp_quant

def fp_quant(x:torch.Tensor, ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False, emax: int=None, emin: int=None):
    """
    Quantize high precision tensor (e.g. FP16) to specific FP format (e.g. FP4)
    params:
        ebit: target FP exponent bit
        mbit: target FP exponent bit
        allow_subnormal: allowing subnormal cases
        detailed_output: also output exponents and mantissas
        use_ceil: use ceilling function for rounding mantissa
        emax: specify exponent maximum value
        emin: specify exponent minimum value
    """
    if emax is None:
        if ebit == 5: # in FP16/E5M2 case, emax is 15
            emax = 15
        else:
            emax = 2**(ebit-1)
    if emin is None:
        emin = -(2**(ebit -1)) + 2

    # zero mask
    x_is_zero = x == 0
    # promote subnormal to normal if it'll end up rounding up to min normal number
    subnormal_promote_rtn_range = 2**-(mbit+1)*2**emin
    subnormal_threshold = get_min_norm(ebit, mbit)-subnormal_promote_rtn_range
    promote_to_normal_mask = (x.abs() < get_min_norm(ebit, mbit)) & (x.abs() >= subnormal_threshold)
    # subnormal mask
    x_is_subnormal = (x.abs() < subnormal_threshold) & (x != 0) if allow_subnormal else torch.zeros_like(x, dtype=torch.bool)
    # if don't allow subnormal, handle underflow cases where we set them to 0
    underflow_mask = (x.abs() < subnormal_threshold) & (x != 0) if not allow_subnormal else torch.zeros_like(x, dtype=torch.bool)

    # set exponent
    full_exp = torch.floor(torch.log2(torch.abs(x) + (x == 0).type(x.dtype)))
    # exp = torch.where(x_is_subnormal, torch.full_like(full_exp, emin), full_exp.clip(min = emin, max = emax))
    exp = full_exp.clip(min = emin, max = emax)
    # exp[x_is_zero | underflow_mask] = 0

    # calculate mantissa
    mant = torch.where(x_is_subnormal,
                        x / 2**exp * 2**mbit, # subnormal case, no implicit 1
                        (x / 2**exp - x.sign()) * 2**mbit) # subtract implicit 1 for normal case

    # mant = torch.floor(torch.abs(mant) + 0.55) if use_ceil else torch.floor(torch.abs(mant) + 0.5)
    mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.floor(torch.abs(mant) + 0.5) # use rounding as default (don't use +0.5 with floor for float16)
    # mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.round(torch.abs(mant)+1e-5) # use rounding as default, add 1e-5 to avoid 0.5 round to 0
    mant_max = 2**mbit - 1
    overflow_mask = (mant > mant_max) & (exp != emax) # round up to emax+1 and mant = 0
    mant = mant.clip(max = mant_max)
    zero_mask_for_mant = x_is_zero | underflow_mask | overflow_mask | promote_to_normal_mask
    mant = torch.where(zero_mask_for_mant, 0, mant)
    exp = torch.where(overflow_mask, exp + 1, exp)

    # reconstruct quantized value
    signed_mant = x.sign() * mant
    x_q = torch.where(x_is_subnormal,
                      signed_mant / 2**mbit * 2**exp,
                      (signed_mant / 2**mbit + x.sign()) * 2**exp)
    zero_mask_for_x_q = x_is_zero | underflow_mask
    x_q = torch.where(zero_mask_for_x_q, 0, x_q)
    if (ebit, mbit) == (4, 3): # in e4m3, max is 448 instead of 480 (1111 111 is reserved for special value)
        x_q = x_q.clip(max = 448, min = -448)

    if detailed_output:
        exp_bias = 2**(ebit-1) - 1
        exp = exp+exp_bias
        zero_mask_for_exp = x_is_zero | underflow_mask | x_is_subnormal
        exp = torch.where(zero_mask_for_exp, 0, exp)
        return x_q, exp, mant
    else:
        return x_q

class _QBase(nn.Module):
    """
        Base quantizer method for weight and activation.
    """
    def __init__(self):
        super().__init__()

    def q(self, x:torch.Tensor):
        """
        Quantization operation
        """
        return x
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.q(x)
        return y

class MXINTQuantizer(_QBase):
    def __init__(
        self, 
        nbit: int, 
        sc_ebit: int=8, 
        sc_mbit: int=0, 
        block_size:int=32, 
        per_tensor_scale: bool = False, 
        scale_allow_subnormal: bool = False,
        use_ceil: bool = False
    ):
        """
        MXINT quantization
        Parameters:
        - nbit: element bit width (conventional MX only support INT4 & INT8)
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        """
        super().__init__()
        self.nbit = nbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.shared_exp = None
        self.use_ceil = use_ceil
        
    def reshape(self, x:torch.Tensor):
        # reshape to blocks along the last dimension
        x, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, [-1], block_size=self.block_size
        )
        return x, axes, orig_shape, padded_shape
   
    def get_shared_scale(self, x:torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            min_normal = FP32_MIN_NORMAL
        elif x.dtype == torch.float16:
            min_normal = FP16_MIN_NORMAL
        else:
            raise NotImplementedError(f"Non-supported data type: {x.dtype}")
        
        max_val, _ = torch.max(torch.abs(x), dim=-1, keepdim=True) # abs max
        ele_emax =  self.nbit-2 # maximum exponent in the element format
        if self.sc_mbit == 0: # conventional MX, shift scaling
            shared_exp = torch.floor(torch.log2(
                max_val + min_normal * (max_val == 0).type(max_val.dtype)
            ))
            # emax and emin depend on your data formats
            # here we use FP16 for scaling factor tensor in fake quant, so the range for scaling factor will be 2**-14 - 2**15
            # this clipping matters when all elements are very small (or even all 0s in a block)
            # making sure scaling factor stays in FP16's range
            emax = 15
            emin = -14
            shared_exp = (shared_exp-ele_emax).clip(max=emax, min=emin)
            shared_scale = 2**shared_exp
            self.shared_exp = shared_exp
        else: # FP scale
            shared_scale = max_val / get_max_norm(ebit=0, mbit=self.nbit)
            shared_scale = fp_quant(shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, use_ceil=self.use_ceil)
            scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
            shared_scale = shared_scale.clip(min = scale_min) # avoid very small case
        return shared_scale

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)

        self.shared_scale = self.get_shared_scale(xg)
        xg = xg / self.shared_scale

        xg = torch.sign(xg) *  torch.floor(torch.abs(xg) + 0.5)
        xg = xg.clip(min = -(2**(self.nbit-1)), max = 2**(self.nbit-1)-1)
        
        xg = xg.mul(self.shared_scale).to(x.dtype)

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPQuantizer(_QBase):
    def __init__(
        self, 
        block_size:int=32, 
        ebit:int=2, 
        mbit:int=3, 
        sc_ebit:int=0, 
        sc_mbit:int=8, 
        per_tensor_scale: bool = False, 
        scale_allow_subnormal: bool = False, 
        use_round: bool=False, 
        use_ceil: bool=False,
        activate_dynamo: bool=False
    ):
        """
        MXFP quantization
        Parameters:
            - block_size: quantization granularity
            - ebit: element exponent bit width
            - mbit: element mantissa bit width
            - sc_ebit: shared scale exponent bit width
            - sc_mbit: shared scale mantissa bit width
            - per_tensor_scale: per tensor scale (used in official NVFP4)
            - scale_allow_subnormal: allow using subnormal in scaling factor
            - use_round: use rounding for MX scaling factor calculation
            - use_ceil: use ceiling for scaling factor calculation for FP8-scale
            - activate_dynamo: use dynamo to compile FP quantization for speedup
        """
        super().__init__()
        self.ebit = ebit
        self.mbit = mbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
        self.use_ceil = use_ceil
        lp_config = FPQuantConfig(ebit=self.ebit, mbit=self.mbit, allow_subnormal=True, detailed_output=False)
        if activate_dynamo:
            self.lpfp_quant = GLOBAL_FP_QUANT_CACHE.get(lp_config)
        else:
            self.lpfp_quant = make_fp_quant(lp_config.ebit, lp_config.mbit, lp_config.allow_subnormal, lp_config.detailed_output, lp_config.use_ceil, lp_config.emax, lp_config.emin)
        
    def reshape(self, x:torch.Tensor):
        # reshape to blocks along the last dimension
        x, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, [-1], block_size=self.block_size
        )
        return x, axes, orig_shape, padded_shape
   
    def get_shared_scale(self, x:torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            min_normal = FP32_MIN_NORMAL
        elif x.dtype == torch.float16:
            min_normal = FP16_MIN_NORMAL
        else:
            raise NotImplementedError(f"Non-supported data type: {x.dtype}")
        
        max_val, _ = torch.max(torch.abs(x), dim=-1, keepdim=True) # abs max
        ele_emax = 2**(self.ebit-1) # maximum exponent in the element format
        if self.sc_mbit == 0: # conventional MXFP, shift scaling
            if self.use_round:
                shared_exp = torch.round(torch.log2(
                    max_val + min_normal * (max_val == 0).type(max_val.dtype)
                    ))
            else:
                shared_exp = torch.floor(torch.log2(
                    max_val + min_normal * (max_val == 0).type(max_val.dtype)
                    ))
            emax = 2**(self.sc_ebit-1) - 1
            emin = 2-2**(self.sc_ebit-1)
            shared_exp = (shared_exp - ele_emax).clip(min=emin, max=emax)
            shared_scale = 2**shared_exp
        else: # NVFP: FP scale
            shared_scale = max_val / get_max_norm(ebit=self.ebit, mbit=self.mbit)
            # in case share scale can't be quantized to the range of [min_subnorm, max_norm], could happend to E < 8 cases
            scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
            shared_scale = shared_scale.clip(min = scale_min, max = get_max_norm(self.sc_ebit, self.sc_mbit))
            shared_scale = fp_quant(shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, use_ceil=self.use_ceil)

        return shared_scale

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)

        if self.per_tensor_scale: # scale xg down to match (element range * scale range), used in official NVFP4
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item()
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale

        shared_scale = self.get_shared_scale(xg)

        xg = xg / shared_scale

        xg = self.lpfp_quant(xg)

        xg = xg.mul(shared_scale)

        if self.per_tensor_scale:
            xg = xg * tensor_scale

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class HBQQuantizer(MXFPQuantizer):
    """
    HBQ quantizer, 2-stage quantization
    """
    L2_SCHEME = ["PoT", "INT", "SIG-1", "FP-1", "FP1", "FP-10", "FP-100", "Mix"]
    def __init__(
        self,
        block_size:int=128,
        ebit:int=2,
        mbit:int=3,
        sc_ebit:int=5,
        sc_mbit:int=3,
        l2_block_size:int=4,
        l2_sc_bit:int=1,
        l2_scheme:str="PoT",
        keep_l2_scale: bool=False,
        per_tensor_scale: bool = False,
        scale_allow_subnormal: bool = False,
        use_round: bool=False,
        use_ceil: bool=False,
        use_quant_kernel: bool=True
    ):
        super().__init__(block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil, use_quant_kernel)
        assert block_size % l2_block_size == 0, "block_size must be divisible by l2_block_size"
        assert l2_scheme in self.L2_SCHEME, f"Invalid L2 scheme: {l2_scheme}"
        self.keep_l2_scale = keep_l2_scale # keep l2 scale statistics, only activate this if you want to extract l2 scale statistics (e.g. for hardware verification)
        self.l2_block_size = l2_block_size  # block size for L2 quantization
        self.l2_sc_bit = l2_sc_bit  # shift bit for L2 quantization
        self.l2_scheme = l2_scheme  # scheme for L2 quantization
        self.win_rate = []
    
    def quant_scale(self, scale:torch.Tensor):
        scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
        scale = scale.clip(min = scale_min, max = get_max_norm(self.sc_ebit, self.sc_mbit))
        scale = fp_quant(scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, use_ceil=self.use_ceil)
        return scale

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        l2_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        l2_scale_pot = torch.pow(2, l2_scale_range)
        l2_scale_int = (l2_scale_range+1)
        l2_scale_sig_1 = 1+l2_scale_range/2 # 1, 1.5, 2, 2.5
        l2_scale_fp = 1+l2_scale_range/(2**self.l2_sc_bit) # 1.XX
        l2_scale_fp1 = 1+(l2_scale_range+1)/(2**self.l2_sc_bit) # 1.XX
        l2_scale_fp_10 = 1+l2_scale_range/(2**(self.l2_sc_bit+1)) # 1.0X
        l2_scale_fp_100 = 1+l2_scale_range/(2**(self.l2_sc_bit+2)) # 1.00X
        if self.l2_scheme == "PoT":
            l2_scale_all = l2_scale_pot[..., None]
        elif self.l2_scheme == "INT":
            l2_scale_all = l2_scale_int[..., None]
        elif self.l2_scheme == "SIG-1": # SIG_1
            l2_scale_all = l2_scale_sig_1[..., None]
        elif self.l2_scheme == "FP-1": # SIG_2
            l2_scale_all = l2_scale_fp[..., None]
        elif self.l2_scheme == "FP1":
            l2_scale_all = l2_scale_fp1[..., None]
        elif self.l2_scheme == "FP-10":  # SIG_3
            l2_scale_all = l2_scale_fp_10[..., None]
        elif self.l2_scheme == "FP-100":  # SIG_4
            l2_scale_all = l2_scale_fp_100[..., None]
        elif self.l2_scheme == "Mix": # Mixture of SIG_2 and SIG_3
            l2_scale_all = torch.stack([l2_scale_fp, l2_scale_fp_10], dim=-1)
        else:
            assert False, f"Invalid L2 scheme: {self.l2_scheme}"

        num_of_scale_scheme = l2_scale_all.shape[-1]
        num_of_l2_scales = 2**self.l2_sc_bit

        ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
        ele_max = torch.full((num_of_scale_scheme,), ele_max, device=xg.device, dtype=xg.dtype)
        ele_max = ele_max * l2_scale_all.max(dim=-2).values
        l1_scale = xg.abs().max(dim=-1, keepdim=True).values.expand(*xg.shape[:-1], num_of_scale_scheme) / ele_max # [out_ch, num_blocks, num_of_scale_scheme]
        l1_scale = self.quant_scale(l1_scale)

        l1_scaled_xg = xg.unsqueeze(-1).expand(*xg.shape, num_of_scale_scheme) / l1_scale[..., None, :] # [out_ch, num_blocks, block_size, num_of_scale_scheme]
        # reshape sub blocks
        l1_scaled_xg = l1_scaled_xg.view(*l1_scaled_xg.shape[:-2], -1, self.l2_block_size, 1, num_of_scale_scheme) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, 1, num_of_scale_scheme]
        # expand for all l2 scales
        l1_scaled_xg = l1_scaled_xg.expand(*l1_scaled_xg.shape[:-2], num_of_l2_scales, l1_scaled_xg.shape[-1]) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
        l2_scale_all_expand = l2_scale_all
        while len(l2_scale_all_expand.shape) < len(l1_scaled_xg.shape): l2_scale_all_expand = l2_scale_all_expand.unsqueeze(0)
        l2_scale_all_expand = l2_scale_all_expand.expand_as(l1_scaled_xg)
        l2_scaled_xg = l1_scaled_xg / l2_scale_all_expand
        l2_scaled_xg_quant = self.lpfp_quant(l2_scaled_xg)
        l2_scaled_xg_quant = l2_scaled_xg_quant * l2_scale_all_expand
        del l1_scaled_xg, l2_scale_all

        l1_scale_expand = l1_scale
        while len(l1_scale_expand.shape) < len(l2_scaled_xg_quant.shape): l1_scale_expand = l1_scale_expand.unsqueeze(-2)
        l2_scaled_xg_dequant = l2_scaled_xg_quant * l1_scale_expand # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
        
        # determine the best l2 scale for each sub block in each scheme
        xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size]
        xgg_expand = xgg[..., None, None].expand_as(l2_scaled_xg_quant) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
        # error in each sub block
        error_in_each_sub_block = (xgg_expand - l2_scaled_xg_dequant).abs().pow(2).sum(dim=-3, keepdim=True) # [out_ch, num_blocks, block_size // l2_block_size, 1, num_of_l2_scales, num_of_scale_scheme]
        _, min_error_idx_l2_scale = error_in_each_sub_block.min(dim=-2, keepdim=True) # [out_ch, num_blocks, block_size // l2_block_size, 1, 1, num_of_scale_scheme]
        del error_in_each_sub_block
        # l2_scale_all_in_each_scheme = l2_scale_all
        # while len(l2_scale_all_in_each_scheme.shape) < len(min_error_idx_l2_scale.shape): l2_scale_all_in_each_scheme = l2_scale_all_in_each_scheme.unsqueeze(0) # [1, 1, 1, 1, num_of_l2_scales, num_of_scale_scheme]
        # l2_scale_all_in_each_scheme = l2_scale_all_in_each_scheme.expand(*min_error_idx_l2_scale.shape[:-2], num_of_l2_scales, num_of_scale_scheme) # [out_ch, num_blocks, block_size // l2_block_size, 1, num_of_l2_scales, num_of_scale_scheme]
        # l2_scale_all_in_each_scheme = l2_scale_all_in_each_scheme.gather(-2, min_error_idx_l2_scale).squeeze(dim=(-2, -3)) # [out_ch, num_blocks, block_size // l2_block_size, num_of_scale_scheme]

        # determine the best scheme for each block
        min_error_idx_l2_scale_expand = min_error_idx_l2_scale.expand(*min_error_idx_l2_scale.shape[:-3], self.l2_block_size, *min_error_idx_l2_scale.shape[-2:])  # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, 1, num_of_scale_scheme]
        l2_scaled_xg_dequant_each_scheme = l2_scaled_xg_dequant.gather(-2, min_error_idx_l2_scale_expand).squeeze(-2) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_scale_scheme]
        xgg_expand_scheme = xgg[..., None].expand_as(l2_scaled_xg_dequant_each_scheme) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_scale_scheme]
        error_in_each_scheme = (xgg_expand_scheme - l2_scaled_xg_dequant_each_scheme).abs().pow(2).sum(dim=(-2, -3), keepdim=True) # [out_ch, num_blocks, 1, 1, num_of_scale_scheme]
        _, min_error_idx_scheme = error_in_each_scheme.min(dim=-1, keepdim=True) # [out_ch, num_blocks, 1, 1, 1]
        self.blk_l2_scheme = min_error_idx_scheme.squeeze(dim=(-1, -2, -3)) # (out_ch, num_blocks)
        self.shared_scale = l1_scale.gather(-1, self.blk_l2_scheme[..., None]).squeeze(dim=-1) # [out_ch, num_blocks]
        # min_error_idx_l2_scale [out_ch, num_blocks, block_size // l2_block_size, 1, 1, num_of_scale_scheme]
        l2_scheme_expand = self.blk_l2_scheme
        while len(l2_scheme_expand.shape) < len(min_error_idx_l2_scale.shape): l2_scheme_expand = l2_scheme_expand.unsqueeze(-1)
        l2_scheme_expand = l2_scheme_expand.expand(*min_error_idx_l2_scale.shape[:-1], -1) # [out_ch, num_blocks, block_size // l2_block_size, 1, 1]
        self.l2_scale = min_error_idx_l2_scale.gather(-1, l2_scheme_expand).squeeze(dim=(-1, -2, -3)) # [out_ch, num_blocks, block_size // l2_block_size]
        del l2_scaled_xg_dequant_each_scheme, xgg_expand_scheme, error_in_each_scheme, min_error_idx_scheme

        # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
        l2_scheme_expand = self.blk_l2_scheme
        while len(l2_scheme_expand.shape) < len(l2_scaled_xg_dequant.shape): l2_scheme_expand = l2_scheme_expand.unsqueeze(-1)
        l2_scheme_expand = l2_scheme_expand.expand(*l2_scaled_xg_dequant.shape[:-1], -1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, 1]
        xg_dequant = l2_scaled_xg_dequant.gather(-1, l2_scheme_expand).squeeze(dim=-1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales]
        l2_scale_expand = self.l2_scale
        while len(l2_scale_expand.shape) < len(xg_dequant.shape): l2_scale_expand = l2_scale_expand.unsqueeze(-1)
        l2_scale_expand = l2_scale_expand.expand(*xg_dequant.shape[:-1], -1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, 1]
        xg_dequant = xg_dequant.gather(-1, l2_scale_expand).squeeze(dim=-1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size]
        xg_dequant = xg_dequant.view(*xg.shape)
        xg_dequant = _undo_reshape_to_blocks(xg_dequant, padded_shape, orig_shape, axes)
        del l2_scheme_expand

        # verification
        # breakpoint()
        # l2_scale_all_expand [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
        if self.keep_l2_scale:
            select_scheme = self.blk_l2_scheme
            while len(select_scheme.shape) < len(l2_scale_all_expand.shape): select_scheme = select_scheme.unsqueeze(-1)
            select_scheme = select_scheme.expand(*l2_scale_all_expand.shape[:-1], -1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales, num_of_scale_scheme]
            real_l2_scale = l2_scale_all_expand.gather(-1, select_scheme).squeeze(dim=-1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, num_of_l2_scales]
            select_l2_scale = self.l2_scale
            while len(select_l2_scale.shape) < len(real_l2_scale.shape): select_l2_scale = select_l2_scale.unsqueeze(-1)
            select_l2_scale = select_l2_scale.expand(*real_l2_scale.shape[:-1], -1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size, 1]
            real_l2_scale = real_l2_scale.gather(-1, select_l2_scale).squeeze(dim=-1) # [out_ch, num_blocks, block_size // l2_block_size, l2_block_size]
            self.real_l2_scale = real_l2_scale[..., 0] # [out_ch, num_blocks, block_size // l2_block_size]
            # scale_product = self.shared_scale[..., None, None] * real_l2_scale
            # scale_product = scale_product.view(*xg.shape)
            # xg_dequant_verification = (self.lpfp_quant(xg / scale_product) * scale_product).view(*x.shape)

            # scale_product = self.shared_scale[..., None] * self.l2_scale
        else:
            del self.l2_scale, self.blk_l2_scheme, self.shared_scale # this could cause OOM for activations

        return xg_dequant

