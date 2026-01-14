"""
Extending Micro Scaling format by MicroSoft
Paper: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

This implementation also supports following features
- FP scaling factor (e.g. E4M3 for NVFP, E5M2 in AMXFP)
- per-tensor scale (for NVFP)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from awq.quantize.base import _QBase, round_ste
from awq.quantize.observer import BaseObserver
# from src.utils.statistics import qsnr, plot_log2_hist
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
        impl = make_quant_element(cfg.ebit, cfg.mbit, cfg.allow_subnormal, cfg.detailed_output, cfg.use_ceil, cfg.emax, cfg.emin)

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
    Adopte the reshape function from microscaling
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

def quant_element_org(x:torch.Tensor, ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False):
    """
    Quantize input tensor to specific fp format
    """
    if ebit == 5: # in FP16/E5M2 case, emax is 15
        emax = 15
    else:
        emax = 2**(ebit-1)
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

    mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.floor(torch.abs(mant) + 0.5) # use rounding as default
    mant_max = 2**mbit - 1
    overflow_mask = (mant > mant_max) & (exp != emax)
    mant = mant.clip(max = mant_max)
    mant[x_is_zero | underflow_mask | overflow_mask] = 0
    mant[promote_to_normal_mask] = 0 # have to handle this case, since x / 2**exp - x.sign() will underflow
    exp[overflow_mask] = exp[overflow_mask] + 1

    # reconstruct quantized value
    signed_mant = x.sign() * mant
    x_q = torch.where(x_is_subnormal,
                      signed_mant / 2**mbit * 2**exp,
                      (signed_mant / 2**mbit + x.sign()) * 2**exp)
    x_q[x_is_zero | underflow_mask] = 0

    if detailed_output:
        exp_bias = 2**(ebit-1) - 1
        exp = exp+exp_bias
        exp[x_is_zero | underflow_mask | x_is_subnormal] = 0
        return x_q, exp, mant
    else:
        return x_q
        
def make_quant_element(ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False, emax: int=None, emin: int=None):
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
    def quant_element(x:torch.Tensor):
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

    return quant_element

def quant_element(x:torch.Tensor, ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False, emax: int=None, emin: int=None):
    """
    Quantize input tensor to specific fp format
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

def quant_element_int(x:torch.Tensor, nbit: int, upscale: float=1.0, unsigned: bool=False):
    f"""
        Quantized range is [0, 2**(nbit-1) - 1]
        upscale: scale the input for finer granularity
            For example, upscale = 2 nbit = 3, then the range becomes {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5} instead of 0-7
        non_zero: If True, it'll not include 0 (for instance 3 bit become 1-8)
    """

    if unsigned:
        ub = 2**(nbit) - 1
    else:
        ub = 2**(nbit-1) - 1
        x_q = x.abs()
    lb = 0
    x_q = x_q * upscale
    x_q = torch.floor(x_q+0.5)
    x_q = x_q.clip(min = lb, max = ub)
    return x_q / upscale if unsigned else x.sign() * x_q / upscale

def quant_element_int_non_zero(x:torch.Tensor, nbit: int):
    f"""
        Quantized range is ±[1, 2**(nbit-1)]
    """
    ub = 2**(nbit-1)
    lb = 1
    x_round = torch.floor(x.abs()+0.5)
    x_q = x_round.clip(min = lb, max = ub)
    return x_q*x.sign()

def reciprocal_lut_div(x, exp, mant, mbit, lut_bit):
    """
    Use reciprocal LUT to calculate division with lut_bit LUT
    LUT has 2**mbit entries(depend on mantissa bit)
    equivalent to x/(1.M*2^E) when lut_bit is 10 for fp16 input
    """
    lut_entries = 2**mbit
    lut_mant = quant_element(2/(1+mant/lut_entries), 2, lut_bit, allow_subnormal=True, detailed_output=False)
    return x*lut_mant*2**(-exp-1)

class MXINTObserver(BaseObserver):
    def __init__(self, nbit: int, sc_ebit: int=8, sc_mbit: int=0, unsigned: bool = False, scale_allow_subnormal: bool = False, use_round: bool=False):
        """
        Parameters:
        - nbit: element bit width (conventional MX only support INT4 & INT8)
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        - use_round: Whether to use rounding for scale (only in MX)
        """
        super().__init__(nbit, unsigned)
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
    def get_bound(self, x:torch.Tensor):
        max_norm = float(2**(self.nbit-1) - 1) / 2**(self.nbit-2)

        # update bound
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)
    
    def get_shared_scale(self, x:torch.Tensor, axis=1) -> torch.Tensor:
        if x.dtype == torch.float32:
            min_normal = FP32_MIN_NORMAL
        elif x.dtype == torch.float16:
            min_normal = FP16_MIN_NORMAL
        else:
            raise NotImplementedError(f"Non-supported data type: {x.dtype}")
        
        ele_max, _ = torch.max(torch.abs(x), dim=axis, keepdim=True) # abs max
        if self.sc_mbit == 0: # conventional MX, shift scaling
            if self.use_round:
                # shared_exp = torch.round(torch.log2(
                #     ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                #     +1e-5)) # add 1e-5 to avoid 0.5 round to 0
                shared_exp = torch.floor(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    )+0.5)
            else:
                shared_exp = torch.floor(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    ))
            emax = 2**(self.sc_ebit-1) - 1
            emin = -emax
            shared_exp[shared_exp > emax] = float("NaN")
            shared_exp[shared_exp < emin] = emin
            shared_scale = 2**shared_exp
        else: # FP scale
            shared_scale = ele_max / get_max_norm(ebit=0, mbit=self.nbit)
            shared_scale = quant_element(shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal)
            scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
            shared_scale = shared_scale.clip(min = scale_min) # avoid very small case
        if (shared_scale == 0).sum() > 0:
            indices = (shared_scale == 0).nonzero()
            breakpoint()
        return shared_scale

    def calculate_qparam(self, x:torch.Tensor, axis:int=2):
        if self.sc_mbit == 0: # conventional MX, shift scaling
            scale = torch.tensor(2**(self.nbit-2), device=x.device) # implicit scale is handled here
        else:
            scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_scale = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale
    
    def forward(self, x:torch.Tensor, axis:int=2):
        self.get_bound(x)
        scale, zero_point, shared_scale = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_scale

class MXINTAsymObserver(BaseObserver):
    def __init__(self, nbit: int, sc_ebit: int=8, sc_mbit: int=0, unsigned: bool = True, scale_allow_subnormal: bool = True, use_round: bool=False):
        super().__init__(nbit, unsigned)
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
    
    def get_bound(self, x:torch.Tensor):
        min_val = x.min(dim=-1, keepdim=True).values
        max_val = x.max(dim=-1, keepdim=True).values

        if self.initialize:    
            self.lb.data = min_val
            self.ub.data = max_val
            
            self.initialize = False
        else:
            lb = torch.min(self.lb, min_val)
            ub = torch.max(self.ub, max_val)

            # update bound
            self.lb.copy_(lb)
            self.ub.copy_(ub)

    def get_shared_scale(self, x:torch.Tensor, axis=1) -> torch.Tensor:
        if x.dtype == torch.float32:
            min_normal = FP32_MIN_NORMAL
        elif x.dtype == torch.float16:
            min_normal = FP16_MIN_NORMAL
        else:
            raise NotImplementedError(f"Non-supported data type: {x.dtype}")
        
        ele_max, _ = torch.max(x, dim=axis, keepdim=True) # max
        ele_min, _ = torch.min(x, dim=axis, keepdim=True) # min
        ele_max_shifted = ele_max + torch.abs(ele_min) # shift all value to positive
        if self.sc_mbit == 0: # conventional MX, shift scaling
            raise NotImplementedError("Conventional MX is not supported for Asymmetric MXINT")
            if self.use_round:
                # shared_exp = torch.round(torch.log2(
                #     ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                #     +1e-5)) # add 1e-5 to avoid 0.5 round to 0
                shared_exp = torch.floor(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    )+0.5)
            else:
                shared_exp = torch.floor(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    ))
            emax = 2**(self.sc_ebit-1) - 1
            emin = -emax
            shared_exp[shared_exp > emax] = float("NaN")
            shared_exp[shared_exp < emin] = emin
            shared_scale = 2**shared_exp
        else: # FP scale
            shared_scale = ele_max_shifted / get_max_norm(ebit=0, mbit=self.nbit, unsigned=True)
            shared_scale[shared_scale == 0] = 1 # avoid all zero case
            zero_point = torch.abs(ele_min) / shared_scale
            shared_scale = quant_element(shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal)
            scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
            shared_scale = shared_scale.clip(min = scale_min) # avoid very small case
        if (shared_scale == 0).sum() > 0:
            indices = (shared_scale == 0).nonzero()
            breakpoint()
        return shared_scale, zero_point

    def calculate_qparam(self, x:torch.Tensor, axis:int=2):
        # zero_point = self.lb.to(x.dtype).to(x.device)
        if self.sc_mbit == 0: # conventional MX, shift scaling
            raise NotImplementedError("Conventional MX is not supported for Asymmetric MXINT")
            scale = torch.tensor(2**(self.nbit-2), device=x.device) # implicit scale is handled here
        else:
            scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        shared_scale, zero_point = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale
    
    def forward(self, x:torch.Tensor, axis:int=2):
        # self.get_bound(x)
        scale, zero_point, shared_scale = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_scale

class MXFPObserver(BaseObserver):
    def __init__(self, nbit: int, ebit: int=2, mbit: int=1, sc_ebit: int=8, sc_mbit: int=0, unsigned: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        Parameters:
        - ebit: element exponent bit width
        - mbit: element mantissa bit width
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        - use_round: Whether to use rounding for scale (only in MX)
        - use_ceil: Whether to use ceil for scale (only in NV-block quantization)
        """
        super().__init__(nbit, unsigned)
        self.ebit = ebit
        self.mbit = mbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
        self.use_ceil = use_ceil


    def get_bound(self, x:torch.Tensor):
        max_norm = float(2**(self.nbit-1) - 1) / 2**(self.nbit-2)

        # update bound
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)
    
    def get_shared_scale(self, x:torch.Tensor, axis=1) -> torch.Tensor:
        if x.dtype == torch.float32:
            min_normal = FP32_MIN_NORMAL
        elif x.dtype == torch.float16:
            min_normal = FP16_MIN_NORMAL
        else:
            raise NotImplementedError(f"Non-supported data type: {x.dtype}")
        
        ele_max, _ = torch.max(torch.abs(x), dim=-1, keepdim=True) # abs max
        ele_emax = 2**(self.ebit-1)
        if self.sc_mbit == 0: # conventional MXFP, shift scaling
            if self.use_round:
                shared_exp = torch.round(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    ))
                # shared_exp = torch.floor(torch.log2(
                #     ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                #     )+0.5)
            else:
                shared_exp = torch.floor(torch.log2(
                    ele_max + min_normal * (ele_max == 0).type(ele_max.dtype)
                    ))
            emax = 2**(self.sc_ebit-1) - 1
            emin = 2-2**(self.sc_ebit-1)
            shared_exp = shared_exp - ele_emax
            shared_exp[shared_exp > emax] = emax
            shared_exp[shared_exp < emin] = emin
            shared_scale = 2**shared_exp
        else: # NVFP: FP scale
            # low bit multiplication x*1/6
            # ele_max = quant_element(ele_max, self.sc_ebit, 6, allow_subnormal=False)
            # div_max_norm = torch.tensor([1/get_max_norm(ebit=self.ebit, mbit=self.mbit)], device=x.device, dtype=x.dtype)
            # div_max_norm = quant_element(div_max_norm, 5, 6, allow_subnormal=False)
            # shared_scale = ele_max * div_max_norm
            # normal division x/6
            shared_scale = ele_max / get_max_norm(ebit=self.ebit, mbit=self.mbit)
            # in case share scale can't be quantized to the range of [min_subnorm, max_norm], could happend to E < 8 cases
            scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
            shared_scale = shared_scale.clip(min = scale_min, max = get_max_norm(self.sc_ebit, self.sc_mbit))
            shared_scale = quant_element(shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, use_ceil=self.use_ceil)
        if (shared_scale == 0).sum() > 0 or shared_scale.isnan().sum() > 0:
            indices = (shared_scale == 0).nonzero()
            breakpoint()
        return shared_scale

    def calculate_qparam(self, x:torch.Tensor, axis:int=2):
        scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_scale = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale
    
    def forward(self, x:torch.Tensor, axis:int=2):
        self.get_bound(x)
        scale, zero_point, shared_scale = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_scale

class MXINTQuantizer(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, sc_ebit: int=8, sc_mbit: int=0, block_size:int=32, per_tensor_scale: bool = False, scale_allow_subnormal: bool = False, use_round: bool=False):
        """
        MXINT quantization
        Parameters:
        - nbit: element bit width (conventional MX only support INT4 & INT8)
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        - use_round: Whether to use rounding for scale (only in MX)
        """
        super().__init__(nbit, train_flag, unsigned)

        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.observer = MXINTObserver(nbit=self.nbit, sc_ebit=self.sc_ebit, sc_mbit=self.sc_mbit, unsigned=self.unsigned, scale_allow_subnormal=scale_allow_subnormal, use_round=use_round)
        
    def register_qparams(self):
        super().register_qparams()
        self.register_buffer("shared_scale", torch.tensor(1.0))

    def reshape(self, x:torch.Tensor):
        if len(x.shape) == 4:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 3:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 2:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [1], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported Layer Type with the shape of {list(x.size())}!")
        return x, axes, orig_shape, padded_shape
    
    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes] # +1 for the new group dimension
        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        zero_mask = xg==0
        xg = xg / self.shared_scale
        xg[zero_mask] = 0
        xg = xg / (self.scale)
        # if xg.isinf().sum() > 0:
            # inf_indices = torch.nonzero(xg.isinf(), as_tuple = False)
            # breakpoint()
        # xg = xg * zero_mask
        # if xg.isnan().sum() > 0:
        # xg = xg / (self.scale) # scale element into INT range
        # xg = xg * zero_mask
        
        # if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
        #     nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
        #     inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
        # import pdb;pdb.set_trace()

        # round to nearest

        xg = torch.sign(xg) *  torch.floor(torch.abs(xg) + 0.5)
        xg = xg.clip(min = -(2**(self.nbit-1)), max = 2**(self.nbit-1)-1)
        
        # truncation (closer to 0)
        # xg = torch.trunc(xg)
        # import pdb;pdb.set_trace()
        
        xg = xg.clamp(self.qlb, self.qub)
        
        # rescale back
        xg = xg.mul(self.scale)
        xg = xg.mul(self.shared_scale).to(x.dtype)
        # if xg.isinf().sum() > 0:
            # inf_indices = torch.nonzero(xg.isinf(), as_tuple = False)
            # breakpoint()

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        # if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
        #     nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
        #     inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
        #     import pdb;pdb.set_trace()
        return xg

    def trainFunc(self, x:torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq

class MXFPQuantizer(_QBase):
    def __init__(
        self, 
        train_flag: bool = True, 
        unsigned: bool = False, 
        block_size:int=32, 
        ebit:int=2, mbit:int=3, 
        sc_ebit:int=0, sc_mbit:int=8, 
        per_tensor_scale: bool = False, 
        scale_allow_subnormal: bool = True, 
        use_round: bool=False, use_ceil: bool=False, 
        use_quant_kernel: bool=True
    ):
        """
        MXFP quantization, support MXFP or NVFP
        Parameters:
        - ebit: element exponent bit width
        - mbit: element mantissa bit width
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        - use_round: Whether to use rounding for scale (only in MX)
        - use_ceil: Whether to use ceil for scale (only in NV-block quantization)
        """
        super().__init__(32, train_flag, unsigned)

        # assert nbit in [4, 8], "According to Microsoft, current MX INT format only support 4bit and 8bit."
        assert ebit != 0, "MXFP quantization can't have 0 exponent bit width!"
        self.ebit = ebit
        self.mbit = mbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
        self.use_ceil = use_ceil
        self.observer = MXFPObserver(nbit=self.nbit, unsigned=self.unsigned, ebit=self.ebit, mbit=self.mbit, sc_ebit=sc_ebit, sc_mbit=sc_mbit, scale_allow_subnormal=scale_allow_subnormal, use_round=use_round, use_ceil=use_ceil)

        # low precision fp quantization kernel
        self.use_quant_kernel = use_quant_kernel
        lp_config = FPQuantConfig(ebit=self.ebit, mbit=self.mbit, allow_subnormal=True, detailed_output=False)
        if use_quant_kernel:
            self.lpfp_quant = GLOBAL_FP_QUANT_CACHE.get(lp_config)
        else:
            self.lpfp_quant = make_quant_element(lp_config.ebit, lp_config.mbit, lp_config.allow_subnormal, lp_config.detailed_output, lp_config.use_ceil, lp_config.emax, lp_config.emin)
    
    def register_qparams(self):
        super().register_qparams()
        self.register_buffer("shared_scale", torch.tensor(1.0))

    def reshape(self, x:torch.Tensor):
        if len(x.shape) == 4:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 3:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 2:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [1], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported Layer Type with the shape of {list(x.size())}!")
        return x, axes, orig_shape, padded_shape
    
    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        # pruning
        if False:
            topk = 2
            xg_abs = torch.abs(xg)
            prune_idx = torch.topk(xg_abs, topk, dim=-1, largest=False).indices
            prune_mask = torch.ones_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
            prune_mask.scatter_(-1, prune_idx, False)
            xg = xg * prune_mask

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        # LUT division
        # _, scale_exp, scale_mant = quant_element(self.shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, detailed_output=True)
        # scale_exp = scale_exp-15 # bias
        # xg = reciprocal_lut_div(xg, scale_exp, scale_mant, self.sc_mbit, 4)
        # Normal division
        xg = xg.contiguous()
        xg = xg / self.shared_scale

        # qxg= quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=False)
        # xg, exp, mant = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)
        xg = self.lpfp_quant(xg)

        # if not torch.allclose(qxg, xg, rtol=1e-5, atol=1e-6):
        #     f"max abs diff: {((qxg - xg).abs().max().item())}"
        #     breakpoint()
        # assert torch.allclose(qxg, xg, rtol=1e-5, atol=1e-6)
        # print(f"exp=3 ratio: {(exp==3).sum()/exp.numel()}")
        # if ((exp==3).sum()/exp.numel()) > 0.5: breakpoint()
        # if self.ebit != 0:
        #     ele_exp = torch.floor(torch.log2(
        #         torch.abs(xg) + (xg == 0).type(xg.dtype)))
        #     ele_emin = -(2**(self.ebit -1)) + 2 # no subnormal case
        #     # import pdb; pdb.set_trace()
        #     ele_exp = ele_exp.clip(min = ele_emin, max = ele_emax)

        # # xg should now in the scale of (-2,2)
        # # import pdb; pdb.set_trace()
        # # scale up to integer range
        # xg = xg / (2**ele_exp) * (2**(self.mbit))
        # # import pdb; pdb.set_trace()
        # # nearest rounding
        # xg = xg.sign() * torch.floor(torch.abs(xg) + 0.5) 
        # self.qub = 2**(self.mbit+1)-1
        # self.qlb = - self.qub
        # xg = xg.clamp(max = self.qub, min = self.qlb)
        # # undo scaling
        # xg = xg / (2**(self.mbit)) * (2**ele_exp)
        xg = xg.mul(self.shared_scale)
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

    def get_data_point(self, ebit, mbit):
        """
        Given exp bit and mantissa bit, list all possible fp numbers in a list
        Args:
            ebit: exponent bit width
            mbit: mantissa bit width
        Returns:
            data_points: list of all possible fp numbers
        """
        data_points = []
        
        # Handle zero
        data_points.append(0.0)
        
        # Calculate bias
        bias = 2**(ebit-1) - 1
        
        # For each possible exponent
        for e in range(2**ebit):
            # For each possible mantissa
            for m in range(2**mbit):
                    
                # Calculate value
                if e == 0:  # Subnormal numbers
                    exp = 1 - bias
                    mant = m / (2**mbit)
                else:  # Normal numbers
                    exp = e - bias
                    mant = 1 + m / (2**mbit)
                    
                val = mant * (2.0**exp)
                
                # Add positive and negative values
                data_points.append(val)
                data_points.append(-val)
                
        # Remove duplicates and sort
        data_points = list(set(data_points))
        data_points.sort()
        
        return data_points

    def lp_check_coverage(self, x:torch.Tensor):
        '''
        Calculate the coverage of the quantized tensor
        Args:
            x: quantized tensor
        Returns:
            coverage: the coverage of the quantized tensor
        '''
        def map_by_masks(x: torch.Tensor, mapping: dict, default=999):
            out = x if default is None else torch.full_like(x, default)
            for k, v in mapping.items():
                mask = (x == k)
                out = torch.where(mask, torch.as_tensor(v, device=x.device, dtype=out.dtype), out)
            return out

        def turn_list_into_dict(ls):
            fp_dict = {}
            for k, v in enumerate(ls):
                fp_dict[v] = k
            return fp_dict
        
        fp_list = self.get_data_point(self.ebit, self.mbit)
        fp_dict = turn_list_into_dict(fp_list)
        x_2d = x.reshape(-1, x.shape[-1])

        max_ele = x_2d.max(dim=-1).values
        min_ele = x_2d.min(dim=-1).values
        mapped_max = map_by_masks(max_ele, fp_dict)
        mapped_min = map_by_masks(min_ele, fp_dict)
        assert 999 not in mapped_max and 999 not in mapped_min, "there exists a number that is not in the fp_list"
        coverage = (mapped_max-mapped_min+1) / len(fp_list)

        max_in_underutilized_side = torch.where(max_ele > min_ele.abs(), min_ele.abs(), max_ele)
        fp_list_abs = torch.tensor(fp_list[-len(fp_list)//2:], device=x.device, dtype=x.dtype)
        counts = (max_in_underutilized_side[:, None]==fp_list_abs).sum(dim=0)
        
        return coverage.mean(), counts

    def get_quant_coverage(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        xg = xg / self.shared_scale

        xg, exp, mant = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)
        coverage, counts = self.lp_check_coverage(xg)
        return coverage, counts

    def trainFunc(self, x:torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq

class MXFPExcludeQuantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        MXFP quantization, support MXFP or NVFP
        Excluding the largest element in each group
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        e.g. conventional MXINT8: nbit = 8, sc_ebit = 0, sc_mbit = 8
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        xg_abs = torch.abs(xg)
        # get xg_abs topk at last dim
        # Example: top 3
        # topk_values, topk_indices = xg_abs.topk(k=8, dim=-1)
        _, max_idx = xg_abs.max(dim=-1, keepdim=True)
        largest_mask = torch.zeros_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
        largest_mask.scatter_(-1, max_idx, True)
        # truncated = xg_abs < xg_abs.max(dim=-1, keepdim=True).values/6
        # truncate_mask = torch.zeros_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
        # truncate_mask.scatter_(-1, truncated, True)
        # _, max_idx = xg_abs.max(dim=-1, keepdim=True)
        # largest_mask = torch.ones_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
        # largest_mask.scatter_(-1, max_idx, False)
        # largest_mask = largest_mask.reshape(xg.shape)
        if self.train_flag:
            remove_largest = xg * largest_mask
            scale, zero_point, shared_scale = self.observer(remove_largest, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        # LUT division
        # _, scale_exp, scale_mant = quant_element(self.shared_scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, detailed_output=True)
        # scale_exp = scale_exp-15 # bias
        # xg = reciprocal_lut_div(xg, scale_exp, scale_mant, self.sc_mbit, 4)
        # Normal division
        xg_scaled = xg / self.shared_scale

        xg_scaled, exp, mant = quant_element(xg_scaled, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)
        xg_scaled = xg_scaled.mul(self.shared_scale)
        xg = torch.where(largest_mask, xg, xg_scaled)
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

# deprecated
class MXFPMixOldQuantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False, subgs:int=2, hp_m_bit:int=5):
        """
        MXFP quantization
        E2M5 for largest number in each subgroup, E2M1 for other numbers in the subgroup
        With mixed precision, hp_m_bit is used in the higher precision mantissa bit
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
        self.subgs = subgs # subgroup size
        self.hp_m_bit = hp_m_bit # high precision mantissa bit
        hp_config = FPQuantConfig(ebit=ebit, mbit=hp_m_bit, allow_subnormal=True, detailed_output=False, use_ceil=use_ceil)
        self.hpfp_quant = GLOBAL_FP_QUANT_CACHE.get(hp_config)
        # hp_config_detailed = FPQuantConfig(ebit=ebit, mbit=hp_m_bit, allow_subnormal=True, detailed_output=True, use_ceil=use_ceil)
        # self.hpfp_quant_detailed = GLOBAL_FP_QUANT_CACHE.get(hp_config_detailed)

    def mixed_quant_by_largest(self, x:torch.Tensor):
        fp8_quant_element = self.hpfp_quant(x)
        fp4_quant_element = self.lpfp_quant(x)
        x_abs = torch.abs(x)
        subgroup = x_abs.view(*(x.shape[:-1]), -1, self.subgs)
        _, max_idx = subgroup.max(dim=-1, keepdim=True)
        largest_mask = torch.zeros_like(subgroup, dtype=torch.bool, device=subgroup.device)
        largest_mask.scatter_(-1, max_idx, True)
        largest_mask = largest_mask.reshape(x.shape)
        mixed_quant = torch.where(largest_mask, fp8_quant_element, fp4_quant_element)
        return mixed_quant

    def mixed_quant_by_random(self, x:torch.Tensor):
        fp8_quant_element = quant_element(x, 2, 5, allow_subnormal=True)
        fp4_quant_element = quant_element(x, 2, 1, allow_subnormal=True)
        subgroup = x.view(*(x.shape[:-1]), -1, self.subgs)
        idx = torch.randint(subgroup.size(-1), size=subgroup.shape[:-1], device=subgroup.device)
        random_mask = torch.zeros_like(subgroup, dtype=torch.bool, device=subgroup.device)
        random_mask.scatter_(-1, idx.unsqueeze(-1), True)
        random_mask = random_mask.reshape(x.shape)
        mixed_quant = torch.where(random_mask, fp8_quant_element, fp4_quant_element)
        return mixed_quant

    def mixed_quant_by_error(self, x:torch.Tensor):
        # fp8_quant_element = quant_element(x, 2, 5, allow_subnormal=True)
        # fp4_quant_element = quant_element(x, 2, 1, allow_subnormal=True)
        fp8_quant_element = self.hpfp_quant(x)
        fp4_quant_element = self.lpfp_quant(x)
        x_abs = torch.abs(x)
        subgroup = x_abs.view(*(x.shape[:-1]), -1, self.subgs)
        fp8_subgroup = fp8_quant_element.view(*(x.shape[:-1]), -1, self.subgs).abs()
        fp4_abs = torch.abs(fp4_quant_element)
        fp4_subgroup = fp4_abs.view(*(x.shape[:-1]), -1, self.subgs)
        error_subgroup = (fp8_subgroup - fp4_subgroup).abs() # get error between fp8 and fp4
        _, max_idx = error_subgroup.max(dim=-1, keepdim=True)
        largest_mask = torch.zeros_like(subgroup, dtype=torch.bool, device=subgroup.device)
        largest_mask.scatter_(-1, max_idx, True)
        largest_mask = largest_mask.reshape(x.shape)
        mixed_quant = torch.where(largest_mask, fp8_quant_element, fp4_quant_element)
        return mixed_quant

    def mixed_quant_by_error_hw(self, x:torch.Tensor):
        hp_quant_element, hp_exp, hp_mant = quant_element(x, self.ebit, self.hp_m_bit, allow_subnormal=True, detailed_output=True)
        # lp_quant_element = quant_element(x, self.ebit, self.mbit, allow_subnormal=True)
        # hp_quant_element, hp_exp, hp_mant = self.hpfp_quant_detailed(x)
        lp_quant_element = self.lpfp_quant(x)
        breakpoint()
        hp_exp = hp_exp.view(*(x.shape[:-1]), -1, self.subgs)
        hp_mant = hp_mant.view(*(x.shape[:-1]), -1, self.subgs)
        mant_round_bit = (hp_mant//(2**3))%2
        truncated_mant = hp_mant%(2**(5-2))
        error_subgroup_pre_exp = torch.where(mant_round_bit==1, 8-truncated_mant, truncated_mant)
        e2_bias = 2**(2-1)-1
        biased_hp_exp = torch.where(hp_exp==0, 0, hp_exp-e2_bias)
        error_subgroup = error_subgroup_pre_exp*2**(biased_hp_exp)
        _, max_idx = error_subgroup.max(dim=-1, keepdim=True)
        largest_mask = torch.zeros_like(hp_mant, dtype=torch.bool, device=hp_mant.device)
        largest_mask.scatter_(-1, max_idx, True)
        largest_mask = largest_mask.reshape(x.shape)
        mixed_quant = torch.where(largest_mask, hp_quant_element, lp_quant_element)
        return mixed_quant

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        # sparsify = (x.shape != 1).sum() == 1 # decode stage
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        # pruning
        # if sparsify:
        #     topk = 1
        #     xg_abs = torch.abs(xg)
        #     prune_idx = torch.topk(xg_abs, topk, dim=-1, largest=False).indices
        #     prune_mask = torch.ones_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
        #     prune_mask.scatter_(-1, prune_idx, False)
        #     xg = xg * prune_mask

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        xg = xg / self.shared_scale

        # xg = self.mixed_quant_by_error_hw(xg)
        xg = self.mixed_quant_by_error(xg)

        xg = xg.mul(self.shared_scale)
        
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPMixQuantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False, N: int=1, M: int=32, hp_m_bit:int=5):
        """
        Mixed precision MXFP quantization
        Using N:M strategy, every M elements in a subgroup, quantize N elements that generate more error with higher precision
        E2M5 for largest number in each subgroup, E2M1 for other numbers in the subgroup
        With mixed precision, hp_m_bit is used in the higher precision mantissa bit
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
        assert M <= block_size, f"In N:M, M should be less than or equal to block size, getting M = {M}"
        assert block_size % M == 0, f"In N:M, block size should be divisible by M, getting M = {M}"
        self.N = N # number of elements to quantize with higher precision
        self.M = M # number of elements in each subgroup
        self.hp_m_bit = hp_m_bit # high precision mantissa bit
        # lp_config = FPQuantConfig(ebit=ebit, mbit=1, allow_subnormal=True, detailed_output=False, use_ceil=False)
        # self.lpfp_quant = GLOBAL_FP_QUANT_CACHE.get(lp_config)
        hp_config = FPQuantConfig(ebit=ebit, mbit=self.hp_m_bit, allow_subnormal=True, detailed_output=False, use_ceil=use_ceil)
        self.hpfp_quant = GLOBAL_FP_QUANT_CACHE.get(hp_config)
        # hp_config_detailed = FPQuantConfig(ebit=ebit, mbit=hp_m_bit, allow_subnormal=True, detailed_output=True, use_ceil=use_ceil)
        # self.hpfp_quant_detailed = GLOBAL_FP_QUANT_CACHE.get(hp_config_detailed)

    def mixed_quant_by_error(self, x:torch.Tensor):
        # fp8_quant_element = quant_element(x, 2, 5, allow_subnormal=True)
        # fp4_quant_element = quant_element(x, 2, 1, allow_subnormal=True)
        fp8_quant_element = self.hpfp_quant(x)
        fp4_quant_element = self.lpfp_quant(x)
        x_abs = torch.abs(x)
        subgroup = x_abs.view(*(x.shape[:-1]), -1, self.M)
        fp8_subgroup = fp8_quant_element.view(*(x.shape[:-1]), -1, self.M).abs()
        fp4_abs = torch.abs(fp4_quant_element)
        fp4_subgroup = fp4_abs.view(*(x.shape[:-1]), -1, self.M)
        error_subgroup = (fp8_subgroup - fp4_subgroup).abs() # get error between fp8 and fp4
        # get topk most error elements
        if self.N == 1:
            _, hp_idx = error_subgroup.max(dim=-1, keepdim=True) # max is much faster than topk
        else:
            hp_idx = torch.topk(error_subgroup, self.N, dim=-1, largest=True).indices
        largest_mask = torch.zeros_like(subgroup, dtype=torch.bool, device=subgroup.device)
        largest_mask.scatter_(-1, hp_idx, True)
        largest_mask = largest_mask.reshape(x.shape)
        mixed_quant = torch.where(largest_mask, fp8_quant_element, fp4_quant_element)
        # mixed_quant = torch.where(x.abs() < 0.125, fp8_quant_element, mixed_quant)
        # fp4_error = (x - fp4_quant_element).abs()
        # fp8_error = (x - fp8_quant_element).abs()
        # if (fp8_error > (fp4_error)).sum() > 0:
        #     breakpoint()
        return mixed_quant

    def mixed_quant_by_error_hw(self, x:torch.Tensor):
        # TODO: implement N:M
        # hp_quant_element, hp_exp, hp_mant = quant_element(x, self.ebit, self.hp_m_bit, allow_subnormal=True, detailed_output=True)
        # lp_quant_element = quant_element(x, self.ebit, self.mbit, allow_subnormal=True)
        hp_quant_element, hp_exp, hp_mant = self.hpfp_quant_detailed(x)
        lp_quant_element = self.lpfp_quant(x)
        hp_exp = hp_exp.view(*(x.shape[:-1]), -1, self.subgs)
        hp_mant = hp_mant.view(*(x.shape[:-1]), -1, self.subgs)
        mant_round_bit = (hp_mant//(2**3))%2
        truncated_mant = hp_mant%(2**(5-2))
        error_subgroup_pre_exp = torch.where(mant_round_bit==1, 8-truncated_mant, truncated_mant)
        e2_bias = 2**(2-1)-1
        biased_fp8_exp = torch.where(hp_exp==0, 0, hp_exp-e2_bias)
        error_subgroup = error_subgroup_pre_exp*2**(biased_fp8_exp)
        _, max_idx = error_subgroup.max(dim=-1, keepdim=True)
        largest_mask = torch.zeros_like(hp_mant, dtype=torch.bool, device=hp_mant.device)
        largest_mask.scatter_(-1, max_idx, True)
        largest_mask = largest_mask.reshape(x.shape)
        mixed_quant = torch.where(largest_mask, hp_quant_element, lp_quant_element)
        return mixed_quant

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        # sparsify = (x.shape != 1).sum() == 1 # decode stage
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        xg = xg / self.shared_scale

        # xg = self.mixed_quant_by_error_hw(xg)
        mixed_xg = self.mixed_quant_by_error(xg)
        w4a8_quant = self.hpfp_quant(xg)
        qsnr_mixed = qsnr(mixed_xg, xg)
        qsnr_w4a8 = qsnr(w4a8_quant, xg)
        if qsnr_mixed > qsnr_w4a8:
            breakpoint()

        xg = mixed_xg.mul(self.shared_scale)
        
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            breakpoint()
            
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPAsymQuantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        Asymmetric MXFP quantization
        Scale other side with 2x
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
   
    def asym_quant(self, xg:torch.Tensor):
        """
        Scale the other side with 2x
        Quantize to same FP format as the other side
        """
        # asym quant
        # check max is pos or neg
        max_is_pos = xg.max(dim=-1).values == xg.abs().max(dim=-1).values
        pos_mask = xg > 0
        scale_mask = (max_is_pos[..., None] & ~pos_mask) | (~max_is_pos[..., None] & pos_mask)
        xg_scaled_2 = xg.clone()
        xg_scaled_2[scale_mask]*=2
        xg_asym_quant_2 = quant_element(xg_scaled_2, self.ebit, self.mbit, allow_subnormal=True)
        xg_asym_quant_2[scale_mask]/=2
        return xg_asym_quant_2
   
    def asym_quant_int(self, xg:torch.Tensor):
        """
        Scale the other side with 2x
        Quantize to same INT format (if FP4, then INT3 with 0, 0.5, ..., 3.5)
        """
        max_is_pos = xg.max(dim=-1).values == xg.abs().max(dim=-1).values
        pos_mask = xg > 0
        scale_mask = (max_is_pos[..., None] & ~pos_mask) | (~max_is_pos[..., None] & pos_mask)
        max_side = xg.clone()
        other_side = xg.clone()
        max_side[~scale_mask] = 0
        other_side[scale_mask] = 0
        max_side_quant = self.lpfp_quant(max_side)
        other_side_quant = quant_element_int(other_side, self.ebit+self.mbit, upscale=2)
        xg_asym_quant_2 = max_side_quant + other_side_quant
        return xg_asym_quant_2

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        # pruning
        if False:
            topk = 2
            xg_abs = torch.abs(xg)
            prune_idx = torch.topk(xg_abs, topk, dim=-1, largest=False).indices
            prune_mask = torch.ones_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
            prune_mask.scatter_(-1, prune_idx, False)
            xg = xg * prune_mask

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        xg = xg / self.shared_scale

        # sym quant
        # xg_sym_quant, exp, mant = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)
        xg_sym_quant = self.lpfp_quant(xg)

        # asym quant
        # xg_asym_quant_2 = self.asym_quant(xg)
        xg_asym_quant_2 = self.asym_quant_int(xg)


        sym_qsnr = qsnr(xg, xg_sym_quant, dim=-1)
        asym_2_qsnr = qsnr(xg, xg_asym_quant_2, dim=-1)
        sym_is_better = sym_qsnr >= asym_2_qsnr
        
        xg = torch.where(sym_is_better[..., None], xg_sym_quant, xg_asym_quant_2)

        xg = xg.mul(self.shared_scale)
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPAsymL2Quantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        Asymmetric MXFP quantization
        Scale other side with 2x or 4x
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
   
    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale
        
        xg = xg / self.shared_scale

        # sym quant
        xg_sym_quant, exp, mant = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)

        # asym quant
        # check max is pos or neg
        max_is_pos = xg.max(dim=-1).values == xg.abs().max(dim=-1).values
        pos_mask = xg > 0
        scale_mask = (max_is_pos[..., None] & ~pos_mask) | (~max_is_pos[..., None] & pos_mask)
        xg_scaled_2 = xg.clone()
        xg_scaled_2[scale_mask]*=2
        # xg_asym_quant_2 = quant_element(xg_scaled_2, self.ebit, self.mbit, allow_subnormal=True)
        xg_asym_quant_2 = quant_element_int(xg_scaled_2, self.mbit+self.ebit, upscale=1)
        xg_asym_quant_2[scale_mask]/=2


        xg_scaled_4 = xg.clone()
        xg_scaled_4[scale_mask]*=4
        # xg_asym_quant_4 = quant_element(xg_scaled_4, self.ebit, self.mbit, allow_subnormal=True)
        xg_asym_quant_4 = quant_element_int(xg_scaled_4, self.mbit+self.ebit, upscale=1)
        xg_asym_quant_4[scale_mask]/=4


        sym_qsnr = qsnr(xg, xg_sym_quant, dim=-1)
        asym_2_qsnr = qsnr(xg, xg_asym_quant_2, dim=-1)
        asym_4_qsnr = qsnr(xg, xg_asym_quant_4, dim=-1)
        sym_is_better = torch.logical_and(sym_qsnr >= asym_2_qsnr, sym_qsnr >= asym_4_qsnr)
        asym_2_is_better = torch.logical_and(asym_2_qsnr >= sym_qsnr, asym_2_qsnr >= asym_4_qsnr)
        
        best_xg = torch.where(sym_is_better[..., None], xg_sym_quant, xg_asym_quant_4)
        best_xg = torch.where(asym_2_is_better[..., None], xg_asym_quant_2, best_xg)
        if (qsnr(xg, best_xg, dim=-1) < qsnr(xg, xg_sym_quant, dim=-1)).sum() != 0:
            breakpoint()

        assert (qsnr(xg, best_xg, dim=-1) < qsnr(xg, xg_sym_quant, dim=-1)).sum() == 0 
        assert (qsnr(xg, best_xg, dim=-1) < qsnr(xg, xg_asym_quant_2, dim=-1)).sum() == 0 
        assert (qsnr(xg, best_xg, dim=-1) < qsnr(xg, xg_asym_quant_4, dim=-1)).sum() == 0 
        xg = best_xg

        xg = xg.mul(self.shared_scale)
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class AMXFPQuantizer(MXFPQuantizer):
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=5, sc_mbit:int=0, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        AMXFP quantization
        (Ref: Lee, Janghwan, et al. "Amxfp4: Taming activation outliers with asymmetric microscaling floating-point for 4-bit llm inference." arXiv preprint arXiv:2411.09909 (2024).)
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
   
    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]


        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0


        pos_mask = xg > 0
        only_pos_xg = xg.clone()
        only_neg_xg = xg.clone()
        only_pos_xg[~pos_mask] = 0
        only_neg_xg[pos_mask] = 0
        _, _, pos_shared_scale = self.observer(only_pos_xg, shared_exp_axes[0])
        _, _, neg_shared_scale = self.observer(only_neg_xg, shared_exp_axes[0])
        del only_pos_xg, only_neg_xg
        
        xg = torch.where(pos_mask, xg / pos_shared_scale, xg / neg_shared_scale)

        # xg, exp, mant = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=True)
        xg = self.lpfp_quant(xg)

        xg = torch.where(pos_mask, xg.mul(pos_shared_scale), xg.mul(neg_shared_scale))
        if self.per_tensor_scale:
            xg = xg * tensor_scale

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXINTAsymQuantizer(MXINTQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, sc_ebit: int=8, sc_mbit: int=0, block_size:int=32, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False):
        """
        MXINTAsym quantization
        Shift all value to positive with zero point
        Quantize with integer quantization
        Now only support FP scale
        nbit: element bit width (conventional MX only support INT4 & INT8)
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        super().__init__(nbit, train_flag, True, sc_ebit, sc_mbit, block_size, per_tensor_scale, scale_allow_subnormal, use_round)
        self.observer = MXINTAsymObserver(nbit=self.nbit, sc_ebit=self.sc_ebit, sc_mbit=self.sc_mbit, unsigned=self.unsigned, scale_allow_subnormal=scale_allow_subnormal, use_round=use_round)

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes] # +1 for the new group dimension
        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        # zero_mask = xg==0
        xg = xg / self.shared_scale
        int_zero_point = torch.floor(self.zero_point + 0.5)
        fp_zero_point = quant_element(self.zero_point, 3, 4, False, False, False)
        xg = xg + fp_zero_point
        # xg[zero_mask] = 0

        xg = xg / (self.scale)
        
        xg = torch.sign(xg) *  torch.floor(torch.abs(xg) + 0.5)
        
        xg = xg.clamp(self.qlb, self.qub)
        
        # rescale back
        xg = xg.mul(self.scale)
        xg = xg - fp_zero_point
        xg = xg.mul(self.shared_scale).to(x.dtype)

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            breakpoint()
            # import pdb;pdb.set_trace()

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXDualQuantizer(MXFPQuantizer):
    """
    MXDual quantization
    Use dual scaling factor within the block, one for inliers and one for outliers
    """
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=128, ebit:int=2, mbit:int=3, sc_ebit:int=5, sc_mbit:int=3, per_tensor_scale: bool = False, scale_allow_subnormal: bool = False, use_round: bool=False, use_ceil: bool=False):
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil)
        self.inlier_observer = self.observer
        self.inlier_lpfp_quant = self.lpfp_quant
        # self.outlier_observer = MXFPObserver(nbit=self.nbit, unsigned=self.unsigned, ebit=ebit, mbit=mbit, sc_ebit=5, sc_mbit=3, scale_allow_subnormal=scale_allow_subnormal, use_round=False, use_ceil=True)
        # self.outlier_lp_config = FPQuantConfig(ebit=ebit, mbit=mbit, allow_subnormal=True, detailed_output=False)
        # self.outlier_lpfp_quant = GLOBAL_FP_QUANT_CACHE.get(self.outlier_lp_config)
        # self.outlier_lpfp_quant = self.inlier_lpfp_quant
        self.outlier_lpfp_quant = partial(quant_element_int_non_zero, nbit=4)
    

    def q_split_first(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        # split to two parts first
        shared_channel = 8
        xg_reshape = xg.view([-1, shared_channel, xg.shape[-2], xg.shape[-1]])
        # xg_reshape_grouped = xg_reshape.pow(2).sum(dim=1, keepdim=True) # not good
        xg_reshape_grouped = xg_reshape.abs().sum(dim=1, keepdim=True)
        xg_reshape_grouped_std = xg_reshape_grouped.std(dim=-1, keepdim=True)
        use_first_scale = xg_reshape_grouped > xg_reshape_grouped.mean(dim=-1, keepdim=True)
        # use_first_scale = xg_reshape_grouped > (xg_reshape_grouped.mean(dim=-1, keepdim=True) + xg_reshape_grouped_std)
        use_first_scale = use_first_scale.repeat([1, shared_channel, 1, 1]).view(xg.shape)
        # breakpoint()

        xg_use_first_scale = xg * use_first_scale
        first_scale = xg_use_first_scale.abs().max(dim=-1, keepdim=True).values/8
        first_scaled_xg = xg_use_first_scale/first_scale
        first_scaled_xg_q = self.outlier_lpfp_quant(first_scaled_xg)
        xg_use_second_scale = xg * ~use_first_scale
        second_scale = xg_use_second_scale.abs().max(dim=-1, keepdim=True).values/6
        second_scaled_xg = xg_use_second_scale/second_scale
        second_scaled_xg_q = self.inlier_lpfp_quant(second_scaled_xg)

        # breakpoint()

        xg = torch.where(use_first_scale, first_scaled_xg_q*first_scale, second_scaled_xg_q*second_scale)

        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg
        

    def q_std(self, x:torch.Tensor):
        """
        Use std 
        """
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        topk = 32
        xg_abs = xg.abs()
        first_scale = xg_abs.max(dim=-1, keepdim=True).values/8
        first_scaled_xg = xg/first_scale
        first_scaled_xg_q = self.outlier_lpfp_quant(first_scaled_xg)
        first_error = (xg - (first_scaled_xg_q*first_scale)).abs()

        std = xg_abs.std(dim=-1, keepdim=True)
        second_scale = (xg_abs.mean(dim=-1, keepdim=True) + std)/6
        second_scaled_xg = xg/second_scale
        second_scaled_xg_q = self.inlier_lpfp_quant(second_scaled_xg)
        second_error = (xg - (second_scaled_xg_q*second_scale)).abs()
        use_first_scale = second_error > first_error

        # breakpoint()
        
        mask = use_first_scale
        # breakpoint()
        # xg_scaled = torch.where(use_first_scale, first_scaled_xg, second_scaled_xg)

        # xg_lp = self.inlier_lpfp_quant(xg_scaled)
        mask_share_channel = 8
        mask_reshape = use_first_scale.view([-1, mask_share_channel, use_first_scale.shape[-2], use_first_scale.shape[-1]])
        mask_reshape_any = mask_reshape.any(dim=1, keepdim=True)
        mask = mask_reshape_any.repeat([1, mask_share_channel, 1, 1]).view(use_first_scale.shape)
        # breakpoint()

        xg = torch.where(mask, first_scaled_xg_q*first_scale, second_scaled_xg_q*second_scale)

        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg
        

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        topk = 32
        xg_abs = xg.abs()
        first_scale = xg_abs.max(dim=-1, keepdim=True).values/8
        first_scaled_xg = xg/first_scale
        first_scaled_xg_q = self.outlier_lpfp_quant(first_scaled_xg)
        first_error = (xg - (first_scaled_xg_q*first_scale)).abs()
        _, topk_idx = first_error.topk(k=topk, dim=-1, largest=True)
        second_scales = (xg_abs.gather(-1, topk_idx)/4).unsqueeze(-2) # [out_ch, blocks, 1, 4]

        xg_expand = xg.unsqueeze(-1).repeat([1] * xg.dim() + [topk])
        xg_expand_scaled = xg_expand/second_scales
        xg_expand_scaled_q = self.inlier_lpfp_quant(xg_expand_scaled)
        second_scale_error = (xg_expand - (xg_expand_scaled_q*second_scales)).abs() # [out_ch, blocks, block_size, 4]

        # first_scaled_xg_q_expand = first_scaled_xg_q.unsqueeze(-1).repeat([1] * first_scaled_xg_q.dim() + [topk])
        first_error_expand = first_error.unsqueeze(-1).repeat([1] * first_error.dim() + [topk]) # [out_ch, blocks, block_size, 4]
        use_first_scale = second_scale_error > first_error_expand
        error = torch.where(use_first_scale, first_error_expand, second_scale_error) # [out_ch, blocks, block_size, 4]
        error_total = error.sum(dim=-2, keepdim=True) # [out_ch, blocks, 1, 4]
        _, min_idx = error_total.min(dim=-1, keepdim=True) # [out_ch, blocks, 1, 1]
        second_scale = second_scales.gather(-1, min_idx).squeeze(-1) # [out_ch, blocks, 1]
        

        # _, max_idx = first_error.max(dim=-1, keepdim=True) # second scale
        # second_scale = xg_abs.gather(-1, max_idx)/4
        second_scaled_xg = xg/second_scale
        second_scaled_xg_q = self.inlier_lpfp_quant(second_scaled_xg)
        second_error = (xg - (second_scaled_xg_q*second_scale)).abs()
        use_first_scale = second_error > first_error
        mask = use_first_scale
        # breakpoint()
        # xg_scaled = torch.where(use_first_scale, first_scaled_xg, second_scaled_xg)

        # xg_lp = self.inlier_lpfp_quant(xg_scaled)
        mask_share_channel = 8
        mask_reshape = use_first_scale.view([-1, mask_share_channel, use_first_scale.shape[-2], use_first_scale.shape[-1]])
        mask_reshape_any = mask_reshape.any(dim=1, keepdim=True)
        # mask_reshape_any = mask_reshape.sum(dim=1, keepdim=True) >= 5
        mask = mask_reshape_any.repeat([1, mask_share_channel, 1, 1]).view(use_first_scale.shape)
        # _, max_idx = xg_abs.max(dim=-1, keepdim=True)
        # largest_mask = torch.zeros_like(xg_abs, dtype=torch.bool, device=xg_abs.device)
        # largest_mask.scatter_(-1, max_idx, True)
        # mask = torch.logical_or(mask, largest_mask)
        # breakpoint()

        xg = torch.where(mask, first_scaled_xg_q*first_scale, second_scaled_xg_q*second_scale)

        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg
        

    def q_naive(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        max_exp = torch.max(torch.floor(torch.log2(xg.abs())), dim=-1, keepdim=True).values # max exp in the block
        # threshold = 2**(max_exp+1) / (1.5*8)
        # threshold = xg.abs().max(dim=-1, keepdim=True).values/2
        # outlier_mask = xg.abs() >= threshold
        # _, topk_indices = xg.abs().topk(k=4, dim=-1)
        # topk_mask = torch.zeros_like(xg, dtype=torch.bool)
        # topk_mask.scatter_(-1, topk_indices, True)
        range_mask = xg.abs() >= xg.abs().max(dim=-1, keepdim=True).values/4
        # outlier_mask = torch.logical_and(topk_mask, range_mask)
        outlier_mask = range_mask
        # outlier_mask = torch.ones_like(xg, dtype=torch.bool)
        # print(outlier_mask.sum()/outlier_mask.numel())
        inliers = xg.clone()
        outliers = xg.clone()
        inliers[outlier_mask] = 0
        outliers[~outlier_mask] = 0
        
        _, _, inlier_shared_scale = self.inlier_observer(inliers, shared_exp_axes[0])
        # _, _, outlier_shared_scale = self.outlier_observer(outliers, shared_exp_axes[0])
        hp_max = 2**(2**(self.ebit)-1)*(1+self.mbit/2**(self.mbit)) # 12
        outlier_shared_scale = xg.abs().max(dim=-1, keepdim=True).values/4
        self.outlier_observer.shared_scale = outlier_shared_scale
        del inliers, outliers
        
        xg_scaled = torch.where(outlier_mask, xg / outlier_shared_scale, xg / inlier_shared_scale)

        xg_lp = self.inlier_lpfp_quant(xg_scaled)
        # xg_hp = self.outlier_lpfp_quant(xg_scaled)
        xg_hp = quant_element_int_non_zero(xg_scaled, 3)
        # xg_hp[xg_hp==0] = xg_scaled.sign()*1
        # xg_hp = xg_scaled
        # if (xg_hp[outlier_mask] == 0).sum() > 0:
        #     breakpoint()
        # assert (xg_hp[outlier_mask] == 0).sum() == 0

        # xg = self.lpfp_quant(xg_scaled)
        xg = torch.where(outlier_mask, xg_hp, xg_lp)

        xg = torch.where(outlier_mask, xg.mul(outlier_shared_scale), xg.mul(inlier_shared_scale))

        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPL2Quantizer(MXFPQuantizer):
    """
    MXShift quantization
    Inspired by Microexponent, use fine-grained shift
    """
    L2_SCHEME = ["PoT", "INT", "FP-1", "FP1", "FP-10", "FP-100", "Mix"]
    def __init__(
        self, 
        train_flag: bool = True, 
        unsigned: bool = False, 
        block_size:int=128, 
        ebit:int=2, mbit:int=3, 
        sc_ebit:int=5, sc_mbit:int=3, 
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
        super().__init__(train_flag, unsigned, block_size, ebit, mbit, sc_ebit, sc_mbit, per_tensor_scale, scale_allow_subnormal, use_round, use_ceil, use_quant_kernel)
        assert block_size % l2_block_size == 0, "block_size must be divisible by l2_block_size"
        assert l2_scheme in self.L2_SCHEME, f"Invalid L2 scheme: {l2_scheme}"
        self.keep_l2_scale = keep_l2_scale # keep l2 scale statistics, only activate this if you want to extract l2 scale statistics (e.g. for hardware verification)
        self.l2_block_size = l2_block_size  # block size for L2 quantization
        self.l2_sc_bit = l2_sc_bit  # shift bit for L2 quantization
        self.l2_scheme = l2_scheme  # scheme for L2 quantization
    
    def quant_scale(self, scale:torch.Tensor):
        scale_min = get_min_subnorm(self.sc_ebit, self.sc_mbit) if self.scale_allow_subnormal else get_min_norm(self.sc_ebit, self.sc_mbit)
        scale = scale.clip(min = scale_min, max = get_max_norm(self.sc_ebit, self.sc_mbit))
        scale = quant_element(scale, self.sc_ebit, self.sc_mbit, allow_subnormal=self.scale_allow_subnormal, use_ceil=self.use_ceil)
        return scale
    def q_max(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        xg = xg.contiguous() # [..., num_blocks, block_size]
        # maximum representable value with 2-level scaling, 1-level scaling is normal FP8, second level is shifting with self.l2_sc_bit
        max_elem = get_max_norm(ebit=self.ebit, mbit=self.mbit)
        scale = xg.abs().max(dim=-1, keepdim=True).values / max_elem # [..., num_blocks, 1]
        possible_scales = scale.unsqueeze(-1).repeat([1] * len(scale.shape) + [2**self.l2_sc_bit]) # [..., num_blocks, 1, 2**shift_bit]
        shift_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        shift_scale_range = torch.pow(2, shift_scale_range) / 2**(2**(self.l2_sc_bit)-1)
        while len(shift_scale_range.shape) < len(possible_scales.shape): shift_scale_range = shift_scale_range.unsqueeze(0)
        possible_scales = possible_scales * shift_scale_range # [..., num_blocks, 1, 2**shift_bit]
        possible_scales = possible_scales.unsqueeze(-2) # add one dimension for l2 [..., num_blocks, 1, 1, 2**shift_bit]

        # split micro blocks
        # reshape the last two dimensions to (..., num_blocks, block_size // l2_block_size, l2_block_size)
        xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size)
        use_second_scale = xgg.abs().max(dim=-1, keepdim=True).values < (xg.abs().max(dim=-1, keepdim=True).values/2)[..., None].expand_as(xgg.abs().max(dim=-1, keepdim=True).values)
        # expand xgg at last dimension to possible_scales.shape
        xgg = xgg.unsqueeze(-1).repeat([1] * len(xgg.shape) + [possible_scales.shape[-1]]) # [..., num_blocks, block_size // l2_block_size, l2_block_size, 2**shift_bit]
        # expand possible_scales to the same shape as xgg
        xgg_q = xgg / possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = self.lpfp_quant(xgg_q) # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = xgg_q * possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_dequant = torch.where(use_second_scale.expand(*xgg_q.shape[:-1]), xgg_q[..., 0], xgg_q[..., 1])

        # error = (xgg - xgg_q).abs().sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
        # _, min_error_idx = error.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
        # # expand min_error since a sub block shares the same scale
        # min_error_idx = min_error_idx.expand(*min_error_idx.shape[:-2], self.l2_block_size, min_error_idx.shape[-1])
        # xgg_dequant = xgg_q.gather(-1, min_error_idx).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
        xgg_dequant = xgg_dequant.view(*x.shape)

        return xgg_dequant

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        l2_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        l2_scale_pot = torch.pow(2, l2_scale_range)
        l2_scale_int = (l2_scale_range+1)
        l2_scale_fp = 1+l2_scale_range/(2**self.l2_sc_bit) # 1.XX
        l2_scale_fp1 = 1+(l2_scale_range+1)/(2**self.l2_sc_bit) # 1.XX
        l2_scale_fp_10 = 1+l2_scale_range/(2**(self.l2_sc_bit+1)) # 1.0X
        l2_scale_fp_100 = 1+l2_scale_range/(2**(self.l2_sc_bit+2)) # 1.00X
        if self.l2_scheme == "PoT":
            l2_scale_all = l2_scale_pot[..., None]
        elif self.l2_scheme == "INT":
            l2_scale_all = l2_scale_int[..., None]
        elif self.l2_scheme == "FP-1":
            l2_scale_all = l2_scale_fp[..., None]
        elif self.l2_scheme == "FP1":
            l2_scale_all = l2_scale_fp1[..., None]
        elif self.l2_scheme == "FP-10":
            l2_scale_all = l2_scale_fp_10[..., None]
        elif self.l2_scheme == "FP-100":
            l2_scale_all = l2_scale_fp_100[..., None]
        elif self.l2_scheme == "Mix":
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

    def q_de(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        xg = xg.contiguous() # [..., num_blocks, block_size]
        # maximum representable value with 2-level scaling, 1-level scaling is normal FP8, second level is shifting with self.l2_sc_bit
        # max_elem = get_max_norm(ebit=self.ebit, mbit=self.mbit)
        # scale = xg.abs().max(dim=-1, keepdim=True).values / max_elem # [..., num_blocks, 1]
        # scale = shared_scale
        possible_scales = self.shared_scale.unsqueeze(-1).repeat([1] * len(self.shared_scale.shape) + [2**self.l2_sc_bit]) # [..., num_blocks, 1, 2**shift_bit]
        shift_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        # PoT l2
        if self.l2_scheme == "PoT":
            shift_scale_range = torch.pow(2, shift_scale_range)
        # INT l2
        elif self.l2_scheme == "INT":
            shift_scale_range = (shift_scale_range+1)
        elif self.l2_scheme == "FP-1":
            shift_scale_range = 1+shift_scale_range/(2**self.l2_sc_bit) # 1.XX
        elif self.l2_scheme == "FP-10":
            shift_scale_range = 1+shift_scale_range/(2**(self.l2_sc_bit+1)) # 1.0X
        elif self.l2_scheme == "FP-100":
            shift_scale_range = 1+shift_scale_range/(2**(self.l2_sc_bit+2)) # 1.00X
        elif self.l2_scheme == "Mix":
            shift_scale_range_pot = torch.pow(2, shift_scale_range)
            shift_scale_range_int = (shift_scale_range+1)
            shift_scale_range_fp = 1+shift_scale_range/(2**self.l2_sc_bit) # 1.XX
            shift_scale_range_fp_10 = 1+shift_scale_range/(2**(self.l2_sc_bit+1)) # 1.0X
            shift_scale_range_fp_100 = 1+shift_scale_range/(2**(self.l2_sc_bit+2)) # 1.00X
            # shift_scale_range_all = torch.stack([shift_scale_range_pot, shift_scale_range_int, shift_scale_range_fp, shift_scale_range_fp_10, shift_scale_range_fp_100], dim=-1)
            # shift_scale_range_all = torch.stack([shift_scale_range_pot, shift_scale_range_int, shift_scale_range_fp], dim=-1)
            shift_scale_range_all = torch.stack([shift_scale_range_int, shift_scale_range_fp], dim=-1)
            possible_scales = possible_scales.unsqueeze(-1).repeat([1] * len(possible_scales.shape) + [shift_scale_range_all.shape[-1]])
            breakpoint()
            while len(shift_scale_range_all.shape) < len(possible_scales.shape): shift_scale_range_all = shift_scale_range_all.unsqueeze(0)
            possible_scales = possible_scales / shift_scale_range_all.max(dim=-2, keepdim=True).values
            possible_scales = possible_scales * shift_scale_range_all
            possible_scales = possible_scales.unsqueeze(-3) # add one dimension for l2 [..., num_blocks, 1, 1, 2**shift_bit, num_scale_scheme]
            # breakpoint()
            xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size) # [..., num_blocks, block_size // l2_block_size, l2_block_size]
            xgg_expand = xgg.unsqueeze(-1).unsqueeze(-1).repeat([1] * len(xgg.shape) + [*possible_scales.shape[-2:]]) # [..., num_blocks, block_size // l2_block_size, l2_block_size, 2**shift_bit, num_scale_scheme]
            xgg_q = xgg_expand / possible_scales
            xgg_q = self.lpfp_quant(xgg_q)
            xgg_q = xgg_q * possible_scales
            # first find propser l2 scale for each sub block and schemes
            error_in_each_scale_scheme = (xgg_expand - xgg_q).abs().pow(2).sum(dim=-3, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit, num_scale_scheme]
            _, min_error_idx_in_each_scale_scheme = error_in_each_scale_scheme.min(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 1, num_scale_scheme]
            min_error_idx_in_each_scale_scheme_expand = min_error_idx_in_each_scale_scheme.expand(*min_error_idx_in_each_scale_scheme.shape[:-3], self.l2_block_size, *min_error_idx_in_each_scale_scheme.shape[-2:])
            xgg_dequant_in_each_scale_scheme = xgg_q.gather(-2, min_error_idx_in_each_scale_scheme_expand).squeeze(-2) # [..., block_size // l2_block_size, l2_block_size, num_scale_scheme]
            # then find the best scheme for each sub block
            xgg_expand_scale_scheme = xgg[..., None].expand_as(xgg_dequant_in_each_scale_scheme)
            error_across_scale_scheme = (xgg_expand_scale_scheme - xgg_dequant_in_each_scale_scheme).abs().pow(2).sum(dim=(-2, -3), keepdim=True) # [..., ch/block_size, 1, 1, num_scale_scheme]
            _, min_error_idx_across_scale_scheme = error_across_scale_scheme.min(dim=-1, keepdim=True) # [..., 1, block_size // l2_block_size, l2_block_size, 1]
            min_error_idx_across_scale_scheme_expand = min_error_idx_across_scale_scheme.expand(*xgg.shape, 1)
            xgg_dequant = xgg_dequant_in_each_scale_scheme.gather(-1, min_error_idx_across_scale_scheme_expand).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
            xgg_dequant = xgg_dequant.view(*x.shape)
            # breakpoint()
            self.l2_scheme = min_error_idx_across_scale_scheme.squeeze() # (out_ch, in_ch/block_size)
            l2_scheme_expand = min_error_idx_across_scale_scheme[..., None].expand(*min_error_idx_in_each_scale_scheme.shape[:-1], -1)
            self.l2_scale = min_error_idx_in_each_scale_scheme.gather(-1, l2_scheme_expand).squeeze()


            
            return xgg_dequant
            

        else:
            raise ValueError(f"Invalid L2 scheme: {self.l2_scheme}")

        while len(shift_scale_range.shape) < len(possible_scales.shape): shift_scale_range = shift_scale_range.unsqueeze(0)
        possible_scales = possible_scales / shift_scale_range # [..., num_blocks, 1, 2**shift_bit]
        possible_scales = possible_scales.unsqueeze(-2) # add one dimension for l2 [..., num_blocks, 1, 1, 2**shift_bit]

        # split micro blocks
        # reshape the last two dimensions to (..., num_blocks, block_size // l2_block_size, l2_block_size)
        xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size)
        # expand xgg at last dimension to possible_scales.shape
        xgg = xgg.unsqueeze(-1).repeat([1] * len(xgg.shape) + [possible_scales.shape[-1]]) # [..., num_blocks, block_size // l2_block_size, l2_block_size, 2**shift_bit]
        # expand possible_scales to the same shape as xgg
        xgg_q = xgg / possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = self.lpfp_quant(xgg_q) # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = xgg_q * possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]

        error = (xgg - xgg_q).abs().sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
        _, min_error_idx = error.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
        # expand min_error since a sub block shares the same scale
        min_error_idx = min_error_idx.expand(*min_error_idx.shape[:-2], self.l2_block_size, min_error_idx.shape[-1])
        # breakpoint()
        xgg_dequant = xgg_q.gather(-1, min_error_idx).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
        xgg_dequant = xgg_dequant.view(*x.shape)

        return xgg_dequant

    def q_old(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        xg = xg.contiguous() # [..., num_blocks, block_size]
        # maximum representable value with 2-level scaling, 1-level scaling is normal FP8, second level is shifting with self.l2_sc_bit
        max_elem = get_max_norm(ebit=self.ebit, mbit=self.mbit)
        scale = xg.abs().max(dim=-1, keepdim=True).values / max_elem # [..., num_blocks, 1]
        possible_scales = scale.unsqueeze(-1).repeat([1] * len(scale.shape) + [2**self.l2_sc_bit]) # [..., num_blocks, 1, 2**shift_bit]
        shift_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        # PoT l2
        if self.l2_scheme == "PoT":
            shift_scale_range = torch.pow(2, shift_scale_range) / 2**(2**(self.l2_sc_bit)-1)
        # INT l2
        elif self.l2_scheme == "INT":
            shift_scale_range = (shift_scale_range+1) / 2**(self.l2_sc_bit)
        elif self.l2_scheme == "FP-1":
            shift_scale_range = shift_scale_range/(2**self.l2_sc_bit) + 1 # 1.XX
        elif self.l2_scheme == "FP-10":
            shift_scale_range = shift_scale_range/(2**(self.l2_sc_bit+1)) + 1 # 1.0X
        elif self.l2_scheme == "Mix":
            shift_scale_range_int = (shift_scale_range+1) / 2**(self.l2_sc_bit)
            shift_scale_range_fp = shift_scale_range/(2**self.l2_sc_bit) + 1 # 1.XX
            while len(shift_scale_range_int.shape) < len(possible_scales.shape): shift_scale_range_int = shift_scale_range_int.unsqueeze(0)
            while len(shift_scale_range_fp.shape) < len(possible_scales.shape): shift_scale_range_fp = shift_scale_range_fp.unsqueeze(0)
            possible_scales_int = (possible_scales * shift_scale_range_int).unsqueeze(-2)
            possible_scales_fp = (possible_scales * shift_scale_range_fp).unsqueeze(-2)
            # breakpoint()
            xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size)
            xgg_expand = xgg.unsqueeze(-1).repeat([1] * len(xgg.shape) + [possible_scales.shape[-1]])
            xgg_q_int = xgg_expand / possible_scales_int
            xgg_q_fp = xgg_expand / possible_scales_fp
            xgg_q_int = self.lpfp_quant(xgg_q_int)
            xgg_q_fp = self.lpfp_quant(xgg_q_fp)
            xgg_q_int = xgg_q_int * possible_scales_int
            xgg_q_fp = xgg_q_fp * possible_scales_fp

            error_int = (xgg_expand - xgg_q_int).abs().pow(2).sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
            error_fp = (xgg_expand - xgg_q_fp).abs().pow(2).sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
            _, min_error_idx_int = error_int.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
            _, min_error_idx_fp = error_fp.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
            # expand min_error since a sub block shares the same scale
            min_error_idx_int = min_error_idx_int.expand(*min_error_idx_int.shape[:-2], self.l2_block_size, min_error_idx_int.shape[-1])
            min_error_idx_fp = min_error_idx_fp.expand(*min_error_idx_fp.shape[:-2], self.l2_block_size, min_error_idx_fp.shape[-1])
            xgg_dequant_int = xgg_q_int.gather(-1, min_error_idx_int).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
            xgg_dequant_fp = xgg_q_fp.gather(-1, min_error_idx_fp).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]

            # sum over block
            int_error = (xgg - xgg_dequant_int).abs().pow(2).sum(dim=(-1, -2), keepdim=True) # [..., ch/block_size, 1, 1]
            fp_error = (xgg - xgg_dequant_fp).abs().pow(2).sum(dim=(-1, -2), keepdim=True) # [..., ch/block_size, 1, 1]
            use_int = int_error < fp_error
            xgg_dequant = torch.where(use_int, xgg_dequant_int, xgg_dequant_fp)
            xgg_dequant = xgg_dequant.view(*x.shape)

            return xgg_dequant

        else:
            raise ValueError(f"Invalid L2 scheme: {self.l2_scheme}")

        while len(shift_scale_range.shape) < len(possible_scales.shape): shift_scale_range = shift_scale_range.unsqueeze(0)
        possible_scales = possible_scales * shift_scale_range # [..., num_blocks, 1, 2**shift_bit]
        possible_scales = possible_scales.unsqueeze(-2) # add one dimension for l2 [..., num_blocks, 1, 1, 2**shift_bit]

        # split micro blocks
        # reshape the last two dimensions to (..., num_blocks, block_size // l2_block_size, l2_block_size)
        xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size)
        # expand xgg at last dimension to possible_scales.shape
        xgg = xgg.unsqueeze(-1).repeat([1] * len(xgg.shape) + [possible_scales.shape[-1]]) # [..., num_blocks, block_size // l2_block_size, l2_block_size, 2**shift_bit]
        # expand possible_scales to the same shape as xgg
        xgg_q = xgg / possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = self.lpfp_quant(xgg_q) # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = xgg_q * possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]

        error = (xgg - xgg_q).abs().sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
        _, min_error_idx = error.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
        # expand min_error since a sub block shares the same scale
        min_error_idx = min_error_idx.expand(*min_error_idx.shape[:-2], self.l2_block_size, min_error_idx.shape[-1])
        xgg_dequant = xgg_q.gather(-1, min_error_idx).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
        xgg_dequant = xgg_dequant.view(*x.shape)

        return xgg_dequant


class MXINTShiftQuantizer(MXINTQuantizer):
    """
    MXINTShift quantization
    Inspired by Microexponent, use fine-grained shift
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, sc_ebit: int=8, sc_mbit: int=0, block_size:int=32, l2_block_size:int=4, l2_sc_bit:int=1, per_tensor_scale: bool = False, scale_allow_subnormal: bool = False, use_round: bool=False):
        super().__init__(nbit, train_flag, unsigned, sc_ebit, sc_mbit, block_size, per_tensor_scale, scale_allow_subnormal, use_round)
        assert block_size % l2_block_size == 0, "block_size must be divisible by l2_block_size"
        self.l2_block_size = l2_block_size  # block size for L2 quantization
        self.l2_sc_bit = l2_sc_bit  # shift bit for L2 quantization

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.per_tensor_scale: # scale xg down to match (element range * scale range)
            ele_max = get_max_norm(ebit=self.ebit, mbit=self.mbit)
            shared_scale_max = get_max_norm(ebit=self.sc_ebit, mbit=self.sc_mbit)
            x_max = xg.abs().max().item() # I think it's possible to calibrate the max through calibration set
            if x_max == 0: # if all elements are 0, don't need tensor scale
                tensor_scale = 1.0
            else:
                tensor_scale = x_max/(ele_max*shared_scale_max)
            xg = xg / tensor_scale
        else:
            tensor_scale = 1.0

        if self.train_flag:
            scale, zero_point, shared_scale = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_scale.data = shared_scale

        xg = xg.contiguous() # [..., num_blocks, block_size]
        # maximum representable value with 2-level scaling, 1-level scaling is normal FP8, second level is shifting with self.l2_sc_bit
        max_elem = get_max_norm(ebit=0, mbit=self.nbit)
        scale = xg.abs().max(dim=-1, keepdim=True).values / max_elem # [..., num_blocks, 1]
        possible_scales = scale.unsqueeze(-1).repeat([1] * len(scale.shape) + [2**self.l2_sc_bit]) # [..., num_blocks, 1, 2**shift_bit]
        shift_scale_range = torch.arange(2**self.l2_sc_bit, device=xg.device, dtype=xg.dtype)
        shift_scale_range = torch.pow(2, shift_scale_range) / 2**(2**(self.l2_sc_bit)-1)
        while len(shift_scale_range.shape) < len(possible_scales.shape): shift_scale_range = shift_scale_range.unsqueeze(0)
        possible_scales = possible_scales * shift_scale_range # [..., num_blocks, 1, 2**shift_bit]
        possible_scales = possible_scales.unsqueeze(-2) # add one dimension for l2 [..., num_blocks, 1, 1, 2**shift_bit]

        # split micro blocks
        # reshape the last two dimensions to (..., num_blocks, block_size // l2_block_size, l2_block_size)
        xgg = xg.view(*xg.shape[:-1], -1, self.l2_block_size)
        # expand xgg at last dimension to possible_scales.shape
        xgg = xgg.unsqueeze(-1).repeat([1] * len(xgg.shape) + [possible_scales.shape[-1]]) # [..., num_blocks, block_size // l2_block_size, l2_block_size, 2**shift_bit]
        # expand possible_scales to the same shape as xgg
        xgg_q = xgg / possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = quant_element_int(xgg_q, self.nbit) # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]
        xgg_q = xgg_q * possible_scales # [..., block_size // l2_block_size, l2_block_size, 2**shift_bit]

        error = (xgg - xgg_q).abs().sum(dim=-2, keepdim=True) # [..., block_size // l2_block_size, 1, 2**shift_bit]
        _, min_error_idx = error.min(dim=-1, keepdim=True) # [..., block_size // l2_block_size, 1, 1]
        # expand min_error since a sub block shares the same scale
        min_error_idx = min_error_idx.expand(*min_error_idx.shape[:-2], self.l2_block_size, min_error_idx.shape[-1])
        xgg_dequant = xgg_q.gather(-1, min_error_idx).squeeze(-1) # [..., block_size // l2_block_size, l2_block_size]
        xgg_dequant = xgg_dequant.view(*x.shape)

        return xgg_dequant    

class FP8Quantizer(_QBase):
    """
        FP8 quantization
        Use FP16 for scaling, and specified FP8 format for target precision
    """
    def __init__(self, train_flag: bool = True, unsigned: bool = False, ebit:int=2, mbit:int=5):
        super().__init__(8, train_flag, unsigned)
        self.ebit = ebit
        self.mbit = mbit
        self.lpfp_quant = GLOBAL_FP_QUANT_CACHE.get(FPQuantConfig(ebit=self.ebit, mbit=self.mbit, allow_subnormal=True, detailed_output=False))
    def q(self, x:torch.Tensor):
        max_per_channel = x.abs().max(dim=-1, keepdim=True).values
        max_value = get_max_norm(ebit=self.ebit, mbit=self.mbit)
        self.scale = max_per_channel / max_value
        x = x / self.scale
        x = self.lpfp_quant(x)
        x = x * self.scale
        return x