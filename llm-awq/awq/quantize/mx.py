'''
Stand-alone block quantizers
Migrate from mx.py
Support following scaling scheme 
    -conventional MX (power of 2 scale)
    -NVidia standard NV-block quantization (FP scale with per-tensor scale),
    -Our FP8-scale quantization (FP scale with per-tensor scale),
For MX quantization, use sc_ebit=0, sc_mbit=8
For NV quantization, use sc_ebit=4, sc_mbit=3, per_tensor_scale=True
For Our FP8-scale quantization, use sc_ebit=5, sc_mbit=3
'''

import torch
FP32_EXPONENT_BIAS = 127
FP16_EXPONENT_BIAS = 15
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
FP16_MIN_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 1)
FP32_MAX_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 2**10)
FP16_MAX_NORMAL = 2 ** (-FP16_EXPONENT_BIAS + 2**5)

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
    else:
        emin = 2-2**(ebit-1)
        return 2**emin

def get_min_subnorm(ebit: int, mbit: int):
    """
    Get the minimum subnormal number of specific bit width
    """
    emin = 2-2**(ebit-1)
    return 2**emin * 2**(-mbit)

def quant_element(x:torch.Tensor, ebit: int, mbit: int, allow_subnormal: bool=True, detailed_output: bool=False, use_ceil: bool=False):
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

    # mant = torch.floor(torch.abs(mant) + 0.55) if use_ceil else torch.floor(torch.abs(mant) + 0.5)
    mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.floor(torch.abs(mant) + 0.5) # use rounding as default (don't use +0.5 with floor for float16)
    # mant = torch.ceil(torch.abs(mant)) if use_ceil else torch.round(torch.abs(mant)+1e-5) # use rounding as default, add 1e-5 to avoid 0.5 round to 0
    mant_max = 2**mbit - 1
    overflow_mask = (mant > mant_max) & (exp != emax)
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

    if detailed_output:
        exp_bias = 2**(ebit-1) - 1
        exp = exp+exp_bias
        zero_mask_for_exp = x_is_zero | underflow_mask | x_is_subnormal
        exp = torch.where(zero_mask_for_exp, 0, exp)
        return x_q, exp, mant
    else:
        return x_q


class MXINTQuantizer():
    def __init__(self, nbit: int, sc_ebit: int=8, sc_mbit: int=0, block_size:int=32, scale_allow_subnormal: bool = False, use_round: bool=False):
        """
        MXINT quantization
        Parameters:
        - nbit: element bit width (conventional MX only support INT4 & INT8)
        - sc_ebit: shared scale exponent bit width
        - sc_mbit: shared scale mantissa bit width
            - Conventional MX: sc_ebit = 8, sc_mbit = 0
            - NVidia standard NV-block quantization: sc_ebit = 4, sc_mbit = 3
        - block_size: block size for quantization (32 for conventional MX, 16 for NV-block quantization)
        - scale_allow_subnormal: Whether to allow subnormal in scale (only in NV-block quantization)
        - use_round: Whether to use rounding for scale (only in MX)
        """
        self.nbit = nbit   
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round

        # low precision range
        self.qlb = -2**(nbit-1)
        self.qub = 2**(nbit-1) - 1
        # self.observer = MXINTObserver(nbit=self.nbit, sc_ebit=self.sc_ebit, sc_mbit=self.sc_mbit, unsigned=self.unsigned, scale_allow_subnormal=scale_allow_subnormal, use_round=use_round)
        
    def reshape(self, x:torch.Tensor):
        # reshape to blocks along the last dimension
        x, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, [-1], block_size=self.block_size
        )
        return x, axes, orig_shape, padded_shape
    
    def get_shared_scale(self, x:torch.Tensor, axis:int=-1):
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
        if (shared_scale == 0).sum() > 0:
            indices = (shared_scale == 0).nonzero()
            breakpoint()
        return shared_scale

    def get_q_params(self, x:torch.Tensor, axis:int=-1):
        if self.sc_mbit == 0: # conventional MX, shift scaling
            scale = torch.tensor(2**(self.nbit-2), device=x.device) # implicit scale is handled here
        else:
            scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_scale = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes] # +1 for the new group dimension

        scale, zero_point, shared_scale = self.get_q_params(xg, shared_exp_axes[0])

        zero_mask = xg==0
        xg = xg / shared_scale
        xg[zero_mask] = 0
        xg = xg * scale

        # round to nearest integer
        xg = torch.sign(xg) *  torch.floor(torch.abs(xg) + 0.5)
        
        xg = xg.clamp(self.qlb, self.qub)
        
        # rescale back
        xg = xg.div(scale)
        xg = xg.mul(shared_scale).to(x.dtype)

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class MXFPQuantizer():
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        AMXFP quantization
        (Ref: Lee, Janghwan, et al. "Amxfp4: Taming activation outliers with asymmetric microscaling floating-point for 4-bit llm inference." arXiv preprint arXiv:2411.09909 (2024).)
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        self.ebit = ebit
        self.mbit = mbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
        self.use_ceil = use_ceil
        
    def reshape(self, x:torch.Tensor):
        # reshape to blocks along the last dimension
        x, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, [-1], block_size=self.block_size
        )
        return x, axes, orig_shape, padded_shape
   
    def get_shared_scale(self, x:torch.Tensor, axis=-1) -> torch.Tensor:
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

    def get_q_params(self, x:torch.Tensor, axis:int=-1):
        scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_scale = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        _, _, shared_scale = self.get_q_params(xg, shared_exp_axes[0])
        
        xg = xg / shared_scale

        xg = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=False)

        xg = xg.mul(shared_scale)

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg

class AMXFPQuantizer():
    def __init__(self, train_flag: bool = True, unsigned: bool = False, block_size:int=32, ebit:int=2, mbit:int=3, sc_ebit:int=0, sc_mbit:int=8, per_tensor_scale: bool = False, scale_allow_subnormal: bool = True, use_round: bool=False, use_ceil: bool=False):
        """
        AMXFP quantization
        (Ref: Lee, Janghwan, et al. "Amxfp4: Taming activation outliers with asymmetric microscaling floating-point for 4-bit llm inference." arXiv preprint arXiv:2411.09909 (2024).)
        ebit: element exponent bit width
        mbit: element mantissa bit width
        sc_ebit: shared scale exponent bit width
        sc_mbit: shared scale mantissa bit width
        """
        self.ebit = ebit
        self.mbit = mbit
        self.sc_ebit = sc_ebit
        self.sc_mbit = sc_mbit
        self.block_size = block_size
        self.per_tensor_scale = per_tensor_scale
        self.scale_allow_subnormal = scale_allow_subnormal
        self.use_round = use_round
        self.use_ceil = use_ceil
        
    def reshape(self, x:torch.Tensor):
        # reshape to blocks along the last dimension
        x, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, [-1], block_size=self.block_size
        )
        return x, axes, orig_shape, padded_shape
   
    def get_shared_scale(self, x:torch.Tensor, axis=-1) -> torch.Tensor:
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

    def get_q_params(self, x:torch.Tensor, axis:int=-1):
        scale = torch.tensor(1, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_scale = self.get_shared_scale(x, axis=axis)
        return scale, zero_point, shared_scale

    def q(self, x:torch.Tensor):
        # x_shape = x.shape
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        pos_mask = xg > 0
        only_pos_xg = xg.clone()
        only_neg_xg = xg.clone()
        only_pos_xg[~pos_mask] = 0
        only_neg_xg[pos_mask] = 0
        _, _, pos_shared_scale = self.get_q_params(only_pos_xg, shared_exp_axes[0])
        _, _, neg_shared_scale = self.get_q_params(only_neg_xg, shared_exp_axes[0])
        del only_pos_xg, only_neg_xg
        
        xg = torch.where(pos_mask, xg / pos_shared_scale, xg / neg_shared_scale)

        xg = quant_element(xg, self.ebit, self.mbit, allow_subnormal=True, detailed_output=False)

        xg = torch.where(pos_mask, xg.mul(pos_shared_scale), xg.mul(neg_shared_scale))

        if xg.isnan().sum() > 0 or xg.isinf().sum()>0:
            nan_indices = torch.nonzero(torch.isnan(xg), as_tuple=False)
            inf_indices = torch.nonzero(torch.isinf(xg), as_tuple=False)
            if xg.isnan().sum() > 0: breakpoint()
            # import pdb;pdb.set_trace()
        # reshape tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)
        return xg


if __name__ == "__main__":
    t = torch.randn(64, 64)
    # MXFP8 quantizer example, quantize to E3M4 format with MX quantization
    mxfp8_quantizer = MXFPQuantizer(ebit=3, mbit=4, sc_ebit=8, sc_mbit=0, block_size=32, use_round=False, use_ceil=False)
    mxfp4_quantizer = MXFPQuantizer(ebit=2, mbit=1, sc_ebit=8, sc_mbit=0, block_size=32, use_round=False, use_ceil=False)
    mxint4_quantizer = MXINTQuantizer(nbit=4, sc_ebit=8, sc_mbit=0, block_size=32, use_round=False, use_ceil=False)
    # NVFP quantizer
    nvfp4_quantizer = MXFPQuantizer(ebit=2, mbit=1, sc_ebit=5, sc_mbit=3, block_size=16, use_round=False, use_ceil=False)
    q_t = mxfp8_quantizer.q(t)
    # q_t = mxint4_quantizer.q(t)
    # q_t = nvfp4_quantizer.q(t)
    print("original tensor:")
    print(t)
    print("quantized tensor:")
    print(q_t)
    breakpoint()
