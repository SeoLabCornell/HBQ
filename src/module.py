import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Callable

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import eager_attention_forward, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2Config
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.activations import ACT2FN

from src.quantizer import _QBase

class _QBaseLinear(nn.Linear):
    r"""Basic low precision linear layer

    Inherited from the base nn.Linear layer.
    
    Args:
    wq (_QBase): Weight quantizer. 
    aq (_QBase): Activation quantizer.
    ops: Customized GeMM operator
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, ops=None):
        super(_QBaseLinear, self).__init__(in_features, out_features, bias)

        # quantizer
        self.wq = _QBase()
        self.aq = _QBase()
        self.yq = _QBase()

        # Customized GeMM operator
        self.ops = ops
    
    def forward(self, x:torch.Tensor):
        # wq = self.wq(self.weight) # already statically quantized in PTQ
        wq = self.weight
        xq = self.aq(x)

        if self.ops == None:
            y = F.linear(xq, wq, self.bias)
        else:
            y = self.ops(xq, wq, self.bias)

        if y.isnan().sum() > 0 or y.isinf().sum()>0:
            inf_indices = torch.nonzero(y.isinf(), as_tuple = False)
            nan_indices = torch.nonzero(y.isnan(), as_tuple = False)
            import pdb; pdb.set_trace()
        out = self.yq(y)
        return out

class QLlamaAttention(LlamaAttention):
    """
    Llama Attention with Low precision operations
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, dtype=torch.float16, quant_attn_wgt=False, quant_post_rope_q=False, quant_k_cache=False, quant_v_cache=False):
        super().__init__(config, layer_idx)

        self.q_proj = _QBaseLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.k_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.v_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.o_proj = _QBaseLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias).to(torch.float16)

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.quant_attn_wgt = quant_attn_wgt
        self.quant_post_rope_q = quant_post_rope_q
        self.quant_k_cache = quant_k_cache
        self.quant_v_cache = quant_v_cache
        self.kv_quantizer = None
        self.attn_wgt_quantizer = None

    def manual_sdpa(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)
        if self.quant_attn_wgt: # quantize attn weight (after softmax)
            attn_weight = self.attn_wgt_quantizer.q(attn_weight)
        return attn_weight @ value, attn_weight

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.quant_post_rope_q:
            query_states = self.q_proj.aq.q(query_states)
        if self.quant_k_cache: # post-RoPE quantize K Cache
            key_states = self.kv_quantizer.q(key_states)
        if self.quant_v_cache:
            value_states = self.kv_quantizer.q(value_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output, attn_weight = self.manual_sdpa(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous() # [B,T,H,D] -> [B,T,H*D]
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weight

class QQwen2Attention(Qwen2Attention):
    """
        Code adapted from Qwen2Attention
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = _QBaseLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = _QBaseLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.quant_attn_wgt = False
        self.quant_post_rope_q = False
        self.quant_k_cache = False
        self.quant_v_cache = False
        self.kv_quantizer = None
        self.attn_wgt_quantizer = None

    def manual_spda( # this is working
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Python reimplementation of torch.nn.functional.scaled_dot_product_attention
        with the same contract as your original function, but fully explicit so
        that you can quantize attention weights before the V @ S GEMM.

        Expected shapes (same as SDPA path in HF):
        query, key, value: [B, num_heads, L, D]
        attention_mask: None or broadcastable to [B, num_heads, L, S]
            - typically [B, 1, L, S] additive mask (0 or -inf-like)
        """

        # --- GQA expansion (same as before) ---
        if hasattr(self, "num_key_value_groups"):
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)

        # HF-style mask slicing: truncate to current key length
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # Contiguity issues for some SDPA backends → keep this anyway
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        B, H, L, D = query.shape
        S = key.shape[-2]

        # --- determine scale and causal flag (match your logic) ---
        if scaling is None:
            scaling = 1.0 / math.sqrt(D)

        if is_causal is None:
            is_causal = (causal_mask is None) and (L > 1)

        # --- upcast to fp32 for matmul + softmax (like SDPA/Flash) ---
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        v = value.to(torch.float32)

        # --- attention logits: [B, H, L, S] ---
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scaling  # q @ k^T

        # --- apply mask: additive or boolean ---
        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                # True = keep, False = mask
                neg_inf = torch.finfo(attn_logits.dtype).min
                attn_logits = attn_logits.masked_fill(~causal_mask, neg_inf)
            else:
                # additive mask (0 or large negative)
                attn_logits = attn_logits + causal_mask

        # --- causal mask if no attention_mask was given ---
        if is_causal and causal_mask is None:
            # [L, S] lower-triangular
            causal = torch.ones(
                (L, S), dtype=torch.bool, device=attn_logits.device
            ).tril()
            neg_inf = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~causal, neg_inf)

        # --- softmax over keys dimension ---
        attn_weights = F.softmax(attn_logits, dim=-1)

        # --- dropout on attention weights (if training) ---
        if dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=dropout)

        # ====== YOUR QUANTIZATION HOOK LIVES HERE ======
        # Example:
        if getattr(self, "quant_attn_wgt", False):
            attn_weights = self.attn_wgt_quantizer.q(attn_weights)
        # ===============================================

        # --- attention output: [B, H, L, D] ---
        attn_output = torch.matmul(attn_weights, v)

        # cast back to original dtype for downstream layers
        attn_output = attn_output.to(query.dtype)

        # keep your original API: transpose out of [B, H, L, D] here or outside?
        # In your original function you did it *after* SDPA, so we do it here:
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, H, D]

        return attn_output, None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.quant_post_rope_q:
            query_states = self.q_proj.aq.q(query_states)
        # if self.quant_k_cache: # post-RoPE quantize K Cache
        #     key_states = self.kv_quantizer.q(key_states)
        # if self.quant_v_cache:
        #     value_states = self.kv_quantizer.q(value_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                pass
                # logger.warning_once(
                #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                # )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = self.manual_spda(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     sliding_window=sliding_window,  # main diff with Llama
        #     **kwargs,
        # )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class QLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = _QBaseLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.quantizer_after_act_fn = None

    def forward(self, x):
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        if self.quantizer_after_act_fn is not None:
            hadamard_output = self.quantizer_after_act_fn.q(self.act_fn(gate_proj)) * up_proj
        else:
            hadamard_output = self.act_fn(gate_proj) * up_proj

        down_proj = self.down_proj(hadamard_output)

        return down_proj

class QQwen2MLP(nn.Module):
    """
    Code adapted from Qwen2MLP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = _QBaseLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.quantizer_after_act_fn = None

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.quantizer_after_act_fn is not None:
            hadamard_output = self.quantizer_after_act_fn.q(self.act_fn(gate)) * up
        else:
            hadamard_output = self.act_fn(gate) * up
        down_proj = self.down_proj(hadamard_output)
        return down_proj

