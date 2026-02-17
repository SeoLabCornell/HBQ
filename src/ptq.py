"""
Post-training quantization of language models
"""
import os
import torch
from tqdm import tqdm
from typing import Dict, Type, Union

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from src.quantizer import MXFPQuantizer, MXINTQuantizer, HBQQuantizer
from src.module import QLlamaAttention, QQwen2Attention, QLlamaMLP, QQwen2MLP, _QBaseLinear, _QBase
from src.ops import FP16AccumMatMul

class PTQ:
    """
    Apply block quantization
    Support different quantization methods:
    - MX/NV FP/INT
    """
    def __init__(self,quant_config, model, tokenizer, logger):
        """
        MX-PoT: sc_ebit = 0, sc_mbit = 8 (conventional MX)
        NVFP: sc_ebit = 4, sc_mbit = 3
        """
        self.quant_config = quant_config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

        # quantizer type
        self.xqtype = quant_config.get("quantization", {})["xqtype"]
        self.wqtype = quant_config.get("quantization", {})["wqtype"]

        # MX/NVFP format
        self.a_ebit = quant_config.get("block_quant", {}).get("act_ebit", 0)
        self.a_mbit = quant_config.get("block_quant", {}).get("act_mbit", 0)
        self.a_sc_ebit = quant_config.get("block_quant", {}).get("act_sc_ebit", 0)
        self.a_sc_mbit = quant_config.get("block_quant", {}).get("act_sc_mbit", 8)
        self.a_block_size = quant_config.get("block_quant", {}).get("act_block_size", 32)
        self.a_per_tensor_scale = quant_config.get("block_quant", {}).get("act_per_tensor_scale", False)
        self.w_ebit = quant_config.get("block_quant", {}).get("wgt_ebit", 0)
        self.w_mbit = quant_config.get("block_quant", {}).get("wgt_mbit", 0)
        self.w_sc_ebit = quant_config.get("block_quant", {}).get("wgt_sc_ebit", 0)
        self.w_sc_mbit = quant_config.get("block_quant", {}).get("wgt_sc_mbit", 8)
        self.w_block_size = quant_config.get("block_quant", {}).get("wgt_block_size", 32)
        self.w_per_tensor_scale = quant_config.get("block_quant", {}).get("wgt_per_tensor_scale", False)
        self.scale_allow_subnormal = quant_config.get("block_quant", {}).get("scale_allow_subnormal", True)

        # MX format
        self.qk_mx_round = quant_config.get("block_quant", {}).get("qk_mx_round", False)

        # NV format
        self.act_scale_use_ceil = quant_config.get("block_quant", {}).get("act_scale_use_ceil", False)
        self.wgt_scale_use_ceil = quant_config.get("block_quant", {}).get("wgt_scale_use_ceil", False)

        # HBQ quantization
        self.wgt_l2_sc_bit = quant_config.get("block_quant", {}).get("wgt_l2_sc_bit", 1)
        self.wgt_l2_block_size = quant_config.get("block_quant", {}).get("wgt_l2_block_size", 4)
        self.wgt_l2_scheme = quant_config.get("block_quant", {}).get("wgt_l2_scheme", "PoT")
        self.act_l2_sc_bit = quant_config.get("block_quant", {}).get("act_l2_sc_bit", 1)
        self.act_l2_block_size = quant_config.get("block_quant", {}).get("act_l2_block_size", 4)
        self.act_l2_scheme = quant_config.get("block_quant", {}).get("act_l2_scheme", "PoT")

        # model config
        self.quant_before_hadamard = quant_config.get("quantization", {}).get("quant_before_hadamard", False)
        self.quant_before_silu = quant_config.get("quantization", {}).get("quant_before_silu", False)
        self.quant_before_rms_norm = quant_config.get("quantization", {}).get("quant_before_rms_norm", False)
        self.quant_embedding_layer = quant_config.get("quantization", {}).get("quant_embedding_layer", False)
        self.quant_output_layer = quant_config.get("quantization", {}).get("quant_output_layer", False)
        self.quant_attn_wgt = quant_config.get("quantization", {}).get("quant_attn_wgt", False)
        self.quant_post_rope_qk = quant_config.get("quantization", {}).get("quant_post_rope_qk", False) # deprecated, don't use this
        self.quant_post_rope_q = quant_config.get("quantization", {}).get("quant_post_rope_q", False) # quantize Q after RoPE, enable low precision computation for HW
        self.quant_4b_kv = quant_config.get("quantization", {}).get("quant_4b_kv", False) # if True, use KV4, otherwise use act quantizer
        self.quant_k_cache = quant_config.get("quantization", {}).get("quant_k_cache", False)
        self.quant_v_cache = quant_config.get("quantization", {}).get("quant_v_cache", False)
        self.quant_attn_wgt_use_quant_kernel = quant_config.get("quantization", {}).get("quant_attn_wgt_use_quant_kernel", True)

        # Low precision accumulation
        self.accumulation_type = quant_config.get("accumulation", {}).get("type", "fp32")
        self.accumulation_group_size = quant_config.get("accumulation", {}).get("group_size", 128)
        self.accumulation_quant_block_size = quant_config.get("accumulation", {}).get("quant_block_size", 16)

    def map_wgt_quantizer(self, use_round=False):
        if self.wqtype == "mxfp":
            return MXFPQuantizer(
                ebit=self.w_ebit,
                mbit=self.w_mbit,
                sc_ebit=self.w_sc_ebit,
                sc_mbit=self.w_sc_mbit,
                block_size=self.w_block_size,
                per_tensor_scale=self.w_per_tensor_scale, 
                scale_allow_subnormal=self.scale_allow_subnormal,
                use_round=use_round,
                use_ceil=self.wgt_scale_use_ceil
                )
        elif self.wqtype == "mxint":
            return MXINTQuantizer(
                nbit=self.w_mbit,
                sc_ebit=self.w_sc_ebit,
                sc_mbit=self.w_sc_mbit,
                block_size=self.w_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal
                )
        elif self.wqtype == "hbq":
            return HBQQuantizer(
                ebit=self.w_ebit,
                mbit=self.w_mbit,
                sc_ebit=self.w_sc_ebit,
                sc_mbit=self.w_sc_mbit,
                block_size=self.w_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal,
                l2_block_size=self.wgt_l2_block_size,
                l2_sc_bit=self.wgt_l2_sc_bit,
                l2_scheme=self.wgt_l2_scheme
            )
        elif self.wqtype == "none":
            return _QBase()
        else:
            assert False, f"Invalid weight quantizer type: {self.wqtype}"

    def map_act_quantizer(self, use_round=False, use_quant_kernel=True):
        if self.xqtype == "mxfp":
            return MXFPQuantizer(
                ebit=self.a_ebit,
                mbit=self.a_mbit,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                per_tensor_scale=self.a_per_tensor_scale,
                scale_allow_subnormal=self.scale_allow_subnormal,
                use_round=use_round,
                use_ceil=self.act_scale_use_ceil
                )
        elif self.xqtype == "mxint":
            return MXINTQuantizer(
                nbit=self.a_mbit,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal
                )
        elif self.xqtype == "hbq":
            return HBQQuantizer(
                ebit=self.a_ebit,
                mbit=self.a_mbit,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal,
                l2_block_size=self.act_l2_block_size,
                l2_sc_bit=self.act_l2_sc_bit,
                l2_scheme=self.act_l2_scheme
            )
        elif self.xqtype == "none":
            return _QBase()
        else:
            assert False, f"Invalid activation quantizer type: {self.xqtype}"

    def map_4b_kv_quantizer(self, use_round=False):
        '''
        Use 4-bit quantization for KV cache
        i.e. The quantizer is the same as the activation quantizer
        but the precision is 4-bit regardless of the activation bitwidth
        '''
        if self.xqtype == "mxfp":
            return MXFPQuantizer(
                ebit=2,
                mbit=1,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                per_tensor_scale=self.a_per_tensor_scale,
                scale_allow_subnormal=self.scale_allow_subnormal,
                use_round=use_round,
                use_ceil=self.act_scale_use_ceil
                )
        elif self.xqtype == "mxint":
            return MXINTQuantizer(
                nbit=4,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal
                )
        elif self.xqtype == "hbq":
            return HBQQuantizer(
                ebit=2,
                mbit=1,
                sc_ebit=self.a_sc_ebit,
                sc_mbit=self.a_sc_mbit,
                block_size=self.a_block_size,
                scale_allow_subnormal=self.scale_allow_subnormal,
                l2_block_size=self.act_l2_block_size,
                l2_sc_bit=self.act_l2_sc_bit,
                l2_scheme=self.act_l2_scheme
            )
        elif self.xqtype == "none":
            return _QBase()
        else:
            assert False, f"Invalid activation quantizer type: {self.aqtype}"

    def q_attn(self, module:Union[QLlamaAttention, QQwen2Attention]):
        qkv = [module.q_proj, module.k_proj, module.v_proj]

        # quantize attention weight (after softmax)
        module.quant_attn_wgt = self.quant_attn_wgt
        module.quant_post_rope_q = self.quant_post_rope_q
        module.quant_k_cache = self.quant_k_cache
        module.quant_v_cache = self.quant_v_cache
        if self.quant_4b_kv: # use 4-bit for KV cache instead of act precision, the quantizer is the same as the activation quantizer
            module.kv_quantizer = self.map_4b_kv_quantizer(use_round=self.qk_mx_round)
        else:
            module.kv_quantizer = self.map_act_quantizer(use_round=self.qk_mx_round)
        if self.quant_attn_wgt:
            module.attn_wgt_quantizer = self.map_act_quantizer(use_round=self.qk_mx_round, use_quant_kernel=self.quant_attn_wgt_use_quant_kernel)
        else:
            module.attn_wgt_quantizer = None
        
        for proj in qkv:
            if proj == module.q_proj:
                proj.wq = self.map_wgt_quantizer(use_round=self.qk_mx_round)
                proj.aq = self.map_act_quantizer(use_round=self.qk_mx_round)
            elif proj == module.k_proj:
                proj.wq = self.map_wgt_quantizer(use_round=self.qk_mx_round)
                proj.aq = self.map_act_quantizer(use_round=self.qk_mx_round)
            elif proj == module.v_proj:
                proj.wq = self.map_wgt_quantizer()
                proj.aq = self.map_act_quantizer()
        
        module.o_proj.wq = self.map_wgt_quantizer()
        module.o_proj.aq = self.map_act_quantizer()
        if self.quant_before_rms_norm:
            module.o_proj.yq = self.map_act_quantizer() # RMS-norm trailed

    def q_mlp(self, module:Union[QLlamaMLP, QQwen2MLP], name:str=None, scales:Dict=None):
        module.up_proj.wq = self.map_wgt_quantizer()
        module.up_proj.aq = self.map_act_quantizer()
        if self.quant_before_hadamard:
            module.up_proj.yq = self.map_act_quantizer() # Hadamard trailed
            module.quantizer_after_act_fn = self.map_act_quantizer(None) # Hadamard trailed

        module.gate_proj.wq = self.map_wgt_quantizer()
        module.gate_proj.aq = self.map_act_quantizer()
        if self.quant_before_silu:
            module.gate_proj.yq = self.map_act_quantizer() # Silu trailed
        
        module.down_proj.wq = self.map_wgt_quantizer()
        module.down_proj.aq = self.map_act_quantizer()
        if self.quant_before_rms_norm:
            module.down_proj.yq = self.map_act_quantizer() # RMS-norm trailed

    @torch.no_grad
    def inject_quantizers(self, act_scales:Dict=None):
        for n,m in self.model.named_modules():
            if isinstance(m, (LlamaDecoderLayer, Qwen2DecoderLayer)):
                self.q_attn(m.self_attn)
                self.q_mlp(m.mlp)

    @torch.no_grad
    def static_quant_wgt(self):
        for n,m in self.model.named_modules():
            if isinstance(m, (LlamaDecoderLayer, Qwen2DecoderLayer)):

                m.self_attn.q_proj.weight.data = m.self_attn.q_proj.wq(m.self_attn.q_proj.weight)
                m.self_attn.k_proj.weight.data = m.self_attn.k_proj.wq(m.self_attn.k_proj.weight)
                m.self_attn.v_proj.weight.data = m.self_attn.v_proj.wq(m.self_attn.v_proj.weight)
                m.self_attn.o_proj.weight.data = m.self_attn.o_proj.wq(m.self_attn.o_proj.weight)
                m.mlp.gate_proj.weight.data = m.mlp.gate_proj.wq(m.mlp.gate_proj.weight)
                m.mlp.up_proj.weight.data = m.mlp.up_proj.wq(m.mlp.up_proj.weight)
                m.mlp.down_proj.weight.data = m.mlp.down_proj.wq(m.mlp.down_proj.weight)

        if self.quant_output_layer:
            self.model.lm_head.proj.weight.data = self.model.lm_head.proj.wq(self.model.lm_head.proj.weight)

    @torch.inference_mode()
    def replace_accum_kernel(self):
        if self.accumulation_type == "fp32":
            self.logger.info("Original accumulation kernel is used")
            return
        elif self.accumulation_type == "fp16":
            self.logger.info(f"FP16 accumulation kernel with group size {self.accumulation_group_size} is used")
            for n, m in self.model.named_modules():
                if isinstance(m, (LlamaDecoderLayer, Qwen2DecoderLayer)):
                    for name, mod in m.named_modules():
                        if isinstance(mod, _QBaseLinear):
                            mod.ops = FP16AccumMatMul(group_size=self.accumulation_group_size)
        elif self.accumulation_type == "fp16_quant":
            self.logger.info(f"FP16 accumulation kernel with MXINT8 psum quantization, group size {self.accumulation_group_size}, quant block size {self.accumulation_quant_block_size} is used")
            for n, m in self.model.named_modules():
                if isinstance(m, (LlamaDecoderLayer, Qwen2DecoderLayer)):
                    for name, mod in m.named_modules():
                        if isinstance(mod, _QBaseLinear):
                            mod.ops = FP16AccumMatMul(group_size=self.accumulation_group_size, quant_block_size=self.accumulation_quant_block_size)
        else:
            raise ValueError(f"Invalid accumulation type: {self.accumulation_type}")

    def log_dict_tree(self, d, indent=0, level="info"):
        """
        Recursively logs a nested dictionary with hierarchical formatting.
        logger: Python logging.Logger instance
        d: dict to log
        indent: starting indentation
        level: logging level name as string (e.g., "info", "debug", "warning")
        """
        log_fn = getattr(self.logger, level)

        for key, value in d.items():
            prefix = " " * indent
            if isinstance(value, dict):
                log_fn(f"{prefix}{key}:")
                self.log_dict_tree(value, indent + 4, level)
            else:
                log_fn(f"{prefix}{key}: {value}")

    def run(self):
        self.logger.info("Quantization config:")
        self.log_dict_tree(self.quant_config)

        self.logger.info(f"PTQ Start! Inserting quantizers...")
        self.model.eval()
        self.inject_quantizers()
        self.logger.info(f"Static quantizing wgt")
        self.static_quant_wgt()
        self.logger.info(f"Done static quantizing wgt")
        
        self.logger.info(f"Replacing accumulation kernel")
        self.replace_accum_kernel()
        self.logger.info(f"Done replacing accumulation kernel")
        self.logger.info(f"Done PTQ!")
        return self.model
