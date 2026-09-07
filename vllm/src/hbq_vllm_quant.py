"""vLLM quantization plugin for HBQ direct block fake quantization.

This prototype targets W4A4 NVFP4 parity with HBQ-llm's fake-quantized
Transformers path and includes optional post-RoPE K/V fake quantization before
vLLM writes tensors into its KV cache.
"""

from __future__ import annotations

from typing import Any

import json
import os
from pathlib import Path

import torch

from src.quantizer import HBQQuantizer, MXFPQuantizer

_ACTIVE_QUANT_CONFIG: dict[str, Any] | None = None
_DEBUG_ATTENTION_CALLS = 0
_AWQ_CACHE: dict[str, dict[str, Any]] = {}

try:
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization import (
        register_quantization_config,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
except ImportError as exc:  # pragma: no cover - exercised only without vLLM.
    raise ImportError(
        "HBQ-vllm requires vLLM. Install the hbq-vllm environment first."
    ) from exc


def _require_supported_qtype(kind: str, qtype: str) -> None:
    if qtype not in {"mxfp", "hbq"}:
        raise ValueError(
            f"HBQ vLLM prototype only supports {kind}=mxfp or hbq, got {qtype!r}"
        )


def _build_quantizer(
    quant_config: dict[str, Any],
    prefix: str,
    qtype: str | None = None,
    force_4bit: bool = False,
) -> MXFPQuantizer:
    block_cfg = quant_config.get("block_quant", {})
    quantization = quant_config.get("quantization", {})
    qtype = qtype or quantization.get("xqtype" if prefix == "act" else "wqtype", "mxfp")
    _require_supported_qtype(prefix, qtype)

    ebit = 2 if force_4bit else block_cfg[f"{prefix}_ebit"]
    mbit = 1 if force_4bit else block_cfg[f"{prefix}_mbit"]
    common = {
        "block_size": block_cfg[f"{prefix}_block_size"],
        "ebit": ebit,
        "mbit": mbit,
        "sc_ebit": block_cfg[f"{prefix}_sc_ebit"],
        "sc_mbit": block_cfg[f"{prefix}_sc_mbit"],
        "per_tensor_scale": block_cfg.get(f"{prefix}_per_tensor_scale", False),
        "scale_allow_subnormal": block_cfg.get("scale_allow_subnormal", False),
        "use_ceil": block_cfg.get(f"{prefix}_scale_use_ceil", False),
    }
    if qtype == "mxfp":
        return MXFPQuantizer(**common, activate_dynamo=False)

    return HBQQuantizer(
        **common,
        l2_block_size=block_cfg[f"{prefix}_l2_block_size"],
        l2_sc_bit=block_cfg[f"{prefix}_l2_sc_bit"],
        l2_scheme=block_cfg[f"{prefix}_l2_scheme"],
        use_quant_kernel=False,
    )


def set_active_quant_config(quant_config: dict[str, Any]) -> None:
    """Set the YAML quant config used when vLLM instantiates the plugin."""
    global _ACTIVE_QUANT_CONFIG
    _ACTIVE_QUANT_CONFIG = quant_config
    # vLLM may construct quantization config objects inside worker processes.
    # Keep a serialized copy in the environment so those workers do not fall
    # back to default_nvfp4_config().
    os.environ["HBQ_VLLM_QUANT_CONFIG_JSON"] = json.dumps(quant_config)


def _active_quant_config() -> dict[str, Any]:
    if _ACTIVE_QUANT_CONFIG is not None:
        return _ACTIVE_QUANT_CONFIG
    raw = os.environ.get("HBQ_VLLM_QUANT_CONFIG_JSON")
    if raw:
        return json.loads(raw)
    return default_nvfp4_config()


def _active_quantization() -> dict[str, Any]:
    return _active_quant_config().get("quantization", {})


def _resolve_config_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    root = Path(__file__).resolve().parents[1]
    rooted = root / candidate
    if rooted.exists():
        return str(rooted)
    return path


def _awq_enabled() -> bool:
    return bool(_active_quant_config().get("awq", {}).get("enable", False))


def _hf_to_vllm_awq_target(name: str) -> str:
    if (
        name.endswith(".self_attn.q_proj")
        or name.endswith(".self_attn.k_proj")
        or name.endswith(".self_attn.v_proj")
    ):
        return name.rsplit(".", 1)[0].rsplit(".", 1)[0] + ".self_attn.qkv_proj"
    if name.endswith(".mlp.gate_proj") or name.endswith(".mlp.up_proj"):
        return name.rsplit(".", 1)[0].rsplit(".", 1)[0] + ".mlp.gate_up_proj"
    return name


def _load_awq_metadata() -> dict[str, Any] | None:
    if not _awq_enabled():
        return None

    awq_cfg = _active_quant_config().get("awq", {})
    load_path = awq_cfg.get("load_path") or awq_cfg.get("dump_path")
    if not load_path:
        raise ValueError("awq.enable is true but neither awq.load_path nor awq.dump_path is set")

    resolved = _resolve_config_path(str(load_path))
    cached = _AWQ_CACHE.get(resolved)
    if cached is not None:
        return cached

    raw = torch.load(resolved, map_location="cpu")
    scales: dict[str, torch.Tensor] = {}
    for _prev_op_name, layer_names, scale in raw.get("scale", []):
        for layer_name in layer_names:
            scales[_hf_to_vllm_awq_target(layer_name)] = scale.detach().cpu()

    clips: dict[str, torch.Tensor] = {}
    for layer_name, max_val in raw.get("clip", []):
        clips[layer_name] = max_val.detach().cpu()

    metadata = {"scale": scales, "clip": clips}
    _AWQ_CACHE[resolved] = metadata
    return metadata


def _awq_scale_for_prefix(prefix: str) -> torch.Tensor | None:
    metadata = _load_awq_metadata()
    if metadata is None:
        return None
    return metadata["scale"].get(prefix)


def _clip_weight_inplace(weight: torch.Tensor, max_val: torch.Tensor) -> None:
    max_val = max_val.to(device=weight.device, dtype=weight.dtype)
    org_shape = weight.shape
    if org_shape[0] != max_val.shape[0]:
        raise ValueError(
            f"AWQ clip row mismatch for weight shape {tuple(org_shape)} "
            f"and clip shape {tuple(max_val.shape)}"
        )
    weight_view = weight.data.reshape(*max_val.shape[:2], -1)
    weight_view.copy_(torch.clamp(weight_view, -max_val, max_val))


def _apply_awq_clip_for_prefix(layer: torch.nn.Module, prefix: str) -> None:
    metadata = _load_awq_metadata()
    if metadata is None:
        return

    weight = getattr(layer, "weight", None)
    if weight is None:
        return

    clips: dict[str, torch.Tensor] = metadata["clip"]
    if prefix.endswith(".self_attn.qkv_proj"):
        base = prefix[: -len(".qkv_proj")]
        v_clip = clips.get(base + ".v_proj")
        if v_clip is not None:
            rows = v_clip.shape[0]
            _clip_weight_inplace(weight[-rows:], v_clip)
        return

    if prefix.endswith(".mlp.gate_up_proj"):
        base = prefix[: -len(".gate_up_proj")]
        gate_clip = clips.get(base + ".gate_proj")
        up_clip = clips.get(base + ".up_proj")
        offset = 0
        if gate_clip is not None:
            rows = gate_clip.shape[0]
            _clip_weight_inplace(weight[offset : offset + rows], gate_clip)
            offset += rows
        if up_clip is not None:
            rows = up_clip.shape[0]
            _clip_weight_inplace(weight[offset : offset + rows], up_clip)
        return

    clip = clips.get(prefix)
    if clip is not None:
        _clip_weight_inplace(weight, clip)


def _debug_attention_dump(call_idx: int, tensors: dict[str, torch.Tensor]) -> None:
    dump_dir = os.environ.get("HBQ_DEBUG_ATTENTION_DIR")
    if not dump_dir:
        return
    target_tokens = os.environ.get("HBQ_DEBUG_ATTENTION_TOKENS")
    if target_tokens is not None:
        hidden_states = tensors.get("hidden_states")
        if hidden_states is None or hidden_states.shape[0] != int(target_tokens):
            return
    max_calls = int(os.environ.get("HBQ_DEBUG_ATTENTION_CALLS", "1"))
    existing = len([p for p in os.listdir(dump_dir) if p.startswith("vllm_attn_call_")]) if os.path.isdir(dump_dir) else 0
    if existing >= max_calls:
        return
    os.makedirs(dump_dir, exist_ok=True)
    payload = {
        name: tensor.detach().float().cpu()
        for name, tensor in tensors.items()
        if tensor is not None
    }
    torch.save(payload, os.path.join(dump_dir, f"vllm_attn_call_{call_idx}.pt"))


def _act_quantizer() -> MXFPQuantizer:
    quant_config = _active_quant_config()
    quantization = quant_config.get("quantization", {})
    return _build_quantizer(
        quant_config,
        "act",
        qtype=quantization.get("xqtype", "mxfp"),
        force_4bit=False,
    )


def _kv_quantizer() -> MXFPQuantizer:
    quant_config = _active_quant_config()
    quantization = quant_config.get("quantization", {})
    return _build_quantizer(
        quant_config,
        "act",
        qtype=quantization.get("xqtype", "mxfp"),
        force_4bit=quantization.get("quant_4b_kv", False),
    )


def _weight_quant_chunk_rows(quant_config: dict[str, Any]) -> int:
    block_cfg = quant_config.get("block_quant", {})
    return int(block_cfg.get("wgt_quant_chunk_rows", os.environ.get("HBQ_WGT_QUANT_CHUNK_ROWS", 1024)))


def _quantize_weight_in_chunks(
    weight: torch.Tensor,
    quantizer: MXFPQuantizer,
    chunk_rows: int,
) -> torch.Tensor:
    # Weight block quantization is along the last dimension, so slicing output
    # rows preserves per-row block boundaries while reducing peak temporary
    # memory for large 70B projections.
    if weight.dim() < 2 or chunk_rows <= 0 or weight.shape[0] <= chunk_rows:
        return quantizer.q(weight).to(dtype=weight.dtype)

    out = torch.empty_like(weight)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        out[start:end].copy_(quantizer.q(weight[start:end]).to(dtype=weight.dtype))
        if weight.is_cuda:
            torch.cuda.empty_cache()
    return out


def _quantize_post_rope_q(q: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    # HBQ-llm uses q_proj.aq for post-RoPE Q, not the 4-bit KV quantizer.
    orig_shape = q.shape
    q_view = q.reshape(-1, num_heads, head_dim)
    q_q = _act_quantizer().q(q_view)
    return q_q.reshape(orig_shape).to(dtype=q.dtype)


def _quantize_k_cache(k: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    # vLLM LlamaAttention uses flattened [tokens, num_kv_heads * head_dim].
    # HBQ-llm quantizes K along head_dim after RoPE.
    orig_shape = k.shape
    k_view = k.reshape(-1, num_kv_heads, head_dim)
    k_q = _kv_quantizer().q(k_view)
    return k_q.reshape(orig_shape).to(dtype=k.dtype)


def _sequence_spans_from_positions(positions: torch.Tensor, n_tokens: int) -> list[tuple[int, int]]:
    if positions.numel() != n_tokens:
        return [(0, n_tokens)]

    pos = positions.detach().to(device="cpu", dtype=torch.long).flatten()
    starts = [0]
    for idx in range(1, pos.numel()):
        if int(pos[idx].item()) != int(pos[idx - 1].item()) + 1:
            starts.append(idx)
    starts.append(n_tokens)
    return list(zip(starts, starts[1:]))


def _quantize_v_cache(
    v: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    # HBQ-llm quantizes V in [batch, kv_heads, head_dim, tokens] layout, so
    # each sequence gets independent token-axis blocks/scales. vLLM flattens
    # all scheduled tokens; split by position resets before quantizing V.
    orig_shape = v.shape
    v_view = v.reshape(-1, num_kv_heads, head_dim)
    spans = _sequence_spans_from_positions(positions, v_view.shape[0]) if positions is not None else [(0, v_view.shape[0])]
    quantizer = _kv_quantizer()

    out = torch.empty_like(v_view)
    for left, right in spans:
        if right <= left:
            continue
        seq = v_view[left:right].permute(1, 2, 0).contiguous()
        seq_q = quantizer.q(seq)
        out[left:right].copy_(seq_q.permute(2, 0, 1).contiguous().to(dtype=v.dtype))
    return out.reshape(orig_shape).to(dtype=v.dtype)


def _center_qwen_k_cache(
    attn_module: torch.nn.Module,
    k: torch.Tensor,
    positions: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    # HBQ-llm centers Qwen K over the prefill token dimension and reuses the
    # same center for later decode tokens from that sequence. vLLM gives this
    # hook flattened token batches but not sequence ids, so track centers by the
    # next position that should consume them. Queues handle batched requests that
    # share the same decode position.
    if positions.numel() != k.shape[0]:
        return k

    if not hasattr(attn_module, "_hbq_qwen_k_center_by_pos"):
        attn_module._hbq_qwen_k_center_by_pos = {}

    center_by_pos: dict[int, list[torch.Tensor]] = attn_module._hbq_qwen_k_center_by_pos
    pos = positions.detach().to(device="cpu", dtype=torch.long).flatten()
    k_view = k.reshape(-1, num_kv_heads, head_dim)
    centered = k_view.clone()
    changed = False

    starts = [0]
    for idx in range(1, pos.numel()):
        if int(pos[idx].item()) != int(pos[idx - 1].item()) + 1:
            starts.append(idx)
    starts.append(pos.numel())

    for left, right in zip(starts, starts[1:]):
        first_pos = int(pos[left].item())
        last_pos = int(pos[right - 1].item())
        span_len = right - left

        if span_len > 1:
            center = k_view[left:right].mean(dim=0, keepdim=True).detach()
            centered[left:right] = k_view[left:right] - center
            center_by_pos.setdefault(last_pos + 1, []).append(center)
            changed = True
            continue

        queue = center_by_pos.get(first_pos)
        if queue:
            center = queue.pop(0)
            if not queue:
                center_by_pos.pop(first_pos, None)
            centered[left:right] = k_view[left:right] - center.to(device=k_view.device, dtype=k_view.dtype)
            center_by_pos.setdefault(first_pos + 1, []).append(center)
            changed = True

    if not changed:
        return k
    return centered.reshape_as(k).to(dtype=k.dtype)


def _hbq_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    qwen: bool,
) -> torch.Tensor:
    global _DEBUG_ATTENTION_CALLS
    call_idx = _DEBUG_ATTENTION_CALLS
    _DEBUG_ATTENTION_CALLS += 1

    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_pre_rope, k_pre_rope, v_pre_quant = q.clone(), k.clone(), v.clone()

    if qwen and getattr(self, "qk_norm", False):
        total_tokens = q.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(total_tokens, self.q_size)
        k = k.view(total_tokens, self.kv_size)

    q, k = self.rotary_emb(positions, q, k)
    q_post_rope, k_post_rope = q.clone(), k.clone()

    quantization = _active_quantization()
    if quantization.get("quant_post_rope_q", False):
        q = _quantize_post_rope_q(q, self.num_heads, self.head_dim)
    if quantization.get("quant_k_cache", False):
        if qwen:
            k = _center_qwen_k_cache(self, k, positions, self.num_kv_heads, self.head_dim)
        k = _quantize_k_cache(k, self.num_kv_heads, self.head_dim)
    if quantization.get("quant_v_cache", False):
        v = _quantize_v_cache(v, self.num_kv_heads, self.head_dim, positions)

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    _debug_attention_dump(
        call_idx,
        {
            "hidden_states": hidden_states,
            "q_pre_rope": q_pre_rope,
            "k_pre_rope": k_pre_rope,
            "v_pre_quant": v_pre_quant,
            "q_post_rope": q_post_rope,
            "k_post_rope": k_post_rope,
            "q_quant": q,
            "k_quant": k,
            "v_quant": v,
            "attn_output": attn_output,
            "output": output,
            "qkv_weight": getattr(self.qkv_proj, "weight", None),
        },
    )
    return output


def _patch_attention_class(attn_cls: type, *, qwen: bool) -> None:
    if getattr(attn_cls, "_hbq_kv_patch", False):
        return

    original_forward = attn_cls.forward

    def hbq_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return _hbq_attention_forward(self, positions, hidden_states, qwen=qwen)

    attn_cls._hbq_original_forward = original_forward
    attn_cls.forward = hbq_forward
    attn_cls._hbq_kv_patch = True


def patch_llama_attention_for_kv_quant() -> None:
    try:
        from vllm.model_executor.models.llama import LlamaAttention
    except Exception:
        return

    _patch_attention_class(LlamaAttention, qwen=False)


def patch_qwen2_attention_for_kv_quant() -> None:
    try:
        from vllm.model_executor.models.qwen2 import Qwen2Attention
    except Exception:
        return

    _patch_attention_class(Qwen2Attention, qwen=True)


patch_llama_attention_for_kv_quant()
patch_qwen2_attention_for_kv_quant()


class HBQNVFP4LinearMethod(UnquantizedLinearMethod):
    """Fake-quantize vLLM linear inputs and loaded weights with HBQ quantizers."""

    def __init__(self, quant_config: dict[str, Any], prefix: str) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix

    def _act_quantizer(self) -> MXFPQuantizer:
        return _build_quantizer(self.quant_config, "act", qtype=self.quant_config.get("quantization", {}).get("xqtype", "mxfp"))

    def _wgt_quantizer(self) -> MXFPQuantizer:
        return _build_quantizer(self.quant_config, "wgt", qtype=self.quant_config.get("quantization", {}).get("wqtype", "mxfp"))

    @torch.inference_mode()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        process = getattr(super(), "process_weights_after_loading", None)
        if process is not None:
            process(layer)

        if getattr(layer, "hbq_weight_quantized", False):
            return
        if os.environ.get("HBQ_DEBUG_SKIP_WGT_QUANT") == "1":
            return

        weight = getattr(layer, "weight", None)
        if weight is None:
            return

        awq_scale = _awq_scale_for_prefix(self.prefix)
        if awq_scale is not None:
            if awq_scale.numel() != weight.shape[1]:
                raise ValueError(
                    f"AWQ scale width mismatch for {self.prefix}: "
                    f"scale={tuple(awq_scale.shape)}, weight={tuple(weight.shape)}"
                )
            scale_view = awq_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)
            weight.data.mul_(scale_view)

        _apply_awq_clip_for_prefix(layer, self.prefix)

        quantizer = self._wgt_quantizer()
        weight.data = _quantize_weight_in_chunks(
            weight.data,
            quantizer,
            _weight_quant_chunk_rows(self.quant_config),
        )
        setattr(layer, "hbq_weight_quantized", True)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        awq_scale = _awq_scale_for_prefix(self.prefix)
        apply_act_scale = self.quant_config.get("awq", {}).get("apply_act_scale", True)
        if awq_scale is not None and apply_act_scale:
            scale_view = awq_scale.to(device=x.device, dtype=x.dtype).view(*([1] * (x.dim() - 1)), -1)
            x = x / scale_view
        xq = self._act_quantizer().q(x)
        return super().apply(layer, xq, bias)


@register_quantization_config("hbq_nvfp4")
class HBQNVFP4Config(QuantizationConfig):
    """vLLM config wrapper for HBQ NVFP4 fake quantization."""

    def __init__(self, quant_config: dict[str, Any] | None = None) -> None:
        self.quant_config = quant_config or _active_quant_config()
        quantization = self.quant_config.get("quantization", {})
        _require_supported_qtype("xqtype", quantization.get("xqtype", "mxfp"))
        _require_supported_qtype("wqtype", quantization.get("wqtype", "mxfp"))
        unsupported = [
            "quant_attn_wgt",
            "quant_output_layer",
        ]
        enabled = [name for name in unsupported if quantization.get(name, False)]
        if enabled:
            raise NotImplementedError(
                "HBQ-vllm direct PTQ currently supports linear W/A quantization "
                "and optional K/V cache fake quantization. "
                f"Unsupported enabled flags: {enabled}"
            )

    def get_name(self) -> str:
        return "hbq_nvfp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HBQNVFP4Config":
        if config:
            return cls(config)
        return cls(_active_quant_config())

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if "lm_head" in prefix:
            return None
        if isinstance(layer, LinearBase):
            return HBQNVFP4LinearMethod(self.quant_config, prefix)
        return None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if user_quant == "hbq_nvfp4":
            return "hbq_nvfp4"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


def default_nvfp4_config() -> dict[str, Any]:
    return {
        "quantization": {
            "xqtype": "mxfp",
            "wqtype": "mxfp",
            "quant_before_hadamard": False,
            "quant_before_silu": False,
            "quant_before_rms_norm": False,
            "quant_output_layer": False,
            "quant_post_rope_q": False,
            "quant_4b_kv": False,
            "quant_k_cache": False,
            "quant_v_cache": False,
            "quant_attn_wgt": False,
        },
        "block_quant": {
            "act_ebit": 2,
            "act_mbit": 1,
            "act_block_size": 16,
            "act_sc_ebit": 5,
            "act_sc_mbit": 3,
            "wgt_ebit": 2,
            "wgt_mbit": 1,
            "wgt_block_size": 16,
            "wgt_sc_ebit": 5,
            "wgt_sc_mbit": 3,
            "wgt_per_tensor_scale": False,
            "act_per_tensor_scale": False,
            "scale_allow_subnormal": False,
            "act_scale_use_ceil": False,
            "wgt_scale_use_ceil": False,
            "act_l2_sc_bit": 2,
            "act_l2_block_size": 8,
            "act_l2_scheme": "SIG-1",
            "wgt_l2_sc_bit": 2,
            "wgt_l2_block_size": 8,
            "wgt_l2_scheme": "Mix",
        },
    }
