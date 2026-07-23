"""
Patching attention & MLP module to insert quantizers
"""
import torch
import torch.nn as nn

from typing import Tuple
from src.module import QLlamaMLP, QQwen2MLP, QMixtralBlockSparseTop2MLP, QLlamaAttention, QQwen2Attention, QMixtralAttention


from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralBlockSparseTop2MLP


from typing import Union, Dict
def get_parent_name(target:str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

class PatchLlama():
    '''
        Patch Llama (support Llama2, 3)
    '''
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def to_half(self, module:nn.Module):
        for param in module.parameters():
            param.data = param.data.to(torch.float16)

        return module
    
    def attn(self, attn:LlamaAttention):
        new_attn = QLlamaAttention(attn.config, attn.layer_idx).to(next(attn.parameters()).device)
        new_attn.load_state_dict(attn.state_dict(), strict=False)

        new_attn = self.to_half(new_attn)
        return new_attn

    def mlp(self, mlp:LlamaMLP):
        new_module = QLlamaMLP(config=mlp.config).to(next(mlp.parameters()).device)
        new_module = new_module.to(torch.float16)
        new_module.load_state_dict(mlp.state_dict(), strict=False)

        new_module = self.to_half(new_module)
        return new_module

    @torch.inference_mode()
    def patch(self):
        # 1) Collect targets first
        modules = dict(self.model.named_modules(remove_duplicate=True))
        targets = []
        for n, m in modules.items():
            if isinstance(m, (LlamaAttention, LlamaMLP)):
                if "." in n:
                    parent_name, child_name = n.rsplit(".", 1)
                    parent = modules[parent_name]
                else:
                    parent, child_name = self.model, n
                targets.append((parent, child_name, m))

        # 2) Replace using assign=True (no copies; reuse old tensors on their device/dtype)
        for parent, child_name, old in targets:
            if isinstance(old, LlamaAttention):
                new = QLlamaAttention(old.config, old.layer_idx)
            else:  # LlamaMLP
                new = QLlamaMLP(config=old.config)

            # No to_empty; directly rebind parameters/buffers from old module
            new.load_state_dict(old.state_dict(), strict=False, assign=True)
            setattr(parent, child_name, new)

        return self.model

class PatchQwen():
    def __init__(self, model: nn.Module) -> None:
        '''
            Patch Qwen (Support Qwen2.5)
        '''
        self.model = model

    def attn(self, attn:Qwen2Attention):
        new_attn = QQwen2Attention(attn.config, attn.layer_idx).to(self.model.device)
        new_attn.load_state_dict(attn.state_dict(), strict=False)
        new_attn = self.to_half(new_attn)
        return new_attn

    def mlp(self, mlp:Qwen2MLP):
        new_mlp = QQwen2MLP(config=mlp.config).to(self.model.device)
        new_mlp.load_state_dict(mlp.state_dict(), strict=False)
        new_mlp = self.to_half(new_mlp)
        return new_mlp

    @torch.inference_mode()
    def patch(self):
        # 1) Get all targets to replace
        targets = []
        for n, m in self.model.named_modules():
            if isinstance(m, (Qwen2Attention, Qwen2MLP)):
                parent_name, name = get_parent_name(n)
                parent = self.model.get_submodule(parent_name)
                targets.append((parent, name, m))

        # 2) Replace using assign=True (no copies; reuse old tensors on their device/dtype)
        for parent, child_name, old in targets:
            if isinstance(old, Qwen2Attention):
                new = QQwen2Attention(old.config, old.layer_idx)
            else:  # Qwen2MLP
                new = QQwen2MLP(config=old.config)

            # No to_empty; directly rebind parameters/buffers from old module
            new.load_state_dict(old.state_dict(), strict=False, assign=True)
            setattr(parent, child_name, new)

        return self.model

class PatchMixtral():
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.inference_mode()
    def patch(self):
        targets = []
        for n, m in self.model.named_modules():
            if isinstance(m, (MixtralAttention, MixtralBlockSparseTop2MLP)):
                parent_name, name = get_parent_name(n)
                parent = self.model.get_submodule(parent_name)
                targets.append((parent, name, m))

        for parent, child_name, old in targets:
            with torch.device("meta"):
                if isinstance(old, MixtralAttention):
                    new = QMixtralAttention(old.config, old.layer_idx)
                else:
                    new = QMixtralBlockSparseTop2MLP(hidden_dim=old.hidden_dim, ffn_dim=old.ffn_dim, act_fn=old.act_fn)
            new.load_state_dict(old.state_dict(), strict=False, assign=True)
            setattr(parent, child_name, new)

        return self.model

