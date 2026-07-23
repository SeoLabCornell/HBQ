import torch


def patch_torch_float8_compat():
    """Provide dtype aliases expected by newer Transformers on older Torch builds."""
    if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fnuz"):
        torch.float8_e8m0fnu = torch.float8_e4m3fnuz
