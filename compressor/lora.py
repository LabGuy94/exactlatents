"""Hand-rolled LoRA used for decoder co-adaptation.

The trainer compiles the decoder trunk, chunks the loss, and maintains its own
EMA and checkpoint state over the trainable parameters. Keeping the small LoRA
mechanism local avoids adding wrapper indirection to those paths.

For a frozen linear weight W, a low-rank residual BA makes the layer compute
``y = Wx + (alpha/r) * B(Ax)``. A is initialized like ``nn.Linear`` and B is
zero-initialized, so the decoder initially matches the frozen base model.

A and B use fp32, like the projector, while the frozen model remains bf16.
"""

import math

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base  # frozen; requires_grad already False
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        lo = nn.functional.linear(nn.functional.linear(x.float(), self.lora_A), self.lora_B)
        return y + (self.scale * lo).to(y.dtype)


def inject_lora(model: nn.Module, r: int, alpha: float, targets=TARGETS):
    """Replace every targeted nn.Linear in-place. Returns the new module paths."""
    sites = [(parent, name) for parent in model.modules()
             for name, child in parent.named_children()
             if name in targets and isinstance(child, nn.Linear)]
    for parent, name in sites:
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return [f"{type(parent).__name__}.{name}" for parent, name in sites]


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state_dict(model: nn.Module, sd):
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, f"unexpected keys in lora state dict: {res.unexpected_keys[:3]}"
    # missing lora_ keys = incomplete/mismatched adapter loading silently as
    # partial identity (base keys are legitimately absent from an adapter sd)
    missing_lora = [k for k in res.missing_keys if "lora_" in k]
    assert not missing_lora, f"adapter tensors missing from state dict: {missing_lora[:3]}"
