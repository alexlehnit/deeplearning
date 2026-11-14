import math
import torch
from torch import nn



class LoRALinear(nn.Module):
    
    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.skalierung = alpha / r
        self.A = nn.Parameter(torch.empty(r, original.in_features))
        self.B = nn.Parameter(torch.zeros(original.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.skalierung


ZIELE = {
    "qkv": lambda block: (block.attn, "qkv"),
    "proj": lambda block: (block.attn, "proj"),
    "fc1": lambda block: (block.mlp, "fc1"),
    "fc2": lambda block: (block.mlp, "fc2"),
}
def lora_einbauen(modell, r=8, alpha=16, ziele=("qkv",)):
    for block in modell.blocks:
        for ziel in ziele:
            eltern, name = ZIELE[ziel](block)
            setattr(eltern, name, LoRALinear(getattr(eltern, name), r=r, alpha=alpha))
    return modell